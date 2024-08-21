use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
const CUDA_KERNEL_MWC: &str = include_str!(concat!(env!("OUT_DIR"), "/mwc_cuda.ptx"));

pub struct MultiplyWithCarryCuda{
    gpu: Arc<CudaDevice>,
}


pub type Error = Box<dyn std::error::Error + Send + Sync>;

impl MultiplyWithCarryCuda {
    pub fn new() -> Result<Self, Error> {
        let gpu = CudaDevice::new(0)?;

        let ptx = Ptx::from_src(CUDA_KERNEL_MWC);
        gpu.load_ptx(ptx, "mwc_module", &["mwc_store_output_kernel", "mwc_find_seed_kernel"])?;
        Ok(Self{
            gpu
        })
    }

    /*
        mwc_store_output_kernel(
        __restrict__ std::uint32_t factor_in,
        std::size_t count_in,
        __restrict__ std::uint32_t* carry_init,
        __restrict__ std::uint32_t* value_init,
        __restrict__ std::uint32_t** out,
        std::size_t advances)
    */
    pub fn store_outputs(&self, init: &[(u32 /* value */, u32 /*carry*/)], factor: u32, advances: usize) -> Result<Vec<Vec<u32>>, Error> {
        let value_init: Vec<u32> = init.iter().map(|(value, _carry)| *value).collect();
        let carry_init: Vec<u32> = init.iter().map(|(_value, carry)| *carry).collect();
        let value_gpu = self.gpu.htod_copy(value_init)?;
        let carry_gpu = self.gpu.htod_copy(carry_init)?;

        use cudarc::driver::CudaSlice;

        let mut output_gpu = vec![];
        for _ in 0..init.len() {
            let a: Vec<u32> = vec![0u32; advances];
            let carry_gpu = self.gpu.htod_copy(a)?;
            output_gpu.push(carry_gpu);
        }

        use cudarc::driver::sys::CUdeviceptr;
        use cudarc::driver::{DevicePtr};
        let mut output_gpu_vec : Vec< CUdeviceptr > = vec![];

        output_gpu_vec = output_gpu.iter().map(|v| *v.device_ptr()).collect();
        let output_gpu_slice = self.gpu.htod_copy(output_gpu_vec)?;
        

        // let output_gpu = self.gpu.htod_copy(output)?;
        let f = self.gpu.get_func("mwc_module", "mwc_store_output_kernel").unwrap();

        unsafe { f.launch(LaunchConfig::for_num_elems(init.len() as u32), (factor, init.len(), &carry_gpu, &value_gpu, &output_gpu_slice, advances)) }?;

        self.gpu.synchronize()?;
        todo!()
    }
}

#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn test_mwc_cuda() -> Result<(), Error> {
        let mut mwc_cuda = MultiplyWithCarryCuda::new()?;
        let v = mwc_cuda.store_outputs(&[(1, 333*2)], 1791398085, 5)?;

        Ok(())
    }
}

