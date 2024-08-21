use cudarc::driver::{CudaDevice, LaunchConfig, DriverError, LaunchAsync, DevicePtr};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// We don't actually have any headers, but lets add them anyway.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Cuda PTX file is generated from the .cu file at build time.
const CUDA_KERNEL_MWC: &str = include_str!(concat!(env!("OUT_DIR"), "/mwc_cuda.ptx"));

/// Helper struct for cuda operations around the MWC rng.
pub struct MultiplyWithCarryCuda{
    gpu: Arc<CudaDevice>,
}


pub type Error = DriverError;

impl MultiplyWithCarryCuda {
    /// Compile the kernel and instantiate the device.
    pub fn new() -> Result<Self, Error> {
        let gpu = CudaDevice::new(0)?;

        let ptx = Ptx::from_src(CUDA_KERNEL_MWC);

        gpu.load_ptx(ptx, "mwc_module", &["mwc_store_output_kernel", "mwc_find_seed_kernel"])?;
        Ok(Self{
            gpu
        })
    }

    /// Calculate the output vectors given the provided inputs.
    pub fn store_outputs(&self, init: &[(u32 /* value */, u32 /*carry*/)], factor: u32, advances: usize) -> Result<Vec<Vec<u32>>, Error> {
        // Convert inputs to gpu vectors.
        let value_init: Vec<u32> = init.iter().map(|(value, _carry)| *value).collect();
        let carry_init: Vec<u32> = init.iter().map(|(_value, carry)| *carry).collect();
        let value_gpu = self.gpu.htod_copy(value_init)?;
        let carry_gpu = self.gpu.htod_copy(carry_init)?;


        // Create the output vectors.
        let mut output_gpu = vec![];
        for _ in 0..init.len() {
            let a: Vec<u32> = vec![0u32; advances];
            let carry_gpu = self.gpu.htod_copy(a)?;
            output_gpu.push(carry_gpu);
        }

        // Create the output vector of points to vectors of values.
        let output_gpu_vec = output_gpu.iter().map(|v| *v.device_ptr()).collect();
        let output_gpu_slice = self.gpu.htod_copy(output_gpu_vec)?;
        
        // Get the appropriate kernel.
        let f = self.gpu.get_func("mwc_module", "mwc_store_output_kernel").unwrap();

        // Start the kernel
        unsafe { f.launch(LaunchConfig::for_num_elems(init.len() as u32), (factor, init.len(), &carry_gpu, &value_gpu, &output_gpu_slice, advances)) }?;

        // Wait for it to complete.
        self.gpu.synchronize()?;

        // Copy the outputs back to the host
        let mut output = vec![];
        for out_gpu in output_gpu {
            output.push(self.gpu.dtoh_sync_copy(&out_gpu)?);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod test{
    use super::*;
    #[test]
    fn test_mwc_cuda() -> Result<(), Error> {
        let mut mwc_cuda = MultiplyWithCarryCuda::new()?;
        let v = mwc_cuda.store_outputs(&[(1, 333*2), (1, 333*2)], 1791398085, 5)?;
        let expected = [0x6AC6935F, 0x2F2ED81B, 0x280687C4, 0xB6AAB839, 0xBFC793C3];
        for k in 0..v.len() {
            for (i, value) in expected.iter().enumerate() {
                // println!("Value from cuda: {value:0>8x}");
                assert_eq!(v[k][i], *value);
            };
        }
        Ok(())
    }
}

