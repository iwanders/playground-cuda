use cudarc::driver::{CudaDevice, DevicePtr, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// We don't actually have any headers, but lets add them anyway.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Cuda PTX file is generated from the .cu file at build time.
const CUDA_KERNEL_MWC: &str = include_str!(concat!(env!("OUT_DIR"), "/mwc_cuda.ptx"));

/// Helper struct for cuda operations around the MWC rng.
pub struct MultiplyWithCarryCuda {
    gpu: Arc<CudaDevice>,
}

pub type Error = DriverError;

impl MultiplyWithCarryCuda {
    /// Compile the kernel and instantiate the device.
    pub fn new() -> Result<Self, Error> {
        let gpu = CudaDevice::new(0)?;

        let ptx = Ptx::from_src(CUDA_KERNEL_MWC);

        gpu.load_ptx(
            ptx,
            "mwc_module",
            &["mwc_store_output_kernel", "mwc_find_seed_kernel"],
        )?;
        Ok(Self { gpu })
    }

    /// Calculate the output vectors given the provided inputs.
    pub fn store_outputs(
        &self,
        init: &[(u32 /* value */, u32 /*carry*/)],
        factor: u32,
        advances: usize,
    ) -> Result<Vec<Vec<u32>>, Error> {
        // Convert inputs to gpu vectors.
        let value_init: Vec<u32> = init.iter().map(|(value, _carry)| *value).collect();
        let carry_init: Vec<u32> = init.iter().map(|(_value, carry)| *carry).collect();
        let value_gpu = self.gpu.htod_copy(value_init)?;
        let carry_gpu = self.gpu.htod_copy(carry_init)?;

        // Create the output vectors.
        let mut output_gpu = vec![];
        for _ in 0..init.len() {
            output_gpu.push(self.gpu.alloc_zeros::<u32>(advances)?);
        }

        // Create the output vector of points to vectors of values.
        let output_gpu_vec = output_gpu.iter().map(|v| *v.device_ptr()).collect();
        let output_gpu_slice = self.gpu.htod_copy(output_gpu_vec)?;

        // Get the appropriate kernel.
        let f = self
            .gpu
            .get_func("mwc_module", "mwc_store_output_kernel")
            .unwrap();

        // Start the kernel
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(init.len() as u32),
                (
                    factor,
                    init.len(),
                    &carry_gpu,
                    &value_gpu,
                    &output_gpu_slice,
                    advances,
                ),
            )
        }?;

        // Wait for it to complete.
        self.gpu.synchronize()?;

        // Copy the outputs back to the host
        let mut output = vec![];
        for out_gpu in output_gpu {
            output.push(self.gpu.dtoh_sync_copy(&out_gpu)?);
        }

        Ok(output)
    }

    pub fn find_seed(
        &self,
        factor: u32,
        init_limit: usize,
        carry_init: u32,
        advance_limit: u32,
        modulo: u32,
        expected: &[u32],
    ) -> Result<Vec<u32>, Error> {
        // __restrict__ std::uint32_t factor_in,
        // __restrict__ std::size_t init_addition,
        // __restrict__ std::size_t init_limit,
        // __restrict__ std::uint32_t carry_init,
        // __restrict__ std::uint32_t advance_limit,
        // __restrict__ std::uint32_t modulo,
        // __restrict__ std::uint32_t* expected,
        // __restrict__ std::size_t expected_count,
        // __restrict__ std::uint32_t* output_matches,
        // __restrict__ std::size_t output_in,
        // __restrict__ std::uint32_t* output_found // handled atomically.

        let expected: Vec<u32> = expected.to_vec();
        let expected_len = expected.len();
        let expected_gpu = self.gpu.htod_copy(expected)?;

        const OUTPUT_IN: usize = 64;
        let output_matches = self.gpu.alloc_zeros::<u32>(OUTPUT_IN)?;
        let output_found = self.gpu.alloc_zeros::<u32>(1)?;

        // Get the appropriate kernel.
        let f = self
            .gpu
            .get_func("mwc_module", "mwc_find_seed_kernel")
            .unwrap();

        // Start the kernel
        unsafe {
            f.launch(
                LaunchConfig::for_num_elems(init_limit as u32),
                (
                    factor,
                    0, // init addition
                    init_limit,
                    carry_init,
                    advance_limit,
                    modulo,
                    &expected_gpu,
                    expected_len,
                    &output_matches,
                    OUTPUT_IN,
                    &output_found,
                ),
            )
        }?;

        // Wait for it to complete.
        self.gpu.synchronize()?;

        let value: Vec<u32> = self.gpu.dtoh_sync_copy(&output_found)?;
        let count = value[0];

        // Copy the outputs back to the host
        let mut matching = self.gpu.dtoh_sync_copy(&output_matches)?;
        matching.truncate(count as usize);
        Ok(matching)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_cuda_mwc() -> Result<(), Error> {
        let mwc_cuda = MultiplyWithCarryCuda::new()?;
        let v = mwc_cuda.store_outputs(&[(1, 333 * 2), (1, 333 * 2)], 1791398085, 5)?;
        let expected = [0x6AC6935F, 0x2F2ED81B, 0x280687C4, 0xB6AAB839, 0xBFC793C3];
        for k in 0..v.len() {
            for (i, value) in expected.iter().enumerate() {
                // println!("Value from cuda: {value:0>8x}");
                assert_eq!(v[k][i], *value);
            }
        }
        Ok(())
    }
    #[test]
    fn test_cuda_seed_find() -> Result<(), Error> {
        let mwc_cuda = MultiplyWithCarryCuda::new()?;
        let l = 150;
        let h = 500;
        let modulo = h - l;
        let expected = [
            418 - l,
            363 - l,
            274 - l,
            348 - l,
            162 - l,
            219 - l,
            282 - l,
        ]; // seed 65536
        let init_limit = 1 << 24;
        let carry_init = 333 * 2;
        let advance_limit = 1024;
        let found_seeds = mwc_cuda.find_seed(
            1791398085,
            init_limit,
            carry_init,
            advance_limit,
            modulo,
            &expected,
        )?;
        println!("found_seeds: {found_seeds:?}");
        assert_eq!(found_seeds.len(), 1);
        assert_eq!(found_seeds[0], 65536);
        Ok(())
    }
}
