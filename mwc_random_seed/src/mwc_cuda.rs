use cudarc::driver::{CudaDevice, LaunchConfig, DeviceRepr, DriverError, LaunchAsync};
use cudarc::nvrtc::Ptx;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
const CUDA_KERNEL_MWC: &str = include_str!(concat!(env!("OUT_DIR"), "/mwc_cuda.ptx"));
