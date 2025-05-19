use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct RKNPUExecutionProvider {}

super::impl_ep!(RKNPUExecutionProvider);

impl ExecutionProvider for RKNPUExecutionProvider {
	fn name(&self) -> &'static str {
		"RknpuExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", target_os = "linux"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "rknpu"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_RKNPU(options: *mut ort_sys::OrtSessionOptions) -> ort_sys::OrtStatusPtr);
			return Ok(unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_RKNPU(session_builder.ptr_mut())) }?);
		}

		Err(RegisterError::MissingFeature)
	}
}
