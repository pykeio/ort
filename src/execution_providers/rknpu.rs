use alloc::format;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "rknpu"))]
extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_RKNPU(options: *mut ort_sys::OrtSessionOptions) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct RKNPUExecutionProvider {}

impl RKNPUExecutionProvider {
	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<RKNPUExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: RKNPUExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for RKNPUExecutionProvider {
	fn as_str(&self) -> &'static str {
		"RKNPUExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", target_os = "linux"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "rknpu"))]
		{
			use crate::AsPointer;

			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_RKNPU(options: *mut ort_sys::OrtSessionOptions) -> ort_sys::OrtStatusPtr);
			return unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_RKNPU(session_builder.ptr_mut())) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
