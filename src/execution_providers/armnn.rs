use alloc::format;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "armnn"))]
extern "C" {
	fn OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct ArmNNExecutionProvider {
	use_arena: bool
}

impl ArmNNExecutionProvider {
	#[must_use]
	pub fn with_arena_allocator(mut self, enable: bool) -> Self {
		self.use_arena = enable;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<ArmNNExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: ArmNNExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for ArmNNExecutionProvider {
	fn as_str(&self) -> &'static str {
		"ArmNNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", any(target_os = "linux", target_os = "android")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "armnn"))]
		{
			use crate::AsPointer;

			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_ArmNN(session_builder.ptr_mut(), self.use_arena.into())) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
