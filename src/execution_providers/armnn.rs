use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "armnn"))]
extern "C" {
	fn OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct ArmNNExecutionProvider {
	use_arena: bool
}

impl ArmNNExecutionProvider {
	#[must_use]
	pub fn with_arena_allocator(mut self) -> Self {
		self.use_arena = true;
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "armnn"))]
		{
			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> ort_sys::OrtStatusPtr);
			return crate::error::status_to_result(unsafe {
				OrtSessionOptionsAppendExecutionProvider_ArmNN(session_builder.session_options_ptr.as_ptr(), self.use_arena.into())
			})
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
