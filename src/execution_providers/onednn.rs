use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "onednn"))]
extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Dnnl(
		options: *mut ort_sys::OrtSessionOptions,
		use_arena: std::os::raw::c_int
	) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct OneDNNExecutionProvider {
	use_arena: bool
}

impl OneDNNExecutionProvider {
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

impl From<OneDNNExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: OneDNNExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for OneDNNExecutionProvider {
	fn as_str(&self) -> &'static str {
		"DnnlExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", any(target_os = "windows", target_os = "linux")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "onednn"))]
		{
			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut ort_sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> ort_sys::OrtStatusPtr);
			return crate::error::status_to_result(unsafe {
				OrtSessionOptionsAppendExecutionProvider_Dnnl(session_builder.session_options_ptr.as_ptr(), self.use_arena.into())
			})
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
