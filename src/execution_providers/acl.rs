use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "acl"))]
extern "C" {
	fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct ACLExecutionProvider {
	use_arena: bool
}

impl ACLExecutionProvider {
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

impl From<ACLExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: ACLExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for ACLExecutionProvider {
	fn as_str(&self) -> &'static str {
		"AclExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_arch = "aarch64")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "acl"))]
		{
			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> ort_sys::OrtStatusPtr);
			return crate::error::status_to_result(unsafe {
				OrtSessionOptionsAppendExecutionProvider_ACL(session_builder.session_options_ptr.as_ptr(), self.use_arena.into())
			})
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
