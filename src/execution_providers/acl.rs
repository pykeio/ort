use alloc::format;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "acl"))]
extern "C" {
	fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct ACLExecutionProvider {
	use_arena: bool
}

impl ACLExecutionProvider {
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "acl"))]
		{
			use crate::AsPointer;

			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_ACL(session_builder.ptr_mut(), self.use_arena.into())) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
