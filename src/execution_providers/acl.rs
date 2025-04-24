use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct ACLExecutionProvider {
	use_arena: bool
}

super::impl_ep!(ACLExecutionProvider);

impl ACLExecutionProvider {
	#[must_use]
	pub fn with_arena_allocator(mut self, enable: bool) -> Self {
		self.use_arena = enable;
		self
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "acl"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return Ok(unsafe {
				crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_ACL(session_builder.ptr_mut(), self.use_arena.into()))
			}?);
		}

		Err(RegisterError::MissingFeature)
	}
}
