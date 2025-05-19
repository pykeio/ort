use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [Arm Compute Library execution provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html)
/// for ARM platforms.
#[derive(Debug, Default, Clone)]
pub struct ACLExecutionProvider {
	fast_math: bool
}

super::impl_ep!(ACLExecutionProvider);

impl ACLExecutionProvider {
	/// Enable/disable ACL's fast math mode. Enabling can improve performance at the cost of some accuracy for
	/// `MatMul`/`Conv` nodes.
	///
	/// ```
	/// # use ort::{execution_providers::acl::ACLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = ACLExecutionProvider::default().with_fast_math(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_fast_math(mut self, enable: bool) -> Self {
		self.fast_math = enable;
		self
	}
}

impl ExecutionProvider for ACLExecutionProvider {
	fn name(&self) -> &'static str {
		"ACLExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_arch = "aarch64")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "acl"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, enable_fast_math: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return Ok(unsafe {
				crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_ACL(session_builder.ptr_mut(), self.fast_math.into()))
			}?);
		}

		Err(RegisterError::MissingFeature)
	}
}
