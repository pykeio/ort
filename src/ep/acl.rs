use core::ffi;

use super::ExecutionProvider;
use crate::{
	AsPointer,
	error::{Error, Result},
	session::builder::SessionBuilder
};

/// [Arm Compute Library execution provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html)
/// for ARM platforms.
#[derive(Debug, Default, Clone)]
pub struct ACL {
	fast_math: bool
}

super::impl_ep!(ACL);

impl ACL {
	/// Enable/disable ACL's fast math mode. Enabling can improve performance at the cost of some accuracy for
	/// `MatMul`/`Conv` nodes.
	///
	/// ```
	/// # use ort::{ep, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = ep::ACL::default().with_fast_math(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_fast_math(mut self, enable: bool) -> Self {
		self.fast_math = enable;
		self
	}
}

impl ExecutionProvider for ACL {
	fn name(&self) -> &'static str {
		"ACLExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut ort_sys::OrtSessionOptions, enable_fast_math: ffi::c_int) -> ort_sys::OrtStatusPtr);
		unsafe { Error::result_from_status(OrtSessionOptionsAppendExecutionProvider_ACL(session_builder.ptr_mut(), self.fast_math.into())) }
	}
}
