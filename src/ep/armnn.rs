#![allow(deprecated)]

use core::ffi;

use super::ExecutionProvider;
use crate::{
	AsPointer,
	error::{Error, Result},
	session::builder::SessionBuilder
};

/// [Arm NN execution provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/ArmNN-ExecutionProvider.html)
/// for ARM platforms.
#[derive(Debug, Default, Clone)]
#[deprecated = "recently removed from ONNX Runtime; use CPU (now optimized for ARM w/ Kleidi) or ACL, XNNPACK"]
pub struct ArmNN {
	use_arena: bool
}

super::impl_ep!(ArmNN);

impl ArmNN {
	/// Enable/disable the usage of the arena allocator.
	///
	/// ```
	/// # use ort::{ep, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = ep::ArmNN::default().with_arena_allocator(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_arena_allocator(mut self, enable: bool) -> Self {
		self.use_arena = enable;
		self
	}
}

impl ExecutionProvider for ArmNN {
	fn name(&self) -> &'static str {
		"ArmNNExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: ffi::c_int) -> ort_sys::OrtStatusPtr);
		unsafe { Error::result_from_status(OrtSessionOptionsAppendExecutionProvider_ArmNN(session_builder.ptr_mut(), self.use_arena.into())) }
	}
}
