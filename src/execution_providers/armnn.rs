use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [Arm NN execution provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/ArmNN-ExecutionProvider.html)
/// for ARM platforms.
#[derive(Debug, Default, Clone)]
pub struct ArmNNExecutionProvider {
	use_arena: bool
}

super::impl_ep!(ArmNNExecutionProvider);

impl ArmNNExecutionProvider {
	/// Enable/disable the usage of the arena allocator.
	///
	/// ```
	/// # use ort::{execution_providers::armnn::ArmNNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = ArmNNExecutionProvider::default().with_arena_allocator(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_arena_allocator(mut self, enable: bool) -> Self {
		self.use_arena = enable;
		self
	}
}

impl ExecutionProvider for ArmNNExecutionProvider {
	fn name(&self) -> &'static str {
		"ArmNNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", any(target_os = "linux", target_os = "android")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "armnn"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_ArmNN(options: *mut ort_sys::OrtSessionOptions, use_arena: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return Ok(unsafe {
				crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_ArmNN(session_builder.ptr_mut(), self.use_arena.into()))
			}?);
		}

		Err(RegisterError::MissingFeature)
	}
}
