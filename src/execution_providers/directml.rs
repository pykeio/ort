use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [DirectML execution provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) for
/// DirectX 12-enabled hardware on Windows.
///
/// # Performance considerations
/// The DirectML EP performs best when the size of inputs & outputs are known when the session is created. For graphs
/// with dynamically sized inputs, you can override individual dimensions by constructing the session with
/// [`SessionBuilder::with_dimension_override`]:
/// ```no_run
/// # use ort::{execution_providers::directml::DirectMLExecutionProvider, session::Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_execution_providers([DirectMLExecutionProvider::default().build()])?
/// 	.with_dimension_override("batch", 1)?
/// 	.with_dimension_override("seq_len", 512)?
/// 	.commit_from_file("gpt2.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct DirectMLExecutionProvider {
	device_id: i32
}

super::impl_ep!(DirectMLExecutionProvider);

impl DirectMLExecutionProvider {
	/// Configures which device the EP should use.
	///
	/// ```
	/// # use ort::{execution_providers::directml::DirectMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = DirectMLExecutionProvider::default().with_device_id(1).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = device_id;
		self
	}
}

impl ExecutionProvider for DirectMLExecutionProvider {
	fn name(&self) -> &'static str {
		"DmlExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_os = "windows")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "directml"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_DML(options: *mut ort_sys::OrtSessionOptions, device_id: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return Ok(unsafe {
				crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_DML(session_builder.ptr_mut(), self.device_id as _))
			}?);
		}

		Err(RegisterError::MissingFeature)
	}
}
