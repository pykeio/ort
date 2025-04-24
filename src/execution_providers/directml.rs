use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct DirectMLExecutionProvider {
	device_id: i32
}

super::impl_ep!(DirectMLExecutionProvider);

impl DirectMLExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = device_id;
		self
	}
}

impl ExecutionProvider for DirectMLExecutionProvider {
	fn as_str(&self) -> &'static str {
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
