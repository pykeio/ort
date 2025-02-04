use alloc::format;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "directml"))]
extern "C" {
	fn OrtSessionOptionsAppendExecutionProvider_DML(options: *mut ort_sys::OrtSessionOptions, device_id: core::ffi::c_int) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct DirectMLExecutionProvider {
	device_id: i32
}

impl DirectMLExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = device_id;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<DirectMLExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: DirectMLExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "directml"))]
		{
			use crate::AsPointer;

			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_DML(options: *mut ort_sys::OrtSessionOptions, device_id: core::ffi::c_int) -> ort_sys::OrtStatusPtr);
			return unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_DML(session_builder.ptr_mut(), self.device_id as _)) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
