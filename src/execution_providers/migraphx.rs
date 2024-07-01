use std::ffi::CString;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct MIGraphXExecutionProvider {
	device_id: i32,
	enable_fp16: bool,
	enable_int8: bool,
	use_native_calibration_table: bool,
	int8_calibration_table_name: Option<CString>
}

impl MIGraphXExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = device_id;
		self
	}

	#[must_use]
	pub fn with_fp16(mut self) -> Self {
		self.enable_fp16 = true;
		self
	}

	#[must_use]
	pub fn with_int8(mut self) -> Self {
		self.enable_int8 = true;
		self
	}

	#[must_use]
	pub fn with_native_calibration_table(mut self, table_name: Option<impl AsRef<str>>) -> Self {
		self.use_native_calibration_table = true;
		self.int8_calibration_table_name = table_name.map(|c| CString::new(c.as_ref()).expect("invalid string"));
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<MIGraphXExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: MIGraphXExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for MIGraphXExecutionProvider {
	fn as_str(&self) -> &'static str {
		"MIGraphXExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "migraphx"))]
		{
			let options = ort_sys::OrtMIGraphXProviderOptions {
				device_id: self.device_id,
				migraphx_fp16_enable: self.enable_fp16.into(),
				migraphx_int8_enable: self.enable_int8.into(),
				migraphx_use_native_calibration_table: self.use_native_calibration_table.into(),
				migraphx_int8_calibration_table_name: self
					.int8_calibration_table_name
					.as_ref()
					.map(|c| c.as_ptr())
					.unwrap_or_else(std::ptr::null)
			};
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_MIGraphX(session_builder.session_options_ptr.as_ptr(), &options) -> Error::ExecutionProvider];
			return Ok(());
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
