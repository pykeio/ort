use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPreference {
	#[default]
	Default,
	HighPerformance,
	LowPower
}

impl PowerPreference {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			PowerPreference::Default => "default",
			PowerPreference::HighPerformance => "high-performance",
			PowerPreference::LowPower => "low-power"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
	CPU,
	GPU,
	NPU
}

impl DeviceType {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			DeviceType::CPU => "cpu",
			DeviceType::GPU => "gpu",
			DeviceType::NPU => "npu"
		}
	}
}

#[derive(Debug, Default, Clone)]
pub struct WebNN {
	options: ExecutionProviderOptions
}

impl WebNN {
	#[must_use]
	pub fn with_device_type(mut self, device_type: DeviceType) -> Self {
		self.options.set("deviceType", device_type.as_str());
		self
	}

	#[must_use]
	pub fn with_power_preference(mut self, pref: PowerPreference) -> Self {
		self.options.set("powerPreference", pref.as_str());
		self
	}

	#[must_use]
	pub fn with_threads(mut self, threads: u32) -> Self {
		self.options.set("numThreads", threads.to_string());
		self
	}
}

super::impl_ep!(arbitrary; WebNN);

impl ExecutionProvider for WebNN {
	fn name(&self) -> &'static str {
		"WebNNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_arch = "wasm32")
	}

	#[allow(unused)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		let ffi_options = self.options.to_ffi();
		ortsys![unsafe SessionOptionsAppendExecutionProvider(
			session_builder.ptr_mut(),
			c"WebNN".as_ptr().cast::<core::ffi::c_char>(),
			ffi_options.key_ptrs(),
			ffi_options.value_ptrs(),
			ffi_options.len(),
		)?];
		Ok(())
	}
}
