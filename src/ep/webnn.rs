use core::fmt;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerPreference {
	#[default]
	Default,
	HighPerformance,
	LowPower
}

impl fmt::Display for PowerPreference {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			PowerPreference::Default => "default",
			PowerPreference::HighPerformance => "high-performance",
			PowerPreference::LowPower => "low-power"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
	CPU,
	GPU,
	NPU
}

impl fmt::Display for DeviceType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			DeviceType::CPU => "cpu",
			DeviceType::GPU => "gpu",
			DeviceType::NPU => "npu"
		})
	}
}

#[derive(Debug, Default, Clone)]
pub struct WebNN(ExecutionProviderOptions);

super::impl_ep!(arbitrary; WebNN);

impl WebNN {
	super::define_options! {
		pub fn with_device_type(mut self, device_type: DeviceType) -> Self = "deviceType";

		pub fn with_power_preference(mut self, pref: PowerPreference) -> Self = "powerPreference";

		pub fn with_threads(mut self, threads: u32) -> Self = "numThreads";
	}
}

impl SimpleExecutionProvider for WebNN {
	const CANONICAL_NAME: &'static str = "WebNNExecutionProvider";
	const SHORT_NAME: &'static str = "WEBNN";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
