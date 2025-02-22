use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QNNExecutionProviderPerformanceMode {
	Default,
	Burst,
	Balanced,
	HighPerformance,
	HighPowerSaver,
	LowPowerSaver,
	LowBalanced,
	PowerSaver,
	SustainedHighPerformance
}

impl QNNExecutionProviderPerformanceMode {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNExecutionProviderPerformanceMode::Default => "default",
			QNNExecutionProviderPerformanceMode::Burst => "burst",
			QNNExecutionProviderPerformanceMode::Balanced => "balanced",
			QNNExecutionProviderPerformanceMode::HighPerformance => "high_performance",
			QNNExecutionProviderPerformanceMode::HighPowerSaver => "high_power_saver",
			QNNExecutionProviderPerformanceMode::LowPowerSaver => "low_power_saver",
			QNNExecutionProviderPerformanceMode::LowBalanced => "low_balanced",
			QNNExecutionProviderPerformanceMode::PowerSaver => "power_saver",
			QNNExecutionProviderPerformanceMode::SustainedHighPerformance => "sustained_high_performance"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QNNExecutionProviderProfilingLevel {
	Off,
	Basic,
	Detailed
}

impl QNNExecutionProviderProfilingLevel {
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNExecutionProviderProfilingLevel::Off => "off",
			QNNExecutionProviderProfilingLevel::Basic => "basic",
			QNNExecutionProviderProfilingLevel::Detailed => "detailed"
		}
	}
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QNNExecutionProviderContextPriority {
	Low,
	#[default]
	Normal,
	NormalHigh,
	High
}

impl QNNExecutionProviderContextPriority {
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNExecutionProviderContextPriority::Low => "low",
			QNNExecutionProviderContextPriority::Normal => "normal",
			QNNExecutionProviderContextPriority::NormalHigh => "normal_high",
			QNNExecutionProviderContextPriority::High => "normal_high"
		}
	}
}

#[derive(Debug, Default, Clone)]
pub struct QNNExecutionProvider {
	options: ExecutionProviderOptions
}

impl QNNExecutionProvider {
	/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
	/// or `libQnnHtp.so` to use the accelerated backend.
	#[must_use]
	pub fn with_backend_path(mut self, path: impl ToString) -> Self {
		self.options.set("backend_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_profiling(mut self, level: QNNExecutionProviderProfilingLevel) -> Self {
		self.options.set("profiling_level", level.as_str());
		self
	}

	#[must_use]
	pub fn with_profiling_path(mut self, path: impl ToString) -> Self {
		self.options.set("profiling_file_path", path.to_string());
		self
	}

	/// Allows client to set up RPC control latency in microseconds.
	#[must_use]
	pub fn with_rpc_control_latency(mut self, latency: u32) -> Self {
		self.options.set("rpc_control_latency", latency.to_string());
		self
	}

	#[must_use]
	pub fn with_vtcm_mb(mut self, mb: usize) -> Self {
		self.options.set("vtcm_mb", mb.to_string());
		self
	}

	#[must_use]
	pub fn with_performance_mode(mut self, mode: QNNExecutionProviderPerformanceMode) -> Self {
		self.options.set("htp_performance_mode", mode.as_str());
		self
	}

	#[must_use]
	pub fn with_saver_path(mut self, path: impl ToString) -> Self {
		self.options.set("qnn_saver_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_context_priority(mut self, priority: QNNExecutionProviderContextPriority) -> Self {
		self.options.set("qnn_context_priority", priority.as_str());
		self
	}

	#[must_use]
	pub fn with_htp_graph_finalization_optimization_mode(mut self, mode: u8) -> Self {
		self.options.set("htp_graph_finalization_optimization_mode", mode.to_string());
		self
	}

	#[must_use]
	pub fn with_soc_model(mut self, model: impl ToString) -> Self {
		self.options.set("soc_model", model.to_string());
		self
	}

	#[must_use]
	pub fn with_htp_arch(mut self, arch: u32) -> Self {
		self.options.set("htp_arch", arch.to_string());
		self
	}

	#[must_use]
	pub fn with_device_id(mut self, device: i32) -> Self {
		self.options.set("device_id", device.to_string());
		self
	}

	#[must_use]
	pub fn with_htp_fp16_precision(mut self, enable: bool) -> Self {
		self.options.set("enable_htp_fp16_precision", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_htp_weight_sharing(mut self, enable: bool) -> Self {
		self.options.set("enable_htp_weight_sharing", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_offload_graph_io_quantization(mut self, enable: bool) -> Self {
		self.options.set("offload_graph_io_quantization", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for QNNExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<QNNExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: QNNExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for QNNExecutionProvider {
	fn as_str(&self) -> &'static str {
		"QNNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", any(target_os = "windows", target_os = "linux", target_os = "android")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "qnn"))]
		{
			use crate::AsPointer;

			let ffi_options = self.options.to_ffi();
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"QNN".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
