use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QNNPerformanceMode {
	Default,
	Burst,
	Balanced,
	HighPerformance,
	HighPowerSaver,
	LowPowerSaver,
	LowBalanced,
	PowerSaver,
	ExtremePowerSaver,
	SustainedHighPerformance
}

impl QNNPerformanceMode {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNPerformanceMode::Default => "default",
			QNNPerformanceMode::Burst => "burst",
			QNNPerformanceMode::Balanced => "balanced",
			QNNPerformanceMode::HighPerformance => "high_performance",
			QNNPerformanceMode::HighPowerSaver => "high_power_saver",
			QNNPerformanceMode::LowPowerSaver => "low_power_saver",
			QNNPerformanceMode::LowBalanced => "low_balanced",
			QNNPerformanceMode::PowerSaver => "power_saver",
			QNNPerformanceMode::ExtremePowerSaver => "extreme_power_saver",
			QNNPerformanceMode::SustainedHighPerformance => "sustained_high_performance"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QNNProfilingLevel {
	Off,
	Basic,
	Detailed
}

impl QNNProfilingLevel {
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNProfilingLevel::Off => "off",
			QNNProfilingLevel::Basic => "basic",
			QNNProfilingLevel::Detailed => "detailed"
		}
	}
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QNNContextPriority {
	Low,
	#[default]
	Normal,
	NormalHigh,
	High
}

impl QNNContextPriority {
	pub fn as_str(&self) -> &'static str {
		match self {
			QNNContextPriority::Low => "low",
			QNNContextPriority::Normal => "normal",
			QNNContextPriority::NormalHigh => "normal_high",
			QNNContextPriority::High => "high"
		}
	}
}

#[derive(Debug, Default, Clone)]
pub struct QNNExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; QNNExecutionProvider);

impl QNNExecutionProvider {
	/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
	/// or `libQnnHtp.so` to use the accelerated backend.
	#[must_use]
	pub fn with_backend_path(mut self, path: impl ToString) -> Self {
		self.options.set("backend_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_profiling(mut self, level: QNNProfilingLevel) -> Self {
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
	pub fn with_performance_mode(mut self, mode: QNNPerformanceMode) -> Self {
		self.options.set("htp_performance_mode", mode.as_str());
		self
	}

	#[must_use]
	pub fn with_saver_path(mut self, path: impl ToString) -> Self {
		self.options.set("qnn_saver_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_context_priority(mut self, priority: QNNContextPriority) -> Self {
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
}

impl ExecutionProvider for QNNExecutionProvider {
	fn name(&self) -> &'static str {
		"QNNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "aarch64", any(target_os = "windows", target_os = "linux", target_os = "android")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "qnn"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"QNN".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
