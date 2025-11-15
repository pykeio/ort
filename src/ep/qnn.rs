use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceMode {
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

impl PerformanceMode {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			PerformanceMode::Default => "default",
			PerformanceMode::Burst => "burst",
			PerformanceMode::Balanced => "balanced",
			PerformanceMode::HighPerformance => "high_performance",
			PerformanceMode::HighPowerSaver => "high_power_saver",
			PerformanceMode::LowPowerSaver => "low_power_saver",
			PerformanceMode::LowBalanced => "low_balanced",
			PerformanceMode::PowerSaver => "power_saver",
			PerformanceMode::ExtremePowerSaver => "extreme_power_saver",
			PerformanceMode::SustainedHighPerformance => "sustained_high_performance"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingLevel {
	Off,
	Basic,
	Detailed
}

impl ProfilingLevel {
	pub fn as_str(&self) -> &'static str {
		match self {
			ProfilingLevel::Off => "off",
			ProfilingLevel::Basic => "basic",
			ProfilingLevel::Detailed => "detailed"
		}
	}
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextPriority {
	Low,
	#[default]
	Normal,
	NormalHigh,
	High
}

impl ContextPriority {
	pub fn as_str(&self) -> &'static str {
		match self {
			ContextPriority::Low => "low",
			ContextPriority::Normal => "normal",
			ContextPriority::NormalHigh => "normal_high",
			ContextPriority::High => "high"
		}
	}
}

#[derive(Debug, Default, Clone)]
pub struct QNN {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; QNN);

impl QNN {
	/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
	/// or `libQnnHtp.so` to use the accelerated backend.
	#[must_use]
	pub fn with_backend_path(mut self, path: impl ToString) -> Self {
		self.options.set("backend_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_profiling(mut self, level: ProfilingLevel) -> Self {
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
	pub fn with_performance_mode(mut self, mode: PerformanceMode) -> Self {
		self.options.set("htp_performance_mode", mode.as_str());
		self
	}

	#[must_use]
	pub fn with_saver_path(mut self, path: impl ToString) -> Self {
		self.options.set("qnn_saver_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_context_priority(mut self, priority: ContextPriority) -> Self {
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

impl ExecutionProvider for QNN {
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
