use alloc::string::ToString;
use core::fmt;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

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

impl fmt::Display for PerformanceMode {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
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
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfilingLevel {
	Off,
	Basic,
	Detailed
}

impl fmt::Display for ProfilingLevel {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			ProfilingLevel::Off => "off",
			ProfilingLevel::Basic => "basic",
			ProfilingLevel::Detailed => "detailed"
		})
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

impl fmt::Display for ContextPriority {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			ContextPriority::Low => "low",
			ContextPriority::Normal => "normal",
			ContextPriority::NormalHigh => "normal_high",
			ContextPriority::High => "high"
		})
	}
}

#[derive(Debug, Default, Clone)]
pub struct QNN(ExecutionProviderOptions);

super::impl_ep!(arbitrary; QNN);

impl QNN {
	super::define_options! {
		/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
		/// or `libQnnHtp.so` to use the accelerated backend.
		pub fn with_backend_path(mut self, path: impl ToString) -> Self = "backend_path";

		pub fn with_profiling(mut self, level: ProfilingLevel) -> Self = "profiling_level";

		pub fn with_profiling_path(mut self, path: impl ToString) -> Self = "profiling_file_path";

		/// Allows client to set up RPC control latency in microseconds.
		pub fn with_rpc_control_latency(mut self, latency: u32) -> Self = "rpc_control_latency";

		pub fn with_vtcm_mb(mut self, mb: usize) -> Self = "vtcm_mb";

		pub fn with_performance_mode(mut self, mode: PerformanceMode) -> Self = "htp_performance_mode";

		pub fn with_saver_path(mut self, path: impl ToString) -> Self = "qnn_saver_path";

		pub fn with_context_priority(mut self, priority: ContextPriority) -> Self = "qnn_context_priority";

		pub fn with_htp_graph_finalization_optimization_mode(mut self, mode: u8) -> Self = "htp_graph_finalization_optimization_mode";

		pub fn with_soc_model(mut self, model: impl ToString) -> Self = "soc_model";

		pub fn with_htp_arch(mut self, arch: u32) -> Self = "htp_arch";

		pub fn with_device_id(mut self, device: i32) -> Self = "device_id";

		pub fn with_htp_fp16_precision(mut self, enable: bool) -> Self = "enable_htp_fp16_precision";

		pub fn with_htp_weight_sharing(mut self, enable: bool) -> Self = "enable_htp_weight_sharing";

		pub fn with_offload_graph_io_quantization(mut self, enable: bool) -> Self = "offload_graph_io_quantization";
	}
}

impl SimpleExecutionProvider for QNN {
	const CANONICAL_NAME: &'static str = "QNNExecutionProvider";
	const SHORT_NAME: &'static str = "QNN";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
