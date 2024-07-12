use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
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
	backend_path: Option<String>,
	profiling_level: Option<QNNExecutionProviderProfilingLevel>,
	profiling_file_path: Option<String>,
	rpc_control_latency: Option<u32>,
	vtcm_mb: Option<usize>,
	htp_performance_mode: Option<QNNExecutionProviderPerformanceMode>,
	qnn_saver_path: Option<String>,
	qnn_context_priority: Option<QNNExecutionProviderContextPriority>,
	htp_graph_finalization_optimization_mode: Option<u8>,
	soc_model: Option<String>,
	htp_arch: Option<u32>,
	device_id: Option<i32>,
	enable_htp_fp16_precision: Option<bool>
}

impl QNNExecutionProvider {
	/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
	/// or `libQnnHtp.so` to use the accelerated backend.
	#[must_use]
	pub fn with_backend_path(mut self, path: impl ToString) -> Self {
		self.backend_path = Some(path.to_string());
		self
	}

	#[must_use]
	pub fn with_profiling(mut self, level: QNNExecutionProviderProfilingLevel) -> Self {
		self.profiling_level = Some(level);
		self
	}

	#[must_use]
	pub fn with_profiling_path(mut self, path: impl ToString) -> Self {
		self.profiling_file_path = Some(path.to_string());
		self
	}

	/// Allows client to set up RPC control latency in microseconds.
	#[must_use]
	pub fn with_rpc_control_latency(mut self, latency: u32) -> Self {
		self.rpc_control_latency = Some(latency);
		self
	}

	#[must_use]
	pub fn with_vtcm_mb(mut self, mb: usize) -> Self {
		self.vtcm_mb = Some(mb);
		self
	}

	#[must_use]
	pub fn with_performance_mode(mut self, mode: QNNExecutionProviderPerformanceMode) -> Self {
		self.htp_performance_mode = Some(mode);
		self
	}

	#[must_use]
	pub fn with_saver_path(mut self, path: impl ToString) -> Self {
		self.qnn_saver_path = Some(path.to_string());
		self
	}

	#[must_use]
	pub fn with_context_priority(mut self, priority: QNNExecutionProviderContextPriority) -> Self {
		self.qnn_context_priority = Some(priority);
		self
	}

	#[must_use]
	pub fn with_htp_graph_finalization_optimization_mode(mut self, mode: u8) -> Self {
		self.htp_graph_finalization_optimization_mode = Some(mode);
		self
	}

	#[must_use]
	pub fn with_soc_model(mut self, model: impl ToString) -> Self {
		self.soc_model = Some(model.to_string());
		self
	}

	#[must_use]
	pub fn with_htp_arch(mut self, arch: u32) -> Self {
		self.htp_arch = Some(arch);
		self
	}

	#[must_use]
	pub fn with_device_id(mut self, device: i32) -> Self {
		self.device_id = Some(device);
		self
	}

	#[must_use]
	pub fn with_htp_fp16_precision(mut self, enable: bool) -> Self {
		self.enable_htp_fp16_precision = Some(enable);
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "qnn"))]
		{
			let (key_ptrs, value_ptrs, len, _keys, _values) = super::map_keys! {
				backend_path = self.backend_path.clone(),
				profiling_level = self.profiling_level.as_ref().map(QNNExecutionProviderProfilingLevel::as_str),
				profiling_file_path = self.profiling_file_path.clone(),
				rpc_control_latency = self.rpc_control_latency,
				vtcm_mb = self.vtcm_mb,
				htp_performance_mode = self.htp_performance_mode.as_ref().map(QNNExecutionProviderPerformanceMode::as_str),
				qnn_saver_path = self.qnn_saver_path.clone(),
				qnn_context_priorty = self.qnn_context_priority.as_ref().map(QNNExecutionProviderContextPriority::as_str),
				htp_graph_finalization_optimization_mode = self.htp_graph_finalization_optimization_mode,
				soc_model = self.soc_model.clone(),
				htp_arch = self.htp_arch,
				device_id = self.device_id,
				enable_htp_fp16_precision = self.enable_htp_fp16_precision.map(<bool as Into<i32>>::into)
			};
			let ep_name = std::ffi::CString::new("QNN").unwrap_or_else(|_| unreachable!());
			return crate::error::status_to_result(crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.session_options_ptr.as_ptr(),
				ep_name.as_ptr(),
				key_ptrs.as_ptr(),
				value_ptrs.as_ptr(),
				len as _,
			)])
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
