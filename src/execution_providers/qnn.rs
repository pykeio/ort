use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Default, Clone)]
pub struct QNNExecutionProvider {
	backend_path: Option<String>,
	qnn_context_cache_enable: Option<bool>,
	qnn_context_cache_path: Option<String>,
	profiling_level: Option<QNNExecutionProviderProfilingLevel>,
	rpc_control_latency: Option<u32>,
	htp_performance_mode: Option<QNNExecutionProviderPerformanceMode>
}

impl QNNExecutionProvider {
	/// The file path to QNN backend library. On Linux/Android, this is `libQnnCpu.so` to use the CPU backend,
	/// or `libQnnHtp.so` to use the accelerated backend.
	#[must_use]
	pub fn with_backend_path(mut self, path: impl ToString) -> Self {
		self.backend_path = Some(path.to_string());
		self
	}

	/// Configure whether to enable QNN graph creation from a cached QNN context file. If enabled, the QNN EP
	/// will load from the cached QNN context binary if it exists, or create one if it does not exist.
	#[must_use]
	pub fn with_enable_context_cache(mut self, enable: bool) -> Self {
		self.qnn_context_cache_enable = Some(enable);
		self
	}

	/// Explicitly provide the QNN context cache file (see [`QNNExecutionProvider::with_enable_context_cache`]).
	/// Defaults to `model_file.onnx.bin` if not provided.
	#[must_use]
	pub fn with_context_cache_path(mut self, path: impl ToString) -> Self {
		self.qnn_context_cache_path = Some(path.to_string());
		self
	}

	#[must_use]
	pub fn with_profiling(mut self, level: QNNExecutionProviderProfilingLevel) -> Self {
		self.profiling_level = Some(level);
		self
	}

	/// Allows client to set up RPC control latency in microseconds.
	#[must_use]
	pub fn with_rpc_control_latency(mut self, latency: u32) -> Self {
		self.rpc_control_latency = Some(latency);
		self
	}

	#[must_use]
	pub fn with_performance_mode(mut self, mode: QNNExecutionProviderPerformanceMode) -> Self {
		self.htp_performance_mode = Some(mode);
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
				qnn_context_cache_enable = self.qnn_context_cache_enable.map(<bool as Into<i32>>::into),
				qnn_context_cache_path = self.qnn_context_cache_path.clone(),
				htp_performance_mode = self.htp_performance_mode.as_ref().map(QNNExecutionProviderPerformanceMode::as_str),
				rpc_control_latency = self.rpc_control_latency
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
