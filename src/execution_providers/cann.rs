use crate::{
	error::{Error, Result},
	execution_providers::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CANNExecutionProviderPrecisionMode {
	/// Convert to float32 first according to operator implementation
	ForceFP32,
	/// Convert to float16 when float16 and float32 are both supported
	ForceFP16,
	/// Convert to float16 when float32 is not supported
	AllowFP32ToFP16,
	/// Keep dtypes as is
	MustKeepOrigin,
	/// Allow mixed precision
	AllowMixedPrecision
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CANNExecutionProviderImplementationMode {
	HighPrecision,
	HighPerformance
}

#[derive(Default, Debug, Clone)]
pub struct CANNExecutionProvider {
	device_id: Option<i32>,
	npu_mem_limit: Option<usize>,
	arena_extend_strategy: Option<ArenaExtendStrategy>,
	enable_cann_graph: Option<bool>,
	dump_graphs: Option<bool>,
	precision_mode: Option<CANNExecutionProviderPrecisionMode>,
	op_select_impl_mode: Option<CANNExecutionProviderImplementationMode>,
	optypelist_for_impl_mode: Option<String>
}

impl CANNExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = Some(device_id);
		self
	}

	/// Configure the size limit of the device memory arena in bytes. This size limit is only for the execution
	/// provider’s arena. The total device memory usage may be higher.
	#[must_use]
	pub fn with_memory_limit(mut self, limit: usize) -> Self {
		self.npu_mem_limit = Some(limit);
		self
	}

	/// Configure the strategy for extending the device's memory arena.
	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.arena_extend_strategy = Some(strategy);
		self
	}

	/// Configure whether to use the graph inference engine to speed up performance. The recommended and default setting
	/// is true. If false, it will fall back to the single-operator inference engine.
	#[must_use]
	pub fn with_cann_graph(mut self, enable: bool) -> Self {
		self.enable_cann_graph = Some(enable);
		self
	}

	/// Configure whether to dump the subgraph into ONNX format for analysis of subgraph segmentation.
	#[must_use]
	pub fn with_dump_graphs(mut self) -> Self {
		self.dump_graphs = Some(true);
		self
	}

	/// Set the precision mode of the operator. See [`CANNExecutionProviderPrecisionMode`].
	#[must_use]
	pub fn with_precision_mode(mut self, mode: CANNExecutionProviderPrecisionMode) -> Self {
		self.precision_mode = Some(mode);
		self
	}

	/// Configure the implementation mode for operators. Some CANN operators can have both high-precision and
	/// high-performance implementations.
	#[must_use]
	pub fn with_implementation_mode(mut self, mode: CANNExecutionProviderImplementationMode) -> Self {
		self.op_select_impl_mode = Some(mode);
		self
	}

	/// Enumerate the list of operators which use the mode specified by
	/// [`CANNExecutionProvider::with_implementation_mode`].
	///
	/// As of ONNX Runtime v1.16.2, the supported operators are:
	/// - `Pooling`
	/// - `SoftmaxV2`
	/// - `LRN`
	/// - `ROIAlign`
	#[must_use]
	pub fn with_implementation_mode_oplist(mut self, list: impl ToString) -> Self {
		self.optypelist_for_impl_mode = Some(list.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<CANNExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: CANNExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for CANNExecutionProvider {
	fn as_str(&self) -> &'static str {
		"CANNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "cann"))]
		{
			let mut cann_options: *mut ort_sys::OrtCANNProviderOptions = std::ptr::null_mut();
			crate::error::status_to_result(crate::ortsys![unsafe CreateCANNProviderOptions(&mut cann_options)]).map_err(Error::ExecutionProvider)?;
			let (key_ptrs, value_ptrs, len, keys, values) = super::map_keys! {
				device_id = self.device_id,
				npu_mem_limit = self.npu_mem_limit,
				arena_extend_strategy = self.arena_extend_strategy.as_ref().map(|v| match v {
					ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
					ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
				}),
				enable_cann_graph = self.enable_cann_graph.map(<bool as Into<i32>>::into),
				dump_graphs = self.dump_graphs.map(<bool as Into<i32>>::into),
				precision_mode = self.precision_mode.as_ref().map(|v| match v {
					CANNExecutionProviderPrecisionMode::ForceFP32 => "force_fp32",
					CANNExecutionProviderPrecisionMode::ForceFP16 => "force_fp16",
					CANNExecutionProviderPrecisionMode::AllowFP32ToFP16 => "allow_fp32_to_fp16",
					CANNExecutionProviderPrecisionMode::MustKeepOrigin => "must_keep_origin_dtype",
					CANNExecutionProviderPrecisionMode::AllowMixedPrecision => "allow_mix_precision"
				}),
				op_select_impl_mode = self.op_select_impl_mode.as_ref().map(|v| match v {
					CANNExecutionProviderImplementationMode::HighPrecision => "high_precision",
					CANNExecutionProviderImplementationMode::HighPerformance => "high_performance"
				}),
				optypelist_for_impl_mode = self.optypelist_for_impl_mode.clone()
			};
			if let Err(e) =
				crate::error::status_to_result(crate::ortsys![unsafe UpdateCANNProviderOptions(cann_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), len as _)])
					.map_err(Error::ExecutionProvider)
			{
				crate::ortsys![unsafe ReleaseCANNProviderOptions(cann_options)];
				std::mem::drop((keys, values));
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_CANN(session_builder.session_options_ptr.as_ptr(), cann_options)];
			crate::ortsys![unsafe ReleaseCANNProviderOptions(cann_options)];
			std::mem::drop((keys, values));
			return crate::error::status_to_result(status).map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
