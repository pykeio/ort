use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
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
	options: ExecutionProviderOptions
}

impl CANNExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// Configure the size limit of the device memory arena in bytes. This size limit is only for the execution
	/// provider’s arena. The total device memory usage may be higher.
	#[must_use]
	pub fn with_memory_limit(mut self, limit: usize) -> Self {
		self.options.set("npu_mem_limit", limit.to_string());
		self
	}

	/// Configure the strategy for extending the device's memory arena.
	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.options.set("arena_extend_strategy", match strategy {
			ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
			ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
		});
		self
	}

	/// Configure whether to use the graph inference engine to speed up performance. The recommended and default setting
	/// is true. If false, it will fall back to the single-operator inference engine.
	#[must_use]
	pub fn with_cann_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cann_graph", if enable { "1" } else { "0" });
		self
	}

	/// Configure whether to dump the subgraph into ONNX format for analysis of subgraph segmentation.
	#[must_use]
	pub fn with_dump_graphs(mut self, enable: bool) -> Self {
		self.options.set("dump_graphs", if enable { "1" } else { "0" });
		self
	}

	/// Set the precision mode of the operator. See [`CANNExecutionProviderPrecisionMode`].
	#[must_use]
	pub fn with_precision_mode(mut self, mode: CANNExecutionProviderPrecisionMode) -> Self {
		self.options.set("precision_mode", match mode {
			CANNExecutionProviderPrecisionMode::ForceFP32 => "force_fp32",
			CANNExecutionProviderPrecisionMode::ForceFP16 => "force_fp16",
			CANNExecutionProviderPrecisionMode::AllowFP32ToFP16 => "allow_fp32_to_fp16",
			CANNExecutionProviderPrecisionMode::MustKeepOrigin => "must_keep_origin_dtype",
			CANNExecutionProviderPrecisionMode::AllowMixedPrecision => "allow_mix_precision"
		});
		self
	}

	/// Configure the implementation mode for operators. Some CANN operators can have both high-precision and
	/// high-performance implementations.
	#[must_use]
	pub fn with_implementation_mode(mut self, mode: CANNExecutionProviderImplementationMode) -> Self {
		self.options.set("op_select_impl_mode", match mode {
			CANNExecutionProviderImplementationMode::HighPrecision => "high_precision",
			CANNExecutionProviderImplementationMode::HighPerformance => "high_performance"
		});
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
		self.options.set("optypelist_for_impl_mode", list.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for CANNExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "cann"))]
		{
			use crate::AsPointer;

			let mut cann_options: *mut ort_sys::OrtCANNProviderOptions = core::ptr::null_mut();
			crate::ortsys![unsafe CreateCANNProviderOptions(&mut cann_options)?];
			let ffi_options = self.options.to_ffi();

			let res = crate::ortsys![unsafe UpdateCANNProviderOptions(
				cann_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)];
			if let Err(e) = unsafe { crate::error::status_to_result(res) } {
				crate::ortsys![unsafe ReleaseCANNProviderOptions(cann_options)];
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_CANN(session_builder.ptr_mut(), cann_options)];
			crate::ortsys![unsafe ReleaseCANNProviderOptions(cann_options)];
			return unsafe { crate::error::status_to_result(status) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
