use alloc::string::ToString;

use super::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CANNPrecisionMode {
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
pub enum CANNImplementationMode {
	/// Prefer high precision, potentially at the cost of some performance.
	HighPrecision,
	/// Prefer high performance, potentially with lower accuracy.
	HighPerformance
}

/// [CANN execution provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
/// for hardware acceleration using Huawei Ascend AI devices.
#[derive(Default, Debug, Clone)]
pub struct CANNExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; CANNExecutionProvider);

impl CANNExecutionProvider {
	/// Configures which device the EP should use.
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_device_id(0).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// Configure the size limit of the device memory arena in bytes. This size limit is only for the execution
	/// provider’s arena; the total device memory usage may be higher.
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_memory_limit(2 * 1024 * 1024 * 1024).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_memory_limit(mut self, limit: usize) -> Self {
		self.options.set("npu_mem_limit", limit.to_string());
		self
	}

	/// Configure the strategy for extending the device's memory arena.
	///
	/// ```
	/// # use ort::{execution_providers::{cann::CANNExecutionProvider, ArenaExtendStrategy}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default()
	/// 	.with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.options.set(
			"arena_extend_strategy",
			match strategy {
				ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
				ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
			}
		);
		self
	}

	/// Configure whether to use the graph inference engine to speed up performance. The recommended and default setting
	/// is `true`. If `false`, it will fall back to the single-operator inference engine.
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_cann_graph(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_cann_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cann_graph", if enable { "1" } else { "0" });
		self
	}

	/// Configure whether to dump the subgraph into ONNX format for analysis of subgraph segmentation.
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_dump_graphs(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_dump_graphs(mut self, enable: bool) -> Self {
		self.options.set("dump_graphs", if enable { "1" } else { "0" });
		self
	}

	/// Configure whether to dump the offline model to an `.om` file.
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_dump_om_model(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_dump_om_model(mut self, enable: bool) -> Self {
		self.options.set("dump_om_model", if enable { "1" } else { "0" });
		self
	}

	/// Configure the precision mode; see [`CANNPrecisionMode`].
	///
	/// ```
	/// # use ort::{execution_providers::cann::{CANNExecutionProvider, CANNPrecisionMode}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_precision_mode(CANNPrecisionMode::ForceFP16).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_precision_mode(mut self, mode: CANNPrecisionMode) -> Self {
		self.options.set(
			"precision_mode",
			match mode {
				CANNPrecisionMode::ForceFP32 => "force_fp32",
				CANNPrecisionMode::ForceFP16 => "force_fp16",
				CANNPrecisionMode::AllowFP32ToFP16 => "allow_fp32_to_fp16",
				CANNPrecisionMode::MustKeepOrigin => "must_keep_origin_dtype",
				CANNPrecisionMode::AllowMixedPrecision => "allow_mix_precision"
			}
		);
		self
	}

	/// Configure the implementation mode for operators. Some CANN operators can have both high-precision and
	/// high-performance implementations.
	///
	/// ```
	/// # use ort::{execution_providers::cann::{CANNExecutionProvider, CANNImplementationMode}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default()
	/// 	.with_implementation_mode(CANNImplementationMode::HighPerformance)
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_implementation_mode(mut self, mode: CANNImplementationMode) -> Self {
		self.options.set(
			"op_select_impl_mode",
			match mode {
				CANNImplementationMode::HighPrecision => "high_precision",
				CANNImplementationMode::HighPerformance => "high_performance"
			}
		);
		self
	}

	/// Configure the list of operators which use the mode specified by
	/// [`CANNExecutionProvider::with_implementation_mode`].
	///
	/// ```
	/// # use ort::{execution_providers::cann::CANNExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CANNExecutionProvider::default().with_implementation_mode_oplist("LayerNorm,Gelu").build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_implementation_mode_oplist(mut self, list: impl ToString) -> Self {
		self.options.set("optypelist_for_implmode", list.to_string());
		self
	}
}

impl ExecutionProvider for CANNExecutionProvider {
	fn name(&self) -> &'static str {
		"CANNExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "cann"))]
		{
			use core::ptr;

			use crate::{AsPointer, ortsys, util};

			let mut cann_options: *mut ort_sys::OrtCANNProviderOptions = ptr::null_mut();
			ortsys![unsafe CreateCANNProviderOptions(&mut cann_options)?];
			let _guard = util::run_on_drop(|| {
				ortsys![unsafe ReleaseCANNProviderOptions(cann_options)];
			});

			let ffi_options = self.options.to_ffi();

			ortsys![unsafe UpdateCANNProviderOptions(
				cann_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			ortsys![unsafe SessionOptionsAppendExecutionProvider_CANN(session_builder.ptr_mut(), cann_options)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
