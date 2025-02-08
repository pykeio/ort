use alloc::{format, string::ToString};
use core::ops::BitOr;

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

// https://github.com/microsoft/onnxruntime/blob/ffceed9d44f2f3efb9dd69fa75fea51163c91d91/onnxruntime/contrib_ops/cpu/bert/attention_common.h#L160-L171
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CUDAExecutionProviderAttentionBackend(u32);

impl CUDAExecutionProviderAttentionBackend {
	pub const FLASH_ATTENTION: Self = Self(1 << 0);
	pub const EFFICIENT_ATTENTION: Self = Self(1 << 1);
	pub const TRT_FUSED_ATTENTION: Self = Self(1 << 2);
	pub const CUDNN_FLASH_ATTENTION: Self = Self(1 << 3);
	pub const MATH: Self = Self(1 << 4);

	pub const TRT_FLASH_ATTENTION: Self = Self(1 << 5);
	pub const TRT_CROSS_ATTENTION: Self = Self(1 << 6);
	pub const TRT_CAUSAL_ATTENTION: Self = Self(1 << 7);

	pub fn none() -> Self {
		CUDAExecutionProviderAttentionBackend(0)
	}

	pub fn all() -> Self {
		Self::FLASH_ATTENTION
			| Self::EFFICIENT_ATTENTION
			| Self::TRT_FUSED_ATTENTION
			| Self::CUDNN_FLASH_ATTENTION
			| Self::MATH
			| Self::TRT_FLASH_ATTENTION
			| Self::TRT_CROSS_ATTENTION
			| Self::TRT_CAUSAL_ATTENTION
	}
}

impl BitOr for CUDAExecutionProviderAttentionBackend {
	type Output = Self;
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(rhs.0 | self.0)
	}
}

/// The type of search done for cuDNN convolution algorithms.
#[derive(Debug, Clone)]
pub enum CUDAExecutionProviderCuDNNConvAlgoSearch {
	/// Expensive exhaustive benchmarking using [`cudnnFindConvolutionForwardAlgorithmEx`][exhaustive].
	/// This function will attempt all possible algorithms for `cudnnConvolutionForward` to find the fastest algorithm.
	/// Exhaustive search trades off between memory usage and speed. The first execution of a graph will be slow while
	/// possible convolution algorithms are tested.
	///
	/// [exhaustive]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnFindConvolutionForwardAlgorithmEx
	Exhaustive,
	/// Lightweight heuristic-based search using [`cudnnGetConvolutionForwardAlgorithm_v7`][heuristic].
	/// Heuristic search sorts available convolution algorithms by expected (based on internal heuristic) relative
	/// performance.
	///
	/// [heuristic]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetConvolutionForwardAlgorithm_v7
	Heuristic,
	/// Uses the default convolution algorithm, [`CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM`][fwdalgo].
	/// The default algorithm may not have the best performance depending on specific hardware used. It's recommended to
	/// use [`Exhaustive`] or [`Heuristic`] to search for a faster algorithm instead. However, `Default` does have its
	/// uses, such as when available memory is tight.
	///
	/// > **NOTE**: This name may be confusing as this is not the default search algorithm for the CUDA EP. The default
	/// > search algorithm is actually [`Exhaustive`].
	///
	/// [fwdalgo]: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionFwdAlgo_t
	/// [`Exhaustive`]: CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive
	/// [`Heuristic`]: CUDAExecutionProviderCuDNNConvAlgoSearch::Heuristic
	Default
}

impl Default for CUDAExecutionProviderCuDNNConvAlgoSearch {
	fn default() -> Self {
		Self::Exhaustive
	}
}

#[derive(Debug, Default, Clone)]
pub struct CUDAExecutionProvider {
	options: ExecutionProviderOptions
}

impl CUDAExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// Configure the size limit of the device memory arena in bytes. This size limit is only for the execution
	/// provider’s arena. The total device memory usage may be higher.
	#[must_use]
	pub fn with_memory_limit(mut self, limit: usize) -> Self {
		self.options.set("gpu_mem_limit", limit.to_string());
		self
	}

	/// Confiure the strategy for extending the device's memory arena.
	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.options.set("arena_extend_strategy", match strategy {
			ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
			ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
		});
		self
	}

	/// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	/// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	/// configuration (input shape, filter shape, etc.) in each `Conv` node. This option controlls the type of search
	/// done for cuDNN convolution algorithms. See [`CUDAExecutionProviderCuDNNConvAlgoSearch`] for more info.
	#[must_use]
	pub fn with_conv_algorithm_search(mut self, search: CUDAExecutionProviderCuDNNConvAlgoSearch) -> Self {
		self.options.set("cudnn_conv_algo_search", match search {
			CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive => "EXHAUSTIVE",
			CUDAExecutionProviderCuDNNConvAlgoSearch::Heuristic => "HEURISTIC",
			CUDAExecutionProviderCuDNNConvAlgoSearch::Default => "DEFAULT"
		});
		self
	}

	/// Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false,
	/// there are race conditions and possibly better performance.
	#[must_use]
	pub fn with_copy_in_default_stream(mut self, enable: bool) -> Self {
		self.options.set("do_copy_in_default_stream", if enable { "1" } else { "0" });
		self
	}

	/// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	/// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	/// configuration (input shape, filter shape, etc.) in each `Conv` node. This sub-step involves querying cuDNN for a
	/// “workspace” memory size and have this allocated so that cuDNN can use this auxiliary memory while determining
	/// the “optimal” convolution algorithm to use.
	///
	/// When `with_conv_max_workspace` is set to false, ORT will clamp the workspace size to 32 MB, which may lead to
	/// cuDNN selecting a suboptimal convolution algorithm. The recommended (and default) value is `true`.
	#[must_use]
	pub fn with_conv_max_workspace(mut self, enable: bool) -> Self {
		self.options.set("cudnn_conv_use_max_workspace", if enable { "1" } else { "0" });
		self
	}

	/// ORT leverages cuDNN for convolution operations. While cuDNN only takes 4-D or 5-D tensors as input for
	/// convolution operations, dimension padding is needed if the input is a 3-D tensor. Given an input tensor of shape
	/// `[N, C, D]`, it can be padded to `[N, C, D, 1]` or `[N, C, 1, D]`. While both of these padding methods produce
	/// the same output, the performance may differ because different convolution algorithms are selected,
	/// especially on some devices such as A100. By default, the input is padded to `[N, C, D, 1]`. Set this option to
	/// true to instead use `[N, C, 1, D]`.
	#[must_use]
	pub fn with_conv1d_pad_to_nc1d(mut self, enable: bool) -> Self {
		self.options.set("cudnn_conv1d_pad_to_nc1d", if enable { "1" } else { "0" });
		self
	}

	/// ORT supports the usage of CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
	/// sequentially. Currently, there are some constraints with regards to using the CUDA Graphs feature:
	///
	/// - Models with control-flow ops (i.e. If, Loop and Scan ops) are not supported.
	/// - Usage of CUDA Graphs is limited to models where-in all the model ops (graph nodes) can be partitioned to the
	///   CUDA EP.
	/// - The input/output types of models must be tensors.
	/// - Shapes of inputs/outputs cannot change across inference calls. Dynamic shape models are supported, but the
	///   input/output shapes must be the same across each inference call.
	/// - By design, CUDA Graphs is designed to read from/write to the same CUDA virtual memory addresses during the
	///   graph replaying step as it does during the graph capturing step. Due to this requirement, usage of this
	///   feature requires using IOBinding so as to bind memory which will be used as input(s)/output(s) for the CUDA
	///   Graph machinery to read from/write to (please see samples below).
	/// - While updating the input(s) for subsequent inference calls, the fresh input(s) need to be copied over to the
	///   corresponding CUDA memory location(s) of the bound `OrtValue` input(s). This is due to the fact that the
	///   “graph replay” will require reading inputs from the same CUDA virtual memory addresses.
	/// - Multi-threaded usage is currently not supported, i.e. `run()` MAY NOT be invoked on the same `Session` object
	///   from multiple threads while using CUDA Graphs.
	///
	/// > **NOTE**: The very first `run()` performs a variety of tasks under the hood like making CUDA memory
	/// > allocations, capturing the CUDA graph for the model, and then performing a graph replay to ensure that the
	/// > graph runs. Due to this, the latency associated with the first `run()` is bound to be high. Subsequent
	/// > `run()`s only perform graph replays of the graph captured and cached in the first `run()`.
	#[must_use]
	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cuda_graph", if enable { "1" } else { "0" });
		self
	}

	/// Whether to use strict mode in the `SkipLayerNormalization` implementation. The default and recommanded setting
	/// is `false`. If enabled, accuracy may improve slightly, but performance may decrease.
	#[must_use]
	pub fn with_skip_layer_norm_strict_mode(mut self, enable: bool) -> Self {
		self.options.set("enable_skip_layer_norm_strict_mode", if enable { "1" } else { "0" });
		self
	}

	/// TF32 is a math mode available on NVIDIA GPUs since Ampere. It allows certain float32 matrix multiplications and
	/// convolutions to run much faster on tensor cores with TensorFloat-32 reduced precision: float32 inputs are
	/// rounded with 10 bits of mantissa and results are accumulated with float32 precision.
	#[must_use]
	pub fn with_tf32(mut self, enable: bool) -> Self {
		self.options.set("use_tf32", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_prefer_nhwc(mut self, enable: bool) -> Self {
		self.options.set("prefer_nhwc", if enable { "1" } else { "0" });
		self
	}

	/// # Safety
	/// The provided `stream` must outlive the environment/session created with the execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.options.set("user_compute_stream", (stream as usize).to_string());
		self
	}

	#[must_use]
	pub fn with_attention_backend(mut self, flags: CUDAExecutionProviderAttentionBackend) -> Self {
		self.options.set("sdpa_kernel", flags.0.to_string());
		self
	}

	#[must_use]
	pub fn with_fuse_conv_bias(mut self, enable: bool) -> Self {
		self.options.set("fuse_conv_bias", if enable { "1" } else { "0" });
		self
	}

	// https://github.com/microsoft/onnxruntime/blob/ffceed9d44f2f3efb9dd69fa75fea51163c91d91/onnxruntime/core/providers/cuda/cuda_execution_provider_info.h#L48
	// https://github.com/microsoft/onnxruntime/blob/fe8a10caa40f64a8fbd144e7049cf5b14c65542d/onnxruntime/core/providers/cuda/cuda_execution_provider_info.cc#L17

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for CUDAExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<CUDAExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: CUDAExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for CUDAExecutionProvider {
	fn as_str(&self) -> &'static str {
		"CUDAExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "cuda"))]
		{
			use crate::AsPointer;

			let mut cuda_options: *mut ort_sys::OrtCUDAProviderOptionsV2 = core::ptr::null_mut();
			crate::ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)?];
			let ffi_options = self.options.to_ffi();

			let res = crate::ortsys![unsafe UpdateCUDAProviderOptions(
				cuda_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)];
			if let Err(e) = unsafe { crate::error::status_to_result(res) } {
				crate::ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(session_builder.ptr_mut(), cuda_options)];
			crate::ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
			return unsafe { crate::error::status_to_result(status) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
