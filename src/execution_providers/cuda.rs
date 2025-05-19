use alloc::string::ToString;
use core::ops::BitOr;

use super::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

// https://github.com/microsoft/onnxruntime/blob/ffceed9d44f2f3efb9dd69fa75fea51163c91d91/onnxruntime/contrib_ops/cpu/bert/attention_common.h#L160-L171
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CUDAAttentionBackend(u32);

impl CUDAAttentionBackend {
	pub const FLASH_ATTENTION: Self = Self(1 << 0);
	pub const EFFICIENT_ATTENTION: Self = Self(1 << 1);
	pub const TRT_FUSED_ATTENTION: Self = Self(1 << 2);
	pub const CUDNN_FLASH_ATTENTION: Self = Self(1 << 3);
	pub const MATH: Self = Self(1 << 4);

	pub const TRT_FLASH_ATTENTION: Self = Self(1 << 5);
	pub const TRT_CROSS_ATTENTION: Self = Self(1 << 6);
	pub const TRT_CAUSAL_ATTENTION: Self = Self(1 << 7);

	pub const LEAN_ATTENTION: Self = Self(1 << 8);

	pub fn none() -> Self {
		CUDAAttentionBackend(0)
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

impl BitOr for CUDAAttentionBackend {
	type Output = Self;
	fn bitor(self, rhs: Self) -> Self::Output {
		Self(rhs.0 | self.0)
	}
}

/// The type of search done for cuDNN convolution algorithms.
#[derive(Debug, Clone)]
pub enum CuDNNConvAlgorithmSearch {
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
	/// [`Exhaustive`]: CuDNNConvAlgorithmSearch::Exhaustive
	/// [`Heuristic`]: CuDNNConvAlgorithmSearch::Heuristic
	Default
}

impl Default for CuDNNConvAlgorithmSearch {
	fn default() -> Self {
		Self::Exhaustive
	}
}

/// [CUDA execution provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for NVIDIA
/// CUDA-enabled GPUs.
#[derive(Debug, Default, Clone)]
pub struct CUDAExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; CUDAExecutionProvider);

impl CUDAExecutionProvider {
	/// Configures which device the EP should use.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_device_id(0).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// Configure the size limit of the device memory arena in bytes.
	///
	/// This only controls how much memory can be allocated to the *arena* - actual memory usage may be higher due to
	/// internal CUDA allocations, like those required for different [`CuDNNConvAlgorithmSearch`] options.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_memory_limit(2 * 1024 * 1024 * 1024).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_memory_limit(mut self, limit: usize) -> Self {
		self.options.set("gpu_mem_limit", limit.to_string());
		self
	}

	/// Configure the strategy for extending the device's memory arena.
	///
	/// ```
	/// # use ort::{execution_providers::{cuda::CUDAExecutionProvider, ArenaExtendStrategy}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default()
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

	/// Controls the search mode used to select a kernel for `Conv` nodes.
	///
	/// cuDNN, the library used by ONNX Runtime's CUDA EP for many operations, provides many different implementations
	/// of the `Conv` node. Each of these implementations has different performance characteristics depending on the
	/// exact hardware and model/input size used. This option controls how cuDNN should determine which implementation
	/// to use.
	///
	/// The default search algorithm, [`Exhaustive`][exh], will benchmark all available implementations and use the most
	/// performant one. This option is very resource intensive (both computationally on first run and peak-memory-wise),
	/// but ensures best performance. It is roughly equivalent to setting `torch.backends.cudnn.benchmark = True` with
	/// PyTorch. See also [`CUDAExecutionProvider::with_conv_max_workspace`] to configure how much memory the exhaustive
	/// search can use (the default is unlimited).
	///
	/// A less resource-intensive option is [`Heuristic`][heu]. Rather than benchmarking every implementation,
	/// an optimal implementation is chosen based on a set of heuristics, thus saving compute. [`Heuristic`][heu] should
	/// generally choose an optimal convolution algorithm, except in some corner cases.
	///
	/// [`Default`][def] can also be passed to instruct cuDNN to always use the default implementation (which is rarely
	/// the most optimal). Note that the "Default" here refers to the **default convolution algorithm** being used, it
	/// is not the *default behavior* (that would be [`Exhaustive`][exh]).
	///
	/// ```
	/// # use ort::{execution_providers::cuda::{CUDAExecutionProvider, CuDNNConvAlgorithmSearch}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default()
	/// 	.with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	///
	/// [exh]: CuDNNConvAlgorithmSearch::Exhaustive
	/// [heu]: CuDNNConvAlgorithmSearch::Heuristic
	/// [def]: CuDNNConvAlgorithmSearch::Default
	#[must_use]
	pub fn with_conv_algorithm_search(mut self, search: CuDNNConvAlgorithmSearch) -> Self {
		self.options.set(
			"cudnn_conv_algo_search",
			match search {
				CuDNNConvAlgorithmSearch::Exhaustive => "EXHAUSTIVE",
				CuDNNConvAlgorithmSearch::Heuristic => "HEURISTIC",
				CuDNNConvAlgorithmSearch::Default => "DEFAULT"
			}
		);
		self
	}

	/// Configure whether the [`Exhaustive`][CuDNNConvAlgorithmSearch::Exhaustive] search can use as much memory as it
	/// needs.
	///
	/// The default is `true`. When `false`, the memory used for the search is limited to 32 MB, which will impact its
	/// ability to find an optimal convolution algorithm.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_conv_max_workspace(false).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_conv_max_workspace(mut self, enable: bool) -> Self {
		self.options.set("cudnn_conv_use_max_workspace", if enable { "1" } else { "0" });
		self
	}

	// Here once lied `do_copy_in_default_stream`. After reading through upstream it doesn't seem like this option is
	// used anymore, so the setter here was removed to reduce confusion.

	/// Configure whether or not to pad 3-dimensional convolutions to `[N, C, 1, D]` (as opposed to the default `[N, C,
	/// D, 1]`).
	///
	/// Enabling this option might significantly improve performance on devices like the A100. This does not affect
	/// convolution operations that do not use 3-dimensional input shapes, or the *result* of such operations.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_conv1d_pad_to_nc1d(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_conv1d_pad_to_nc1d(mut self, enable: bool) -> Self {
		self.options.set("cudnn_conv1d_pad_to_nc1d", if enable { "1" } else { "0" });
		self
	}

	/// Configures whether to create a CUDA graph.
	///
	/// CUDA graphs eliminate the overhead of launching kernels sequentially by capturing the launch sequence into a
	/// graph that is 'replayed' across runs, reducing CPU overhead and possibly improving performance.
	///
	/// Using CUDA graphs comes with limitations, notably:
	/// - Models with control flow operators (like `If`, `Loop`, or `Scan`) are not supported.
	/// - Input/output shapes cannot change across inference calls.
	/// - The address of inputs/outputs cannot change across inference calls, so
	///   [`IoBinding`](crate::io_binding::IoBinding) must be used.
	/// - `Session`s using CUDA graphs are technically not `Send` or `Sync`.
	///
	/// Consult the [ONNX Runtime documentation on CUDA graphs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview) for more information.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_cuda_graph(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cuda_graph", if enable { "1" } else { "0" });
		self
	}

	/// Enable 'strict' mode for `SkipLayerNorm` nodes (created via fusion of `Add` & `LayerNorm` nodes).
	///
	/// `SkipLayerNorm`'s strict mode trades performance for accuracy. The default is `false` (strict mode disabled).
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_skip_layer_norm_strict_mode(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_skip_layer_norm_strict_mode(mut self, enable: bool) -> Self {
		self.options.set("enable_skip_layer_norm_strict_mode", if enable { "1" } else { "0" });
		self
	}

	/// Enable the usage of the reduced-precision [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/)
	/// format for matrix multiplications & convolutions.
	///
	/// TensorFloat-32 is a reduced-precision floating point format available on NVIDIA GPUs since the Ampere
	/// microarchitecture. It allows `MatMul` & `Conv` to run much faster on Ampere's Tensor cores. This option is
	/// **disabled** by default.
	///
	/// This option is roughly equivalent to `torch.backends.cudnn.allow_tf32 = True` &
	/// `torch.backends.cuda.matmul.allow_tf32 = True` or `torch.set_float32_matmul_precision("medium")` in PyTorch.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_tf32(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_tf32(mut self, enable: bool) -> Self {
		self.options.set("use_tf32", if enable { "1" } else { "0" });
		self
	}

	/// Configure whether to prefer `[N, H, W, C]` layout operations over the default `[N, C, H, W]` layout.
	///
	/// Tensor cores usually operate more efficiently with the NHWC layout, so enabling this option for
	/// convolution-heavy models on Tensor core-enabled GPUs may provide a significant performance improvement.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::CUDAExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default().with_prefer_nhwc(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_prefer_nhwc(mut self, enable: bool) -> Self {
		self.options.set("prefer_nhwc", if enable { "1" } else { "0" });
		self
	}

	/// Use a custom CUDA device stream rather than the default one.
	///
	/// # Safety
	/// The provided `stream` must outlive the environment/session configured to use this execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.options.set("user_compute_stream", (stream as usize).to_string());
		self
	}

	/// Configures the available backends used for `Attention` nodes.
	///
	/// ```
	/// # use ort::{execution_providers::cuda::{CUDAExecutionProvider, CUDAAttentionBackend}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CUDAExecutionProvider::default()
	/// 	.with_attention_backend(CUDAAttentionBackend::FLASH_ATTENTION | CUDAAttentionBackend::TRT_FUSED_ATTENTION)
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_attention_backend(mut self, flags: CUDAAttentionBackend) -> Self {
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
}

impl ExecutionProvider for CUDAExecutionProvider {
	fn name(&self) -> &'static str {
		"CUDAExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "cuda"))]
		{
			use core::ptr;

			use crate::{AsPointer, ortsys, util};

			let mut cuda_options: *mut ort_sys::OrtCUDAProviderOptionsV2 = ptr::null_mut();
			ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)?];
			let _guard = util::run_on_drop(|| {
				ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
			});

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe UpdateCUDAProviderOptions(
				cuda_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(session_builder.ptr_mut(), cuda_options)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}

#[cfg(windows)]
pub const CUDA_DYLIBS: &[&str] = &["cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudart64_12.dll"];
#[cfg(not(windows))]
pub const CUDA_DYLIBS: &[&str] = &["libcublasLt.so.12", "libcublas.so.12", "libnvrtc.so.12", "libcurand.so.10", "libcufft.so.11", "libcudart.so.12"];

#[cfg(windows)]
pub const CUDNN_DYLIBS: &[&str] = &[
	"cudnn_engines_runtime_compiled64_9.dll",
	"cudnn_engines_precompiled64_9.dll",
	"cudnn_heuristic64_9.dll",
	"cudnn_ops64_9.dll",
	"cudnn_adv64_9.dll",
	"cudnn_graph64_9.dll",
	"cudnn64_9.dll"
];
#[cfg(not(windows))]
pub const CUDNN_DYLIBS: &[&str] = &[
	"libcudnn_engines_runtime_compiled.so.9",
	"libcudnn_engines_precompiled.so.9",
	"libcudnn_heuristic.so.9",
	"libcudnn_ops.so.9",
	"libcudnn_adv.so.9",
	"libcudnn_graph.so.9",
	"libcudnn.so.9"
];

/// Preload the dylibs required by CUDA/cuDNN.
///
/// This attempts to load all dynamic libraries required by the CUDA execution provider from the given CUDA and cuDNN
/// directories, if they are provided. Passing `None` will prevent preloading binaries for that component. This function
/// will immediately return with an error when a library fails to load, without attempting to load the rest of the
/// libraries.
///
/// Preloading a library in this way will prioritize it in the search order when the CUDA EP attempts to load its
/// dependencies, effectively allowing you to customize the CUDA install path without modifying the `PATH` environment
/// variable. Note that this function intentionally leaks memory; see [`crate::util::preload_dylib`] for more
/// information.
///
/// ```
/// # use std::path::Path;
/// use ort::execution_providers::cuda;
///
/// let cuda_root = Path::new(r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"#);
/// let cudnn_root = Path::new(r#"D:\cudnn_9.8.0"#);
///
/// // Load CUDA & cuDNN
/// let _ = cuda::preload_dylibs(Some(cuda_root), Some(cudnn_root));
///
/// // Only preload cuDNN
/// let _ = cuda::preload_dylibs(None, Some(cudnn_root));
/// ```
#[cfg(feature = "load-dynamic")]
#[cfg_attr(docsrs, doc(cfg(feature = "load-dynamic")))]
pub fn preload_dylibs(cuda_root_dir: Option<&std::path::Path>, cudnn_root_dir: Option<&std::path::Path>) -> Result<()> {
	use crate::util::preload_dylib;
	if let Some(cuda_root_dir) = cuda_root_dir {
		for dylib in CUDA_DYLIBS {
			preload_dylib(cuda_root_dir.join(dylib)).map_err(|e| crate::Error::new(format!("Failed to preload `{dylib}`: {e}")))?;
		}
	}
	if let Some(cudnn_root_dir) = cudnn_root_dir {
		for dylib in CUDNN_DYLIBS {
			preload_dylib(cudnn_root_dir.join(dylib)).map_err(|e| crate::Error::new(format!("Failed to preload `{dylib}`: {e}")))?;
		}
	}
	Ok(())
}
