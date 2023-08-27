#![allow(unused)]

use std::{
	collections::HashMap,
	ffi::{c_void, CString},
	os::raw::c_char,
	ptr
};

use crate::{
	error::status_to_result,
	ortsys,
	sys::{self, size_t, OrtArenaCfg},
	OrtApiError, OrtError, OrtResult
};

#[cfg(all(not(feature = "load-dynamic"), not(target_arch = "x86")))]
extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "acl")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "onednn")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "coreml")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut sys::OrtSessionOptions, flags: u32) -> sys::OrtStatusPtr;
	#[cfg(feature = "directml")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_DML(options: *mut sys::OrtSessionOptions, device_id: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "nnapi")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Nnapi(options: *mut sys::OrtSessionOptions, flags: u32) -> sys::OrtStatusPtr;
	#[cfg(feature = "tvm")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Tvm(options: *mut sys::OrtSessionOptions, opt_str: *const std::os::raw::c_char)
	-> sys::OrtStatusPtr;
}
#[cfg(all(not(feature = "load-dynamic"), target_arch = "x86"))]
extern "stdcall" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
}

#[derive(Debug, Clone, Default)]
pub struct CPUExecutionProviderOptions {
	pub use_arena: bool
}

/// The strategy for extending the device memory arena.
#[derive(Debug, Clone)]
pub enum ArenaExtendStrategy {
	/// (Default) Subsequent extensions extend by larger amounts (multiplied by powers of two)
	NextPowerOfTwo,
	/// Memory extends by the requested amount.
	SameAsRequested
}

impl Default for ArenaExtendStrategy {
	fn default() -> Self {
		Self::NextPowerOfTwo
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

#[derive(Debug, Clone)]
pub struct CUDAExecutionProviderOptions {
	pub device_id: u32,
	/// The size limit of the device memory arena in bytes. This size limit is only for the execution provider’s arena.
	/// The total device memory usage may be higher.
	pub gpu_mem_limit: usize,
	/// The strategy for extending the device memory arena. See [`ArenaExtendStrategy`].
	pub arena_extend_strategy: ArenaExtendStrategy,
	/// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	/// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	/// configuration (input shape, filter shape, etc.) in each `Conv` node. This option controlls the type of search
	/// done for cuDNN convolution algorithms. See [`CUDAExecutionProviderCuDNNConvAlgoSearch`] for more info.
	pub cudnn_conv_algo_search: CUDAExecutionProviderCuDNNConvAlgoSearch,
	/// Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false,
	/// there are race conditions and possibly better performance.
	pub do_copy_in_default_stream: bool,
	/// ORT leverages cuDNN for convolution operations and the first step in this process is to determine an
	/// “optimal” convolution algorithm to use while performing the convolution operation for the given input
	/// configuration (input shape, filter shape, etc.) in each `Conv` node. This sub-step involves querying cuDNN for a
	/// “workspace” memory size and have this allocated so that cuDNN can use this auxiliary memory while determining
	/// the “optimal” convolution algorithm to use.
	///
	/// When `cudnn_conv_use_max_workspace` is false, ORT will clamp the workspace size to 32 MB, which may lead to
	/// cuDNN selecting a suboptimal convolution algorithm. The recommended (and default) value is `true`.
	pub cudnn_conv_use_max_workspace: bool,
	/// ORT leverages cuDNN for convolution operations. While cuDNN only takes 4-D or 5-D tensors as input for
	/// convolution operations, dimension padding is needed if the input is a 3-D tensor. Given an input tensor of shape
	/// `[N, C, D]`, it can be padded to `[N, C, D, 1]` or `[N, C, 1, D]`. While both of these padding methods produce
	/// the same output, the performance may differ because different convolution algorithms are selected,
	/// especially on some devices such as A100. By default, the input is padded to `[N, C, D, 1]`. Set this option to
	/// true to instead use `[N, C, 1, D]`.
	pub cudnn_conv1d_pad_to_nc1d: bool,
	/// ORT supports the usage of CUDA Graphs to remove CPU overhead associated with launching CUDA kernels
	/// sequentially. To enable the usage of CUDA Graphs, set `enable_cuda_graph` to true.
	/// Currently, there are some constraints with regards to using the CUDA Graphs feature:
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
	pub enable_cuda_graph: bool,
	/// Whether to use strict mode in the `SkipLayerNormalization` implementation. The default and recommanded setting
	/// is `false`. If enabled, accuracy may improve slightly, but performance may decrease.
	pub enable_skip_layer_norm_strict_mode: bool
}

impl Default for CUDAExecutionProviderOptions {
	fn default() -> Self {
		Self {
			device_id: 0,
			gpu_mem_limit: usize::MAX,
			arena_extend_strategy: ArenaExtendStrategy::NextPowerOfTwo,
			cudnn_conv_algo_search: CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive,
			do_copy_in_default_stream: true,
			cudnn_conv_use_max_workspace: true,
			cudnn_conv1d_pad_to_nc1d: false,
			enable_cuda_graph: false,
			enable_skip_layer_norm_strict_mode: false
		}
	}
}

#[derive(Debug, Clone)]
pub struct TensorRTExecutionProviderOptions {
	pub device_id: u32,
	pub max_workspace_size: u32,
	pub max_partition_iterations: u32,
	pub min_subgraph_size: u32,
	pub fp16_enable: bool,
	pub int8_enable: bool,
	pub int8_calibration_table_name: String,
	pub int8_use_native_calibration_table: bool,
	pub dla_enable: bool,
	pub dla_core: u32,
	pub engine_cache_enable: bool,
	pub engine_cache_path: String,
	pub dump_subgraphs: bool,
	pub force_sequential_engine_build: bool,
	pub enable_context_memory_sharing: bool,
	pub layer_norm_fp32_fallback: bool,
	pub timing_cache_enable: bool,
	pub force_timing_cache: bool,
	pub detailed_build_log: bool,
	pub enable_build_heuristics: bool,
	pub enable_sparsity: bool,
	pub builder_optimization_level: u8,
	pub auxiliary_streams: i8,
	pub tactic_sources: String,
	pub extra_plugin_lib_paths: String,
	pub profile_min_shapes: String,
	pub profile_max_shapes: String,
	pub profile_opt_shapes: String
}

impl Default for TensorRTExecutionProviderOptions {
	fn default() -> Self {
		Self {
			device_id: 0,
			max_workspace_size: 1073741824,
			max_partition_iterations: 1000,
			min_subgraph_size: 1,
			fp16_enable: false,
			int8_enable: false,
			int8_calibration_table_name: String::new(),
			int8_use_native_calibration_table: false,
			dla_enable: false,
			dla_core: 0,
			engine_cache_enable: false,
			engine_cache_path: String::new(),
			dump_subgraphs: false,
			force_sequential_engine_build: false,
			enable_context_memory_sharing: false,
			layer_norm_fp32_fallback: false,
			timing_cache_enable: false,
			force_timing_cache: false,
			detailed_build_log: false,
			enable_build_heuristics: false,
			enable_sparsity: false,
			builder_optimization_level: 3,
			auxiliary_streams: -1,
			tactic_sources: String::new(),
			extra_plugin_lib_paths: String::new(),
			profile_min_shapes: String::new(),
			profile_max_shapes: String::new(),
			profile_opt_shapes: String::new()
		}
	}
}

#[derive(Debug, Clone)]
pub struct OpenVINOExecutionProviderOptions {
	/// Overrides the accelerator hardware type and precision with these values at runtime. If this option is not
	/// explicitly set, default hardware and precision specified during build time is used.
	pub device_type: Option<String>,
	/// Selects a particular hardware device for inference. If this option is not explicitly set, an arbitrary free
	/// device will be automatically selected by OpenVINO runtime.
	pub device_id: Option<String>,
	/// Overrides the accelerator default value of number of threads with this value at runtime. If this option is not
	/// explicitly set, default value of 8 is used during build time.
	pub num_threads: size_t,
	/// Explicitly specify the path to save and load the blobs enabling model caching feature.
	pub cache_dir: Option<String>,
	/// This option is only alvailable when OpenVINO EP is built with OpenCL flags enabled. It takes in the remote
	/// context i.e the `cl_context` address as a void pointer.
	pub context: *mut c_void,
	/// This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU).
	pub enable_opencl_throttling: bool,
	/// This option if enabled works for dynamic shaped models whose shape will be set dynamically based on the infer
	/// input image/data shape at run time in CPU. This gives best result for running multiple inferences with varied
	/// shaped images/data.
	pub enable_dynamic_shapes: bool,
	pub enable_vpu_fast_compile: bool
}

impl Default for OpenVINOExecutionProviderOptions {
	fn default() -> Self {
		Self {
			device_type: None,
			device_id: None,
			num_threads: 8,
			cache_dir: None,
			context: std::ptr::null_mut(),
			enable_opencl_throttling: false,
			enable_dynamic_shapes: false,
			enable_vpu_fast_compile: false
		}
	}
}

#[derive(Debug, Clone, Default)]
pub struct OneDNNExecutionProviderOptions {
	pub use_arena: bool
}

#[derive(Debug, Clone, Default)]
pub struct ACLExecutionProviderOptions {
	pub use_arena: bool
}

#[derive(Debug, Clone, Default)]
pub struct CoreMLExecutionProviderOptions {
	/// Limit CoreML to running on CPU only. This may decrease the performance but will provide reference output value
	/// without precision loss, which is useful for validation.
	pub use_cpu_only: bool,
	/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a Loop, Scan or If operator).
	pub enable_on_subgraph: bool,
	/// By default the CoreML EP will be enabled for all compatible Apple devices. Setting this option will only enable
	/// CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE). Note, enabling this option does not
	/// guarantee the entire model to be executed using ANE only.
	pub only_enable_device_with_ane: bool
}

#[derive(Debug, Clone, Default)]
pub struct DirectMLExecutionProviderOptions {
	pub device_id: u32
}

#[derive(Debug, Clone, Default)]
pub struct ROCmExecutionProviderOptions {
	pub device_id: i32,
	pub miopen_conv_exhaustive_search: bool,
	pub gpu_mem_limit: size_t,
	pub arena_extend_strategy: ArenaExtendStrategy,
	pub do_copy_in_default_stream: bool,
	pub user_compute_stream: Option<*mut c_void>,
	pub default_memory_arena_cfg: Option<*mut sys::OrtArenaCfg>,
	pub tunable_op_enable: bool,
	pub tunable_op_tuning_enable: bool
}

#[derive(Debug, Clone, Default)]
pub struct NNAPIExecutionProviderOptions {
	/// Use fp16 relaxation in NNAPI EP. This may improve performance but can also reduce accuracy due to the lower
	/// precision.
	pub use_fp16: bool,
	/// Use the NCHW layout in NNAPI EP. This is only available for Android API level 29 and higher. Please note that
	/// for now, NNAPI might have worse performance using NCHW compared to using NHWC.
	pub use_nchw: bool,
	/// Prevents NNAPI from using CPU devices. NNAPI is more efficient using GPU or NPU for execution, and NNAPI
	/// might fall back to its own CPU implementation for operations not supported by the GPU/NPU. However, the
	/// CPU implementation of NNAPI might be less efficient than the optimized versions of operators provided by
	/// ORT's default MLAS execution provider. It might be better to disable the NNAPI CPU fallback and instead
	/// use MLAS kernels. This option is only available after Android API level 29.
	pub disable_cpu: bool,
	/// Using CPU only in NNAPI EP, this may decrease the perf but will provide reference output value without precision
	/// loss, which is useful for validation. This option is only available for Android API level 29 and higher, and
	/// will be ignored for Android API level 28 and lower.
	pub cpu_only: bool
}
#[derive(Debug, Clone)]
enum QNNExecutionHTPPerformanceMode {
	/// Default mode.
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

impl QNNExecutionHTPPerformanceMode {
	fn as_str(&self) -> &'static str {
		match self {
			QNNExecutionHTPPerformanceMode::Default => "default",
			QNNExecutionHTPPerformanceMode::Burst => "burst",
			QNNExecutionHTPPerformanceMode::Balanced => "balanced",
			QNNExecutionHTPPerformanceMode::HighPerformance => "high_performance",
			QNNExecutionHTPPerformanceMode::HighPowerSaver => "high_power_saver",
			QNNExecutionHTPPerformanceMode::LowPowerSaver => "low_power_saver",
			QNNExecutionHTPPerformanceMode::LowBalanced => "low_balanced",
			QNNExecutionHTPPerformanceMode::PowerSaver => "power_saver",
			QNNExecutionHTPPerformanceMode::SustainedHighPerformance => "sustained_high_performance"
		}
	}
}
#[derive(Debug, Clone)]
pub struct QNNExecutionProviderOptions {
	/// The file path to QNN backend library.On Linux/Android: libQnnCpu.so for CPU backend, libQnnHtp.so for GPU
	/// backend.
	backend_path: String,
	/// true to enable QNN graph creation from cached QNN context file. If it's enabled: QNN EP will
	/// load from cached QNN context binary if it exist. It will generate a context binary file if it's not exist
	qnn_context_cache_enable: bool,
	/// explicitly provide the QNN context cache file. Default to model_file.onnx.bin if not provided.
	qnn_context_cache_path: Option<String>,
	/// QNN profiling level, options: "off", "basic", "detailed". Default to off.
	profiling_level: Option<String>,
	/// Allows client to set up RPC control latency in microseconds.
	rpc_control_latency: Option<u32>,
	/// QNN performance mode, options: "burst", "balanced", "default", "high_performance",
	/// "high_power_saver", "low_balanced", "low_power_saver", "power_saver", "sustained_high_performance". Default to
	/// "default".
	htp_performance_mode: Option<QNNExecutionHTPPerformanceMode>
}

impl Default for QNNExecutionProviderOptions {
	fn default() -> Self {
		Self {
			backend_path: String::from("libQnnHtp.so"),
			qnn_context_cache_enable: false,
			qnn_context_cache_path: Some(String::from("model_file.onnx.bin")),
			profiling_level: Some(String::from("off")),
			rpc_control_latency: Some(10),
			htp_performance_mode: Some(QNNExecutionHTPPerformanceMode::Default)
		}
	}
}

macro_rules! get_ep_register {
	($symbol:ident($($id:ident: $type:ty),*) -> $rt:ty) => {
		#[cfg(feature = "load-dynamic")]
		#[allow(non_snake_case)]
		let $symbol = unsafe {
			use crate::G_ORT_LIB;
			let dylib = *G_ORT_LIB
				.lock()
				.expect("failed to acquire ONNX Runtime dylib lock; another thread panicked?")
				.get_mut();
			let symbol: Result<
				libloading::Symbol<unsafe extern "C" fn($($id: $type),*) -> $rt>,
				libloading::Error
			> = (*dylib).get(stringify!($symbol).as_bytes());
			match symbol {
				Ok(symbol) => symbol,
				Err(e) => {
					return Err(OrtError::DlLoad { symbol: stringify!($symbol), error: e.to_string() });
				}
			}
		};
	};
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TVMExecutorType {
	GraphExecutor,
	VirtualMachine
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TVMTuningType {
	AutoTVM,
	Ansor
}

#[derive(Default, Debug, Clone)]
pub struct TVMExecutionProviderOptions {
	/// Executor type used by TVM. There is a choice between two types, `GraphExecutor` and `VirtualMachine`. Default is
	/// [`TVMExecutorType::VirtualMachine`].
	pub executor: Option<TVMExecutorType>,
	/// Path to folder with set of files (`.ro-`, `.so`/`.dll`-files and weights) obtained after model tuning.
	pub so_folder: Option<String>,
	/// Whether or not to perform a hash check on the model obtained in the `so_folder`.
	pub check_hash: Option<bool>,
	/// A path to a file that contains the pre-computed hash for the ONNX model located in the `so_folder` for checking
	/// when `check_hash` is `Some(true)`.
	pub hash_file_path: Option<String>,
	pub target: Option<String>,
	pub target_host: Option<String>,
	pub opt_level: Option<usize>,
	/// Whether or not all model weights are kept on compilation stage, otherwise they are downloaded on each inference.
	/// `true` is recommended for best performance and is the default.
	pub freeze_weights: Option<bool>,
	pub to_nhwc: Option<bool>,
	pub tuning_type: Option<TVMTuningType>,
	/// Path to AutoTVM or Ansor tuning file which gives specifications for given model and target for the best
	/// performance.
	pub tuning_file_path: Option<String>,
	pub input_names: Option<String>,
	pub input_shapes: Option<String>
}

/// Execution provider container. See [the ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/) for more
/// info on execution providers. Execution providers are actually registered via the functions [`crate::SessionBuilder`]
/// (per-session) or [`EnvBuilder`](crate::environment::EnvBuilder) (default for all sessions in an environment).
#[derive(Debug, Clone)]
pub enum ExecutionProvider {
	CPU(CPUExecutionProviderOptions),
	CUDA(CUDAExecutionProviderOptions),
	TensorRT(TensorRTExecutionProviderOptions),
	OpenVINO(OpenVINOExecutionProviderOptions),
	ACL(ACLExecutionProviderOptions),
	OneDNN(OneDNNExecutionProviderOptions),
	CoreML(CoreMLExecutionProviderOptions),
	DirectML(DirectMLExecutionProviderOptions),
	ROCm(ROCmExecutionProviderOptions),
	NNAPI(NNAPIExecutionProviderOptions),
	QNN(QNNExecutionProviderOptions),
	TVM(TVMExecutionProviderOptions)
}

macro_rules! map_keys {
	($($fn_name:ident = $ex:expr),*) => {
		{
			let mut keys = Vec::<CString>::new();
			let mut values = Vec::<CString>::new();
			$(
				keys.push(CString::new(stringify!($fn_name)).unwrap());
				values.push(CString::new(($ex).to_string().as_str()).unwrap());
			)*
			assert_eq!(keys.len(), values.len()); // sanity check
			let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
			let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
			(key_ptrs, value_ptrs, keys.len(), keys, values)
		}
	};
}

#[inline]
fn bool_as_int(x: bool) -> i32 {
	match x {
		true => 1,
		false => 0
	}
}

impl ExecutionProvider {
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::CPU(_) => "CPUExecutionProvider",
			Self::CUDA(_) => "CUDAExecutionProvider",
			Self::TensorRT(_) => "TensorrtExecutionProvider",
			Self::OpenVINO(_) => "OpenVINOExecutionProvider",
			Self::ACL(_) => "AclExecutionProvider",
			Self::OneDNN(_) => "DnnlExecutionProvider",
			Self::CoreML(_) => "CoreMLExecutionProvider",
			Self::DirectML(_) => "DmlExecutionProvider",
			Self::ROCm(_) => "ROCmExecutionProvider",
			Self::NNAPI(_) => "NnapiExecutionProvider",
			Self::QNN(_) => "QNNExecutionProvider",
			Self::TVM(_) => "TvmExecutionProvider"
		}
	}

	/// Returns `true` if this execution provider is available, `false` otherwise.
	/// The CPU execution provider will always be available.
	pub fn is_available(&self) -> bool {
		let mut providers: *mut *mut c_char = std::ptr::null_mut();
		let mut num_providers = 0;
		if status_to_result(ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers)]).is_err() {
			return false;
		}

		for i in 0..num_providers {
			let avail = unsafe { std::ffi::CStr::from_ptr(*providers.offset(i as isize)) }
				.to_string_lossy()
				.into_owned();
			if self.as_str() == avail {
				let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
				return true;
			}
		}

		let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
		false
	}

	pub(crate) fn apply(&self, session_options: *mut sys::OrtSessionOptions) -> OrtResult<()> {
		match &self {
			&Self::CPU(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr);
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_CPU(session_options, options.use_arena.into()) })
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "cuda"))]
			&Self::CUDA(options) => {
				let mut cuda_options: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
				status_to_result(ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)]).map_err(OrtError::ExecutionProvider)?;
				let (key_ptrs, value_ptrs, len, keys, values) = map_keys! {
					device_id = options.device_id,
					arena_extend_strategy = match options.arena_extend_strategy {
						ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
						ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
					},
					cudnn_conv_algo_search = match options.cudnn_conv_algo_search {
						CUDAExecutionProviderCuDNNConvAlgoSearch::Exhaustive => "EXHAUSTIVE",
						CUDAExecutionProviderCuDNNConvAlgoSearch::Heuristic => "HEURISTIC",
						CUDAExecutionProviderCuDNNConvAlgoSearch::Default => "DEFAULT"
					},
					do_copy_in_default_stream = bool_as_int(options.do_copy_in_default_stream),
					cudnn_conv_use_max_workspace = bool_as_int(options.cudnn_conv_use_max_workspace),
					cudnn_conv1d_pad_to_nc1d = bool_as_int(options.cudnn_conv1d_pad_to_nc1d),
					enable_cuda_graph = bool_as_int(options.enable_cuda_graph),
					enable_skip_layer_norm_strict_mode = bool_as_int(options.enable_skip_layer_norm_strict_mode)
				};
				if let Err(e) = status_to_result(ortsys![unsafe UpdateCUDAProviderOptions(cuda_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), len as _)])
					.map_err(OrtError::ExecutionProvider)
				{
					ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
					std::mem::drop((keys, values));
					return Err(e);
				}

				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, cuda_options)];
				ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
				std::mem::drop((keys, values));
				status_to_result(status).map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "tensorrt"))]
			&Self::TensorRT(options) => {
				let mut trt_options: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
				status_to_result(ortsys![unsafe CreateTensorRTProviderOptions(&mut trt_options)]).map_err(OrtError::ExecutionProvider)?;
				let (key_ptrs, value_ptrs, len, keys, values) = map_keys! {
					device_id = options.device_id,
					trt_max_workspace_size = options.max_workspace_size,
					trt_max_partition_iterations = options.max_partition_iterations,
					trt_min_subgraph_size = options.min_subgraph_size,
					trt_fp16_enable = bool_as_int(options.fp16_enable),
					trt_int8_enable = bool_as_int(options.int8_enable),
					trt_int8_calibration_table_name = options.int8_calibration_table_name,
					trt_dla_enable = bool_as_int(options.dla_enable),
					trt_dla_core = options.dla_core,
					trt_engine_cache_enable = bool_as_int(options.engine_cache_enable),
					trt_engine_cache_path = options.engine_cache_path,
					trt_dump_subgraphs = bool_as_int(options.dump_subgraphs),
					trt_force_sequential_engine_build = bool_as_int(options.force_sequential_engine_build),
					trt_context_memory_sharing_enable = bool_as_int(options.enable_context_memory_sharing),
					trt_layer_norm_fp32_fallback = bool_as_int(options.layer_norm_fp32_fallback),
					trt_timing_cache_enable = bool_as_int(options.timing_cache_enable),
					trt_force_timing_cache = bool_as_int(options.force_timing_cache),
					trt_detailed_build_log = bool_as_int(options.detailed_build_log),
					trt_build_heuristics_enable = bool_as_int(options.enable_build_heuristics),
					trt_sparsity_enable = bool_as_int(options.enable_sparsity),
					trt_builder_optimization_level = options.builder_optimization_level,
					trt_auxiliary_streams = options.auxiliary_streams,
					trt_tactic_sources = options.tactic_sources,
					trt_extra_plugin_lib_paths = options.extra_plugin_lib_paths,
					trt_profile_min_shapes = options.profile_min_shapes,
					trt_profile_max_shapes = options.profile_max_shapes,
					trt_profile_opt_shapes = options.profile_opt_shapes
				};
				if let Err(e) = status_to_result(ortsys![unsafe UpdateTensorRTProviderOptions(trt_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), len as _)])
					.map_err(OrtError::ExecutionProvider)
				{
					ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
					std::mem::drop((keys, values));
					return Err(e);
				}

				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(session_options, trt_options)];
				ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
				std::mem::drop((keys, values));
				status_to_result(status).map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "acl"))]
			&Self::ACL(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr);
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_ACL(session_options, options.use_arena.into()) })
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "onednn"))]
			&Self::OneDNN(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr);
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, options.use_arena.into()) })
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "coreml"))]
			&Self::CoreML(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut sys::OrtSessionOptions, flags: u32) -> sys::OrtStatusPtr);
				let mut flags = 0;
				if options.use_cpu_only {
					flags |= 0x001;
				}
				if options.enable_on_subgraph {
					flags |= 0x002;
				}
				if options.only_enable_device_with_ane {
					flags |= 0x004;
				}
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, flags) }).map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "directml"))]
			&Self::DirectML(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_DML(options: *mut sys::OrtSessionOptions, device_id: std::os::raw::c_int) -> sys::OrtStatusPtr);
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_DML(session_options, options.device_id as _) })
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "rocm"))]
			&Self::ROCm(options) => {
				let rocm_options = sys::OrtROCMProviderOptions {
					device_id: options.device_id,
					miopen_conv_exhaustive_search: bool_as_int(options.miopen_conv_exhaustive_search),
					gpu_mem_limit: options.gpu_mem_limit as _,
					arena_extend_strategy: match options.arena_extend_strategy {
						ArenaExtendStrategy::NextPowerOfTwo => 0,
						ArenaExtendStrategy::SameAsRequested => 1
					},
					do_copy_in_default_stream: bool_as_int(options.do_copy_in_default_stream),
					has_user_compute_stream: bool_as_int(options.user_compute_stream.is_some()),
					user_compute_stream: options.user_compute_stream.unwrap_or(ptr::null_mut()),
					default_memory_arena_cfg: options.default_memory_arena_cfg.unwrap_or(ptr::null_mut()),
					tunable_op_enable: bool_as_int(options.tunable_op_enable),
					tunable_op_tuning_enable: bool_as_int(options.tunable_op_tuning_enable)
				};
				status_to_result(ortsys![unsafe SessionOptionsAppendExecutionProvider_ROCM(session_options, &rocm_options as *const _)])
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "nnapi"))]
			&Self::NNAPI(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_Nnapi(options: *mut sys::OrtSessionOptions, flags: u32) -> sys::OrtStatusPtr);
				let mut flags = 0;
				if options.use_fp16 {
					flags |= 0x001;
				}
				if options.use_nchw {
					flags |= 0x002;
				}
				if options.disable_cpu {
					flags |= 0x004;
				}
				if options.cpu_only {
					flags |= 0x008;
				}
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, flags) }).map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "openvino"))]
			&Self::OpenVINO(options) => {
				let openvino_options = sys::OrtOpenVINOProviderOptions {
					device_type: options
						.device_type
						.clone()
						.map(|x| x.as_bytes().as_ptr() as *const c_char)
						.unwrap_or_else(ptr::null),
					device_id: options
						.device_id
						.clone()
						.map(|x| x.as_bytes().as_ptr() as *const c_char)
						.unwrap_or_else(ptr::null),
					num_of_threads: options.num_threads,
					cache_dir: options
						.cache_dir
						.clone()
						.map(|x| x.as_bytes().as_ptr() as *const c_char)
						.unwrap_or_else(ptr::null),
					context: options.context,
					enable_opencl_throttling: bool_as_int(options.enable_opencl_throttling) as _,
					enable_dynamic_shapes: bool_as_int(options.enable_dynamic_shapes) as _,
					enable_vpu_fast_compile: bool_as_int(options.enable_vpu_fast_compile) as _
				};
				status_to_result(ortsys![unsafe SessionOptionsAppendExecutionProvider_OpenVINO(session_options, &openvino_options as *const _)])
					.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "qnn"))]
			&Self::QNN(options) => {
				let (key_ptrs, value_ptrs, len, keys, values) = map_keys! {
					backend_path = options.backend_path,
					profiling_level = options.profiling_level.clone().unwrap_or("off".to_string()),
					qnn_context_cache_enable = bool_as_int(options.qnn_context_cache_enable),
					qnn_context_cache_path = options.qnn_context_cache_path.clone().unwrap_or("model_file.onnx.bin".to_string()),
					htp_performance_mode = options.htp_performance_mode.clone().unwrap_or(QNNExecutionHTPPerformanceMode::Default).as_str(),
					rpc_control_latency = options.rpc_control_latency.unwrap_or(10)
				};
				let name = CString::new("QNN").unwrap();
				status_to_result(ortsys![unsafe SessionOptionsAppendExecutionProvider(
					session_options,
					name.as_ptr(),
					key_ptrs.as_ptr(),
					value_ptrs.as_ptr(),
					len as _,
				)])
				.map_err(OrtError::ExecutionProvider)?;
			}
			#[cfg(any(feature = "load-dynamic", feature = "tvm"))]
			&Self::TVM(options) => {
				get_ep_register!(OrtSessionOptionsAppendExecutionProvider_Tvm(options: *mut sys::OrtSessionOptions, opt_str: *const std::os::raw::c_char) -> sys::OrtStatusPtr);
				let mut option_string = Vec::new();
				if let Some(check_hash) = options.check_hash {
					option_string.push(format!("check_hash:{}", if check_hash { "True" } else { "False" }));
				}
				if let Some(executor) = options.executor {
					option_string.push(format!(
						"executor:{}",
						match executor {
							TVMExecutorType::GraphExecutor => "graph",
							TVMExecutorType::VirtualMachine => "vm"
						}
					));
				}
				if let Some(freeze_weights) = options.freeze_weights {
					option_string.push(format!("freeze_weights:{}", if freeze_weights { "True" } else { "False" }));
				}
				if let Some(hash_file_path) = options.hash_file_path.as_ref() {
					option_string.push(format!("hash_file_path:{hash_file_path}"));
				}
				if let Some(input_names) = options.input_names.as_ref() {
					option_string.push(format!("input_names:{input_names}"));
				}
				if let Some(input_shapes) = options.input_shapes.as_ref() {
					option_string.push(format!("input_shapes:{input_shapes}"));
				}
				if let Some(opt_level) = options.opt_level {
					option_string.push(format!("opt_level:{opt_level}"));
				}
				if let Some(so_folder) = options.so_folder.as_ref() {
					option_string.push(format!("so_folder:{so_folder}"));
				}
				if let Some(target) = options.target.as_ref() {
					option_string.push(format!("target:{target}"));
				}
				if let Some(target_host) = options.target_host.as_ref() {
					option_string.push(format!("target_host:{target_host}"));
				}
				if let Some(to_nhwc) = options.to_nhwc {
					option_string.push(format!("to_nhwc:{}", if to_nhwc { "True" } else { "False" }));
				}
				let options_string = CString::new(option_string.join(",")).unwrap();
				status_to_result(unsafe { OrtSessionOptionsAppendExecutionProvider_Tvm(session_options, options_string.as_ptr()) })
					.map_err(OrtError::ExecutionProvider)?;
			}
			_ => return Err(OrtError::ExecutionProviderNotRegistered(self.as_str()))
		}
		Ok(())
	}
}

#[tracing::instrument(skip_all)]
pub(crate) fn apply_execution_providers(options: *mut sys::OrtSessionOptions, execution_providers: impl AsRef<[ExecutionProvider]>) {
	let mut fallback_to_cpu = true;
	for ex in execution_providers.as_ref() {
		if let Err(e) = ex.apply(options) {
			if let &OrtError::ExecutionProviderNotRegistered(_) = &e {
				tracing::debug!("{}", e);
			} else {
				tracing::warn!("An error occurred when attempting to register `{}`: {e}", ex.as_str());
			}
		} else {
			tracing::info!("Successfully registered `{}`", ex.as_str());
			fallback_to_cpu = false;
		}
	}
	if fallback_to_cpu {
		tracing::warn!("No execution providers registered successfully. Falling back to CPU.");
	}
}
