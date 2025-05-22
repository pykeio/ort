#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

extern crate core;

#[doc(hidden)]
#[cfg(feature = "std")]
pub mod internal;

pub const ORT_API_VERSION: u32 = 22;

pub use core::ffi::{c_char, c_int, c_ulong, c_ulonglong, c_ushort, c_void};

#[cfg(target_os = "windows")]
pub type ortchar = c_ushort;
#[cfg(not(target_os = "windows"))]
pub type ortchar = c_char;

#[repr(i32)]
#[doc = " Copied from TensorProto::DataType\n Currently, Ort doesn't support complex64, complex128"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ONNXTensorElementDataType {
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 = 14,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 = 15,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 = 16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN = 17,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ = 18,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 = 19,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = 20,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 = 21,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 = 22
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ONNXType {
	ONNX_TYPE_UNKNOWN = 0,
	ONNX_TYPE_TENSOR = 1,
	ONNX_TYPE_SEQUENCE = 2,
	ONNX_TYPE_MAP = 3,
	ONNX_TYPE_OPAQUE = 4,
	ONNX_TYPE_SPARSETENSOR = 5,
	ONNX_TYPE_OPTIONAL = 6
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtSparseFormat {
	ORT_SPARSE_UNDEFINED = 0,
	ORT_SPARSE_COO = 1,
	ORT_SPARSE_CSRC = 2,
	ORT_SPARSE_BLOCK_SPARSE = 4
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtSparseIndicesFormat {
	ORT_SPARSE_COO_INDICES = 0,
	ORT_SPARSE_CSR_INNER_INDICES = 1,
	ORT_SPARSE_CSR_OUTER_INDICES = 2,
	ORT_SPARSE_BLOCK_SPARSE_INDICES = 3
}
#[repr(i32)]
#[doc = " \\brief Logging severity levels\n\n In typical API usage, specifying a logging severity level specifies the minimum severity of log messages to show."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtLoggingLevel {
	#[doc = "< Verbose informational messages (least severe)."]
	ORT_LOGGING_LEVEL_VERBOSE = 0,
	#[doc = "< Informational messages."]
	ORT_LOGGING_LEVEL_INFO = 1,
	#[doc = "< Warning messages."]
	ORT_LOGGING_LEVEL_WARNING = 2,
	#[doc = "< Error messages."]
	ORT_LOGGING_LEVEL_ERROR = 3,
	#[doc = "< Fatal error messages (most severe)."]
	ORT_LOGGING_LEVEL_FATAL = 4
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtErrorCode {
	ORT_OK = 0,
	ORT_FAIL = 1,
	ORT_INVALID_ARGUMENT = 2,
	ORT_NO_SUCHFILE = 3,
	ORT_NO_MODEL = 4,
	ORT_ENGINE_ERROR = 5,
	ORT_RUNTIME_EXCEPTION = 6,
	ORT_INVALID_PROTOBUF = 7,
	ORT_MODEL_LOADED = 8,
	ORT_NOT_IMPLEMENTED = 9,
	ORT_INVALID_GRAPH = 10,
	ORT_EP_FAIL = 11
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtOpAttrType {
	ORT_OP_ATTR_UNDEFINED = 0,
	ORT_OP_ATTR_INT = 1,
	ORT_OP_ATTR_INTS = 2,
	ORT_OP_ATTR_FLOAT = 3,
	ORT_OP_ATTR_FLOATS = 4,
	ORT_OP_ATTR_STRING = 5,
	ORT_OP_ATTR_STRINGS = 6
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtEnv {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtStatus {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtMemoryInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtIoBinding {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtSession {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtValue {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtRunOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTypeInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTensorTypeAndShapeInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtMapTypeInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtSequenceTypeInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtOptionalTypeInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtSessionOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCustomOpDomain {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtModelMetadata {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtThreadPoolParams {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtThreadingOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtArenaCfg {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtPrepackedWeightsContainer {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTensorRTProviderOptionsV2 {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtNvTensorRtRtxProviderOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCUDAProviderOptionsV2 {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCANNProviderOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtDnnlProviderOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtOp {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtOpAttr {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtLogger {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtShapeInferContext {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtLoraAdapter {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtValueInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtNode {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtGraph {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtModel {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtModelCompilationOptions {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtHardwareDevice {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtEpDevice {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtKeyValuePairs {
	_unused: [u8; 0]
}
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
#[must_use = "statuses must be freed with `OrtApi::ReleaseStatus` if they are not null"]
pub struct OrtStatusPtr(pub *mut OrtStatus);
impl Default for OrtStatusPtr {
	fn default() -> Self {
		OrtStatusPtr(core::ptr::null_mut())
	}
}
#[doc = " \\brief Memory allocation interface\n\n Structure of function pointers that defines a memory allocator. This can be created and filled in by the user for custom allocators.\n\n When an allocator is passed to any function, be sure that the allocator object is not destroyed until the last allocated object using it is freed."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtAllocator {
	#[doc = "< Must be initialized to ORT_API_VERSION"]
	pub version: u32,
	#[doc = "< Returns a pointer to an allocated block of `size` bytes"]
	pub Alloc: Option<unsafe extern "system" fn(this_: *mut OrtAllocator, size: usize) -> *mut core::ffi::c_void>,
	#[doc = "< Free a block of memory previously allocated with OrtAllocator::Alloc"]
	pub Free: Option<unsafe extern "system" fn(this_: *mut OrtAllocator, p: *mut core::ffi::c_void)>,
	#[doc = "< Return a pointer to an ::OrtMemoryInfo that describes this allocator"]
	pub Info: Option<unsafe extern "system" fn(this_: *const OrtAllocator) -> *const OrtMemoryInfo>,
	pub Reserve: Option<unsafe extern "system" fn(this_: *const OrtAllocator, size: usize) -> *mut core::ffi::c_void>
}
pub type OrtLoggingFunction = unsafe extern "system" fn(
	param: *mut core::ffi::c_void,
	severity: OrtLoggingLevel,
	category: *const core::ffi::c_char,
	logid: *const core::ffi::c_char,
	code_location: *const core::ffi::c_char,
	message: *const core::ffi::c_char
);
#[repr(i32)]
#[doc = " \\brief Graph optimization level\n\n Refer to https://www.onnxruntime.ai/docs/performance/graph-optimizations.html#graph-optimization-levels\n for an in-depth understanding of the Graph Optimization Levels."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum GraphOptimizationLevel {
	ORT_DISABLE_ALL = 0,
	ORT_ENABLE_BASIC = 1,
	ORT_ENABLE_EXTENDED = 2,
	ORT_ENABLE_ALL = 99
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ExecutionMode {
	ORT_SEQUENTIAL = 0,
	ORT_PARALLEL = 1
}
#[repr(i32)]
#[doc = " \\brief Language projection identifiers\n /see OrtApi::SetLanguageProjection"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtLanguageProjection {
	ORT_PROJECTION_C = 0,
	ORT_PROJECTION_CPLUSPLUS = 1,
	ORT_PROJECTION_CSHARP = 2,
	ORT_PROJECTION_PYTHON = 3,
	ORT_PROJECTION_JAVA = 4,
	ORT_PROJECTION_WINML = 5,
	ORT_PROJECTION_NODEJS = 6
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtKernelInfo {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtKernelContext {
	_unused: [u8; 0]
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtAllocatorType {
	OrtInvalidAllocator = -1,
	OrtDeviceAllocator = 0,
	OrtArenaAllocator = 1
}
impl OrtMemType {
	pub const OrtMemTypeCPU: OrtMemType = OrtMemType::OrtMemTypeCPUOutput;
}
#[repr(i32)]
#[doc = " \\brief Memory types for allocated memory, execution provider specific types should be extended in each provider."]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtMemType {
	#[doc = "< Any CPU memory used by non-CPU execution provider"]
	OrtMemTypeCPUInput = -2,
	#[doc = "< CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED"]
	OrtMemTypeCPUOutput = -1,
	#[doc = "< The default allocator for execution provider"]
	OrtMemTypeDefault = 0
}
#[repr(i32)]
#[doc = " \\brief This mimics OrtDevice type constants so they can be returned in the API"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtMemoryInfoDeviceType {
	OrtMemoryInfoDeviceType_CPU = 0,
	OrtMemoryInfoDeviceType_GPU = 1,
	OrtMemoryInfoDeviceType_FPGA = 2
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtHardwareDeviceType {
	OrtHardwareDeviceType_CPU = 0,
	OrtHardwareDeviceType_GPU = 1,
	OrtHardwareDeviceType_NPU = 2
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtExecutionProviderDevicePolicy {
	OrtExecutionProviderDevicePolicy_DEFAULT = 0,
	OrtExecutionProviderDevicePolicy_PREFER_CPU = 1,
	OrtExecutionProviderDevicePolicy_PREFER_NPU = 2,
	OrtExecutionProviderDevicePolicy_PREFER_GPU = 3,
	OrtExecutionProviderDevicePolicy_MAX_PERFORMANCE = 4,
	OrtExecutionProviderDevicePolicy_MAX_EFFICIENCY = 5,
	OrtExecutionProviderDevicePolicy_MIN_OVERALL_POWER = 6
}
pub type EpSelectionDelegate = Option<
	unsafe extern "system" fn(
		ep_devices: *const *const OrtEpDevice,
		num_devices: usize,
		model_metadata: *const OrtKeyValuePairs,
		runtime_metadata: *const OrtKeyValuePairs,
		selected: *mut *const OrtEpDevice,
		max_selected: usize,
		num_selected: *mut usize,
		state: *mut c_void
	) -> OrtStatusPtr
>;
#[repr(i32)]
#[doc = " \\brief Algorithm to use for cuDNN Convolution Op"]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtCudnnConvAlgoSearch {
	OrtCudnnConvAlgoSearchExhaustive = 0,
	OrtCudnnConvAlgoSearchHeuristic = 1,
	OrtCudnnConvAlgoSearchDefault = 2
}
#[doc = " \\brief CUDA Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_CUDA"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCUDAProviderOptions {
	#[doc = " \\brief CUDA device Id\n   Defaults to 0."]
	pub device_id: core::ffi::c_int,
	#[doc = " \\brief CUDA Convolution algorithm search configuration.\n   See enum OrtCudnnConvAlgoSearch for more details.\n   Defaults to OrtCudnnConvAlgoSearchExhaustive."]
	pub cudnn_conv_algo_search: OrtCudnnConvAlgoSearch,
	#[doc = " \\brief CUDA memory limit (To use all possible memory pass in maximum usize)\n   Defaults to SIZE_MAX.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub gpu_mem_limit: usize,
	#[doc = " \\brief Strategy used to grow the memory arena\n   0 = kNextPowerOfTwo<br>\n   1 = kSameAsRequested<br>\n   Defaults to 0.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub arena_extend_strategy: core::ffi::c_int,
	#[doc = " \\brief Flag indicating if copying needs to take place on the same stream as the compute stream in the CUDA EP\n   0 = Use separate streams for copying and compute.\n   1 = Use the same stream for copying and compute.\n   Defaults to 1.\n   WARNING: Setting this to 0 may result in data races for some models.\n   Please see issue #4829 for more details."]
	pub do_copy_in_default_stream: core::ffi::c_int,
	#[doc = " \\brief Flag indicating if there is a user provided compute stream\n   Defaults to 0."]
	pub has_user_compute_stream: core::ffi::c_int,
	#[doc = " \\brief User provided compute stream.\n   If provided, please set `has_user_compute_stream` to 1."]
	pub user_compute_stream: *mut core::ffi::c_void,
	#[doc = " \\brief CUDA memory arena configuration parameters"]
	pub default_memory_arena_cfg: *mut OrtArenaCfg,
	#[doc = " \\brief Enable TunableOp for using.\n   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_ENABLE."]
	pub tunable_op_enable: core::ffi::c_int,
	#[doc = " \\brief Enable TunableOp for tuning.\n   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_TUNING_ENABLE."]
	pub tunable_op_tuning_enable: core::ffi::c_int,
	#[doc = " \\brief Max tuning duration time limit for each instance of TunableOp.\n   Defaults to 0 to disable the limit."]
	pub tunable_op_max_tuning_duration_ms: core::ffi::c_int
}
#[doc = " \\brief ROCM Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_ROCM"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtROCMProviderOptions {
	#[doc = " \\brief ROCM device Id\n   Defaults to 0."]
	pub device_id: core::ffi::c_int,
	#[doc = " \\brief ROCM MIOpen Convolution algorithm exaustive search option.\n   Defaults to 0 (false)."]
	pub miopen_conv_exhaustive_search: core::ffi::c_int,
	#[doc = " \\brief ROCM memory limit (To use all possible memory pass in maximum usize)\n   Defaults to SIZE_MAX.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub gpu_mem_limit: usize,
	#[doc = " \\brief Strategy used to grow the memory arena\n   0 = kNextPowerOfTwo<br>\n   1 = kSameAsRequested<br>\n   Defaults to 0.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub arena_extend_strategy: core::ffi::c_int,
	#[doc = " \\brief Flag indicating if copying needs to take place on the same stream as the compute stream in the ROCM EP\n   0 = Use separate streams for copying and compute.\n   1 = Use the same stream for copying and compute.\n   Defaults to 1.\n   WARNING: Setting this to 0 may result in data races for some models.\n   Please see issue #4829 for more details."]
	pub do_copy_in_default_stream: core::ffi::c_int,
	#[doc = " \\brief Flag indicating if there is a user provided compute stream\n   Defaults to 0."]
	pub has_user_compute_stream: core::ffi::c_int,
	#[doc = " \\brief User provided compute stream.\n   If provided, please set `has_user_compute_stream` to 1."]
	pub user_compute_stream: *mut core::ffi::c_void,
	#[doc = " \\brief ROCM memory arena configuration parameters"]
	pub default_memory_arena_cfg: *mut OrtArenaCfg,
	pub enable_hip_graph: core::ffi::c_int,
	#[doc = " \\brief Enable TunableOp for using.\n   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_ENABLE."]
	pub tunable_op_enable: core::ffi::c_int,
	#[doc = " \\brief Enable TunableOp for tuning.\n   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_TUNING_ENABLE."]
	pub tunable_op_tuning_enable: core::ffi::c_int,
	#[doc = " \\brief Max tuning duration time limit for each instance of TunableOp.\n   Defaults to 0 to disable the limit."]
	pub tunable_op_max_tuning_duration_ms: core::ffi::c_int
}
#[doc = " \\brief TensorRT Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_TensorRT"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTensorRTProviderOptions {
	#[doc = "< CUDA device id (0 = default device)"]
	pub device_id: core::ffi::c_int,
	pub has_user_compute_stream: core::ffi::c_int,
	pub user_compute_stream: *mut core::ffi::c_void,
	pub trt_max_partition_iterations: core::ffi::c_int,
	pub trt_min_subgraph_size: core::ffi::c_int,
	pub trt_max_workspace_size: usize,
	pub trt_fp16_enable: core::ffi::c_int,
	pub trt_int8_enable: core::ffi::c_int,
	pub trt_int8_calibration_table_name: *const core::ffi::c_char,
	pub trt_int8_use_native_calibration_table: core::ffi::c_int,
	pub trt_dla_enable: core::ffi::c_int,
	pub trt_dla_core: core::ffi::c_int,
	pub trt_dump_subgraphs: core::ffi::c_int,
	pub trt_engine_cache_enable: core::ffi::c_int,
	pub trt_engine_cache_path: *const core::ffi::c_char,
	pub trt_engine_decryption_enable: core::ffi::c_int,
	pub trt_engine_decryption_lib_path: *const core::ffi::c_char,
	pub trt_force_sequential_engine_build: core::ffi::c_int
}
#[doc = " \\brief MIGraphX Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtMIGraphXProviderOptions {
	pub device_id: core::ffi::c_int,
	pub migraphx_fp16_enable: core::ffi::c_int,
	pub migraphx_int8_enable: core::ffi::c_int,
	pub migraphx_use_native_calibration_table: core::ffi::c_int,
	pub migraphx_int8_calibration_table_name: *const core::ffi::c_char,
	pub migraphx_save_compiled_model: core::ffi::c_int,
	pub migraphx_save_model_path: *const core::ffi::c_char,
	pub migraphx_load_compiled_model: core::ffi::c_int,
	pub migraphx_load_model_path: *const core::ffi::c_char,
	pub migraphx_exhaustive_tune: bool
}
#[doc = " \\brief OpenVINO Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtOpenVINOProviderOptions {
	#[doc = " \\brief Device type string\n\n Valid settings are one of: \"CPU_FP32\", \"CPU_FP16\", \"GPU_FP32\", \"GPU_FP16\""]
	pub device_type: *const core::ffi::c_char,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_npu_fast_compile: core::ffi::c_uchar,
	pub device_id: *const core::ffi::c_char,
	#[doc = "< 0 = Use default number of threads"]
	pub num_of_threads: usize,
	pub cache_dir: *const core::ffi::c_char,
	pub context: *mut core::ffi::c_void,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_opencl_throttling: core::ffi::c_uchar,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_dynamic_shapes: core::ffi::c_uchar
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTrainingSession {
	_unused: [u8; 0]
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCheckpointState {
	_unused: [u8; 0]
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtPropertyType {
	OrtIntProperty = 0,
	OrtFloatProperty = 1,
	OrtStringProperty = 2
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTrainingApi {
	pub LoadCheckpoint: unsafe extern "system" fn(checkpoint_path: *const ortchar, checkpoint_state: *mut *mut OrtCheckpointState) -> OrtStatusPtr,
	pub SaveCheckpoint:
		unsafe extern "system" fn(checkpoint_state: *mut OrtCheckpointState, checkpoint_path: *const ortchar, include_optimizer_state: bool) -> OrtStatusPtr,
	pub CreateTrainingSession: unsafe extern "system" fn(
		env: *const OrtEnv,
		options: *const OrtSessionOptions,
		checkpoint_state: *mut OrtCheckpointState,
		train_model_path: *const ortchar,
		eval_model_path: *const ortchar,
		optimizer_model_path: *const ortchar,
		out: *mut *mut OrtTrainingSession
	) -> OrtStatusPtr,
	pub CreateTrainingSessionFromBuffer: unsafe extern "system" fn(
		env: *const OrtEnv,
		options: *const OrtSessionOptions,
		checkpoint_state: *mut OrtCheckpointState,
		train_model_data: *const (),
		train_data_length: usize,
		eval_model_data: *const (),
		eval_data_length: usize,
		optimizer_model_data: *const (),
		optimizer_data_length: usize,
		out: *mut *mut OrtTrainingSession
	) -> OrtStatusPtr,
	pub TrainingSessionGetTrainingModelOutputCount: unsafe extern "system" fn(sess: *const OrtTrainingSession, out: *mut usize) -> OrtStatusPtr,
	pub TrainingSessionGetEvalModelOutputCount: unsafe extern "system" fn(sess: *const OrtTrainingSession, out: *mut usize) -> OrtStatusPtr,
	pub TrainingSessionGetTrainingModelOutputName:
		unsafe extern "system" fn(sess: *const OrtTrainingSession, index: usize, allocator: *mut OrtAllocator, output: *mut *const c_char) -> OrtStatusPtr,
	pub TrainingSessionGetEvalModelOutputName:
		unsafe extern "system" fn(sess: *const OrtTrainingSession, index: usize, allocator: *mut OrtAllocator, output: *mut *const c_char) -> OrtStatusPtr,
	pub LazyResetGrad: unsafe extern "system" fn(session: *mut OrtTrainingSession) -> OrtStatusPtr,
	pub TrainStep: unsafe extern "system" fn(
		session: *mut OrtTrainingSession,
		run_options: *const OrtRunOptions,
		inputs_len: usize,
		inputs: *const *const OrtValue,
		outputs_len: usize,
		outputs: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub EvalStep: unsafe extern "system" fn(
		session: *mut OrtTrainingSession,
		run_options: *const OrtRunOptions,
		inputs_len: usize,
		inputs: *const *const OrtValue,
		outputs_len: usize,
		outputs: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub SetLearningRate: unsafe extern "system" fn(session: *mut OrtTrainingSession, learning_rate: f32) -> OrtStatusPtr,
	pub GetLearningRate: unsafe extern "system" fn(session: *mut OrtTrainingSession, learning_rate: *mut f32) -> OrtStatusPtr,
	pub OptimizerStep: unsafe extern "system" fn(session: *mut OrtTrainingSession, run_options: *const OrtRunOptions) -> OrtStatusPtr,
	pub RegisterLinearLRScheduler:
		unsafe extern "system" fn(session: *mut OrtTrainingSession, warmup_step_count: i64, total_step_count: i64, initial_lr: f32) -> OrtStatusPtr,
	pub SchedulerStep: unsafe extern "system" fn(session: *mut OrtTrainingSession) -> OrtStatusPtr,
	pub GetParametersSize: unsafe extern "system" fn(session: *mut OrtTrainingSession, out: *mut usize, trainable_only: bool) -> OrtStatusPtr,
	pub CopyParametersToBuffer:
		unsafe extern "system" fn(session: *mut OrtTrainingSession, parameters_buffer: *mut OrtValue, trainable_only: bool) -> OrtStatusPtr,
	pub CopyBufferToParameters:
		unsafe extern "system" fn(session: *mut OrtTrainingSession, parameters_buffer: *mut OrtValue, trainable_only: bool) -> OrtStatusPtr,
	pub ReleaseTrainingSession: unsafe extern "system" fn(input: *mut OrtTrainingSession),
	pub ReleaseCheckpointState: unsafe extern "system" fn(input: *mut OrtCheckpointState),
	pub ExportModelForInferencing: unsafe extern "system" fn(
		session: *mut OrtTrainingSession,
		inference_model_path: *const ortchar,
		graph_outputs_len: usize,
		graph_output_names: *const *const c_char
	) -> OrtStatusPtr,
	pub SetSeed: unsafe extern "system" fn(seed: i64) -> OrtStatusPtr,
	pub TrainingSessionGetTrainingModelInputCount: unsafe extern "system" fn(session: *const OrtTrainingSession, out: *mut usize) -> OrtStatusPtr,
	pub TrainingSessionGetEvalModelInputCount: unsafe extern "system" fn(session: *const OrtTrainingSession, out: *mut usize) -> OrtStatusPtr,
	pub TrainingSessionGetTrainingModelInputName:
		unsafe extern "system" fn(session: *const OrtTrainingSession, index: usize, allocator: *mut OrtAllocator, output: *mut *const c_char) -> OrtStatusPtr,
	pub TrainingSessionGetEvalModelInputName:
		unsafe extern "system" fn(session: *const OrtTrainingSession, index: usize, allocator: *mut OrtAllocator, output: *mut *const c_char) -> OrtStatusPtr,
	pub AddProperty: unsafe extern "system" fn(
		checkpoint_state: *mut OrtCheckpointState,
		property_name: *const c_char,
		property_type: OrtPropertyType,
		property_value: *const ()
	) -> OrtStatusPtr,
	pub GetProperty: unsafe extern "system" fn(
		checkpoint_state: *mut OrtCheckpointState,
		property_name: *const c_char,
		allocator: *mut OrtAllocator,
		property_type: *mut OrtPropertyType,
		property_value: *mut *const ()
	) -> OrtStatusPtr,
	pub LoadCheckpointFromBuffer:
		unsafe extern "system" fn(checkpoint_buffer: *const (), num_bytes: usize, checkpoint_state: *mut *mut OrtCheckpointState) -> OrtStatusPtr,
	pub GetParameterTypeAndShape: unsafe extern "system" fn(
		checkpoint_state: *const OrtCheckpointState,
		parameter_name: *const c_char,
		parameter_type_and_shape: *mut *mut OrtTensorTypeAndShapeInfo
	) -> OrtStatusPtr,
	pub UpdateParameter:
		unsafe extern "system" fn(checkpoint_state: *mut OrtCheckpointState, parameter_name: *const c_char, parameter: *mut OrtValue) -> OrtStatusPtr,
	pub GetParameter: unsafe extern "system" fn(
		checkpoint_state: *const OrtCheckpointState,
		parameter_name: *const c_char,
		allocator: *mut OrtAllocator,
		parameter: *mut *mut OrtValue
	) -> OrtStatusPtr
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtModelEditorApi {
	pub CreateTensorTypeInfo: unsafe extern "system" fn(tensor_info: *const OrtTensorTypeAndShapeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub CreateSparseTensorTypeInfo: unsafe extern "system" fn(tensor_info: *const OrtTensorTypeAndShapeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub CreateMapTypeInfo: unsafe extern "system" fn(
		map_key_type: ONNXTensorElementDataType,
		map_value_type: *const OrtTypeInfo,
		type_info: *mut *mut OrtTypeInfo
	) -> OrtStatusPtr,
	pub CreateSequenceTypeInfo: unsafe extern "system" fn(sequence_type: *const OrtTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub CreateOptionalTypeInfo: unsafe extern "system" fn(contained_type: *const OrtTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub CreateValueInfo: unsafe extern "system" fn(name: *const c_char, type_info: *const OrtTypeInfo, value_info: *mut *mut OrtValueInfo) -> OrtStatusPtr,
	pub CreateNode: unsafe extern "system" fn(
		operator_name: *const c_char,
		domain_name: *const c_char,
		node_name: *const c_char,
		input_names: *const *const c_char,
		input_names_len: usize,
		output_names: *const *const c_char,
		output_names_len: usize,
		attributes: *mut *mut OrtOpAttr,
		attribs_len: usize,
		node: *mut *mut OrtNode
	) -> OrtStatusPtr,
	pub CreateGraph: unsafe extern "system" fn(graph: *mut *mut OrtGraph) -> OrtStatusPtr,
	pub SetGraphInputs: unsafe extern "system" fn(graph: *mut OrtGraph, inputs: *mut *mut OrtValueInfo, inputs_len: usize) -> OrtStatusPtr,
	pub SetGraphOutputs: unsafe extern "system" fn(graph: *mut OrtGraph, outputs: *mut *mut OrtValueInfo, outputs_len: usize) -> OrtStatusPtr,
	pub AddInitializerToGraph:
		unsafe extern "system" fn(graph: *mut OrtGraph, name: *const c_char, tensor: *mut OrtValue, data_is_external: bool) -> OrtStatusPtr,
	pub AddNodeToGraph: unsafe extern "system" fn(graph: *mut OrtGraph, node: *mut OrtNode) -> OrtStatusPtr,
	pub CreateModel: unsafe extern "system" fn(
		domain_names: *const *const c_char,
		opset_versions: *const i32,
		opset_entries_len: usize,
		model: *mut *mut OrtModel
	) -> OrtStatusPtr,
	pub AddGraphToModel: unsafe extern "system" fn(model: *mut OrtModel, graph: *mut OrtGraph) -> OrtStatusPtr,
	pub CreateSessionFromModel:
		unsafe extern "system" fn(env: *const OrtEnv, model: *const OrtModel, options: *const OrtSessionOptions, out: *mut *mut OrtSession) -> OrtStatusPtr,
	pub CreateModelEditorSession:
		unsafe extern "system" fn(env: *const OrtEnv, model_path: *const ortchar, options: *const OrtSessionOptions, out: *mut *mut OrtSession) -> OrtStatusPtr,
	pub CreateModelEditorSessionFromArray: unsafe extern "system" fn(
		env: *const OrtEnv,
		model_data: *const c_void,
		model_data_length: usize,
		options: *const OrtSessionOptions,
		out: *mut *mut OrtSession
	) -> OrtStatusPtr,
	pub SessionGetOpsetForDomain: unsafe extern "system" fn(session: *const OrtSession, domain: *const c_char, opset: *mut i32) -> OrtStatusPtr,
	pub ApplyModelToModelEditorSession: unsafe extern "system" fn(session: *mut OrtSession, model: *mut OrtModel) -> OrtStatusPtr,
	pub FinalizeModelEditorSession: unsafe extern "system" fn(
		session: *mut OrtSession,
		options: *const OrtSessionOptions,
		prepacked_weights_container: *const OrtPrepackedWeightsContainer
	) -> OrtStatusPtr
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCompileApi {
	pub ReleaseModelCompilationOptions: unsafe extern "system" fn(input: *mut OrtModelCompilationOptions),
	pub CreateModelCompilationOptionsFromSessionOptions:
		unsafe extern "system" fn(env: *const OrtEnv, session_options: *const OrtSessionOptions, out: *mut *mut OrtModelCompilationOptions) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetInputModelPath:
		unsafe extern "system" fn(model_compile_options: *mut OrtModelCompilationOptions, input_model_path: *const ortchar) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetInputModelFromBuffer: unsafe extern "system" fn(
		model_compile_options: *mut OrtModelCompilationOptions,
		input_model_data: *const c_void,
		input_model_data_size: usize
	) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetOutputModelPath:
		unsafe extern "system" fn(model_compile_options: *mut OrtModelCompilationOptions, output_model_path: *const ortchar) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetOutputModelExternalInitializersFile: unsafe extern "system" fn(
		model_compile_options: *mut OrtModelCompilationOptions,
		external_initializers_file_path: *const ortchar,
		external_initializers_size_threshold: usize
	) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetOutputModelBuffer: unsafe extern "system" fn(
		model_compile_options: *mut OrtModelCompilationOptions,
		allocator: *mut OrtAllocator,
		output_model_buffer_ptr: *mut *mut c_void,
		output_model_buffer_size_ptr: *mut usize
	) -> OrtStatusPtr,
	pub ModelCompilationOptions_SetEpContextEmbedMode:
		unsafe extern "system" fn(model_compile_options: *mut OrtModelCompilationOptions, embed_ep_context_in_model: bool) -> OrtStatusPtr,
	pub CompileModel: unsafe extern "system" fn(env: *const OrtEnv, model_options: *const OrtModelCompilationOptions) -> OrtStatusPtr
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtEpApi {
	_unused: [u8; 0]
}
#[doc = " \\brief The helper interface to get the right version of OrtApi\n\n Get a pointer to this structure through ::OrtGetApiBase"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtApiBase {
	#[doc = " \\brief Get a pointer to the requested version of the ::OrtApi\n\n \\param[in] version Must be ::ORT_API_VERSION\n \\return The ::OrtApi for the version requested, nullptr will be returned if this version is unsupported, for example when using a runtime\n   older than the version created with this header file.\n\n One can call GetVersionString() to get the version of the Onnxruntime library for logging\n and error reporting purposes."]
	pub GetApi: unsafe extern "system" fn(version: u32) -> *const OrtApi,
	#[doc = " \\brief Returns a null terminated string of the version of the Onnxruntime library (eg: \"1.8.1\")\n\n  \\return UTF-8 encoded version string. Do not deallocate the returned buffer."]
	pub GetVersionString: unsafe extern "system" fn() -> *const core::ffi::c_char
}
extern "system" {
	#[doc = " \\brief The Onnxruntime library's entry point to access the C API\n\n Call this to get the a pointer to an ::OrtApiBase"]
	pub fn OrtGetApiBase() -> *const OrtApiBase;
}
#[doc = " \\brief Thread work loop function\n\n Onnxruntime will provide the working loop on custom thread creation\n Argument is an onnxruntime built-in type which will be provided when thread pool calls OrtCustomCreateThreadFn"]
pub type OrtThreadWorkerFn = unsafe extern "system" fn(ort_worker_fn_param: *mut core::ffi::c_void);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCustomHandleType {
	pub __place_holder: core::ffi::c_char
}
pub type OrtCustomThreadHandle = *const OrtCustomHandleType;
#[doc = " \\brief Ort custom thread creation function\n\n The function should return a thread handle to be used in onnxruntime thread pools\n Onnxruntime will throw exception on return value of nullptr or 0, indicating that the function failed to create a thread"]
pub type OrtCustomCreateThreadFn = Option<
	unsafe extern "system" fn(
		ort_custom_thread_creation_options: *mut core::ffi::c_void,
		ort_thread_worker_fn: OrtThreadWorkerFn,
		ort_worker_fn_param: *mut core::ffi::c_void
	) -> OrtCustomThreadHandle
>;
#[doc = " \\brief Custom thread join function\n\n Onnxruntime thread pool destructor will call the function to join a custom thread.\n Argument ort_custom_thread_handle is the value returned by OrtCustomCreateThreadFn"]
pub type OrtCustomJoinThreadFn = Option<unsafe extern "system" fn(ort_custom_thread_handle: OrtCustomThreadHandle)>;
#[doc = " \\brief Callback function for RunAsync\n\n \\param[in] user_data User specific data that passed back to the callback\n \\param[out] outputs On succeed, outputs host inference results, on error, the value will be nullptr\n \\param[out] num_outputs Number of outputs, on error, the value will be zero\n \\param[out] status On error, status will provide details"]
pub type RunAsyncCallbackFn =
	Option<unsafe extern "system" fn(user_data: *mut core::ffi::c_void, outputs: *mut *mut OrtValue, num_outputs: usize, status: OrtStatusPtr)>;
#[doc = " \\brief The C API\n\n All C API functions are defined inside this structure as pointers to functions.\n Call OrtApiBase::GetApi to get a pointer to it\n\n \\nosubgrouping"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtApi {
	#[doc = " \\brief Create an OrtStatus from a null terminated string\n\n \\param[in] code\n \\param[in] msg A null-terminated string. Its contents will be copied.\n \\return A new OrtStatus object, must be destroyed with OrtApi::ReleaseStatus"]
	pub CreateStatus: unsafe extern "system" fn(code: OrtErrorCode, msg: *const core::ffi::c_char) -> OrtStatusPtr,
	#[doc = " \\brief Get OrtErrorCode from OrtStatus\n\n \\param[in] status\n \\return OrtErrorCode that \\p status was created with"]
	pub GetErrorCode: unsafe extern "system" fn(status: *const OrtStatus) -> OrtErrorCode,
	#[doc = " \\brief Get error string from OrtStatus\n\n \\param[in] status\n \\return The error message inside the `status`. Do not free the returned value."]
	pub GetErrorMessage: unsafe extern "system" fn(status: *const OrtStatus) -> *const core::ffi::c_char,
	pub CreateEnv: unsafe extern "system" fn(log_severity_level: OrtLoggingLevel, logid: *const core::ffi::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr,
	pub CreateEnvWithCustomLogger: unsafe extern "system" fn(
		logging_function: OrtLoggingFunction,
		logger_param: *mut core::ffi::c_void,
		log_severity_level: OrtLoggingLevel,
		logid: *const core::ffi::c_char,
		out: *mut *mut OrtEnv
	) -> OrtStatusPtr,
	pub EnableTelemetryEvents: unsafe extern "system" fn(env: *const OrtEnv) -> OrtStatusPtr,
	pub DisableTelemetryEvents: unsafe extern "system" fn(env: *const OrtEnv) -> OrtStatusPtr,
	pub CreateSession:
		unsafe extern "system" fn(env: *const OrtEnv, model_path: *const ortchar, options: *const OrtSessionOptions, out: *mut *mut OrtSession) -> OrtStatusPtr,
	pub CreateSessionFromArray: unsafe extern "system" fn(
		env: *const OrtEnv,
		model_data: *const core::ffi::c_void,
		model_data_length: usize,
		options: *const OrtSessionOptions,
		out: *mut *mut OrtSession
	) -> OrtStatusPtr,
	pub Run: unsafe extern "system" fn(
		session: *mut OrtSession,
		run_options: *const OrtRunOptions,
		input_names: *const *const core::ffi::c_char,
		inputs: *const *const OrtValue,
		input_len: usize,
		output_names: *const *const core::ffi::c_char,
		output_names_len: usize,
		outputs: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub CreateSessionOptions: unsafe extern "system" fn(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr,
	pub SetOptimizedModelFilePath: unsafe extern "system" fn(options: *mut OrtSessionOptions, optimized_model_filepath: *const ortchar) -> OrtStatusPtr,
	pub CloneSessionOptions: unsafe extern "system" fn(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr,
	pub SetSessionExecutionMode: unsafe extern "system" fn(options: *mut OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr,
	pub EnableProfiling: unsafe extern "system" fn(options: *mut OrtSessionOptions, profile_file_prefix: *const ortchar) -> OrtStatusPtr,
	pub DisableProfiling: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub EnableMemPattern: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub DisableMemPattern: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub EnableCpuMemArena: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub DisableCpuMemArena: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub SetSessionLogId: unsafe extern "system" fn(options: *mut OrtSessionOptions, logid: *const core::ffi::c_char) -> OrtStatusPtr,
	pub SetSessionLogVerbosityLevel: unsafe extern "system" fn(options: *mut OrtSessionOptions, session_log_verbosity_level: core::ffi::c_int) -> OrtStatusPtr,
	pub SetSessionLogSeverityLevel: unsafe extern "system" fn(options: *mut OrtSessionOptions, session_log_severity_level: core::ffi::c_int) -> OrtStatusPtr,
	pub SetSessionGraphOptimizationLevel:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr,
	pub SetIntraOpNumThreads: unsafe extern "system" fn(options: *mut OrtSessionOptions, intra_op_num_threads: core::ffi::c_int) -> OrtStatusPtr,
	pub SetInterOpNumThreads: unsafe extern "system" fn(options: *mut OrtSessionOptions, inter_op_num_threads: core::ffi::c_int) -> OrtStatusPtr,
	pub CreateCustomOpDomain: unsafe extern "system" fn(domain: *const core::ffi::c_char, out: *mut *mut OrtCustomOpDomain) -> OrtStatusPtr,
	pub CustomOpDomain_Add: unsafe extern "system" fn(custom_op_domain: *mut OrtCustomOpDomain, op: *const OrtCustomOp) -> OrtStatusPtr,
	pub AddCustomOpDomain: unsafe extern "system" fn(options: *mut OrtSessionOptions, custom_op_domain: *mut OrtCustomOpDomain) -> OrtStatusPtr,
	pub RegisterCustomOpsLibrary: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		library_path: *const core::ffi::c_char,
		library_handle: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub SessionGetInputCount: unsafe extern "system" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,
	pub SessionGetOutputCount: unsafe extern "system" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,
	pub SessionGetOverridableInitializerCount: unsafe extern "system" fn(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr,
	pub SessionGetInputTypeInfo: unsafe extern "system" fn(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub SessionGetOutputTypeInfo: unsafe extern "system" fn(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub SessionGetOverridableInitializerTypeInfo:
		unsafe extern "system" fn(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub SessionGetInputName:
		unsafe extern "system" fn(session: *const OrtSession, index: usize, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub SessionGetOutputName:
		unsafe extern "system" fn(session: *const OrtSession, index: usize, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub SessionGetOverridableInitializerName:
		unsafe extern "system" fn(session: *const OrtSession, index: usize, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub CreateRunOptions: unsafe extern "system" fn(out: *mut *mut OrtRunOptions) -> OrtStatusPtr,
	pub RunOptionsSetRunLogVerbosityLevel: unsafe extern "system" fn(options: *mut OrtRunOptions, log_verbosity_level: core::ffi::c_int) -> OrtStatusPtr,
	pub RunOptionsSetRunLogSeverityLevel: unsafe extern "system" fn(options: *mut OrtRunOptions, log_severity_level: core::ffi::c_int) -> OrtStatusPtr,
	pub RunOptionsSetRunTag: unsafe extern "system" fn(options: *mut OrtRunOptions, run_tag: *const core::ffi::c_char) -> OrtStatusPtr,
	pub RunOptionsGetRunLogVerbosityLevel: unsafe extern "system" fn(options: *const OrtRunOptions, log_verbosity_level: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub RunOptionsGetRunLogSeverityLevel: unsafe extern "system" fn(options: *const OrtRunOptions, log_severity_level: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub RunOptionsGetRunTag: unsafe extern "system" fn(options: *const OrtRunOptions, run_tag: *mut *const core::ffi::c_char) -> OrtStatusPtr,
	pub RunOptionsSetTerminate: unsafe extern "system" fn(options: *mut OrtRunOptions) -> OrtStatusPtr,
	pub RunOptionsUnsetTerminate: unsafe extern "system" fn(options: *mut OrtRunOptions) -> OrtStatusPtr,
	pub CreateTensorAsOrtValue: unsafe extern "system" fn(
		allocator: *mut OrtAllocator,
		shape: *const i64,
		shape_len: usize,
		type_: ONNXTensorElementDataType,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub CreateTensorWithDataAsOrtValue: unsafe extern "system" fn(
		info: *const OrtMemoryInfo,
		p_data: *mut core::ffi::c_void,
		p_data_len: usize,
		shape: *const i64,
		shape_len: usize,
		type_: ONNXTensorElementDataType,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub IsTensor: unsafe extern "system" fn(value: *const OrtValue, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub GetTensorMutableData: unsafe extern "system" fn(value: *mut OrtValue, out: *mut *mut core::ffi::c_void) -> OrtStatusPtr,
	pub FillStringTensor: unsafe extern "system" fn(value: *mut OrtValue, s: *const *const core::ffi::c_char, s_len: usize) -> OrtStatusPtr,
	pub GetStringTensorDataLength: unsafe extern "system" fn(value: *const OrtValue, len: *mut usize) -> OrtStatusPtr,
	pub GetStringTensorContent:
		unsafe extern "system" fn(value: *const OrtValue, s: *mut core::ffi::c_void, s_len: usize, offsets: *mut usize, offsets_len: usize) -> OrtStatusPtr,
	pub CastTypeInfoToTensorInfo: unsafe extern "system" fn(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub GetOnnxTypeFromTypeInfo: unsafe extern "system" fn(type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr,
	pub CreateTensorTypeAndShapeInfo: unsafe extern "system" fn(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub SetTensorElementType: unsafe extern "system" fn(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr,
	pub SetDimensions: unsafe extern "system" fn(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: usize) -> OrtStatusPtr,
	pub GetTensorElementType: unsafe extern "system" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr,
	pub GetDimensionsCount: unsafe extern "system" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr,
	pub GetDimensions: unsafe extern "system" fn(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: usize) -> OrtStatusPtr,
	pub GetSymbolicDimensions:
		unsafe extern "system" fn(info: *const OrtTensorTypeAndShapeInfo, dim_params: *mut *const core::ffi::c_char, dim_params_length: usize) -> OrtStatusPtr,
	pub GetTensorShapeElementCount: unsafe extern "system" fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr,
	pub GetTensorTypeAndShape: unsafe extern "system" fn(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub GetTypeInfo: unsafe extern "system" fn(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub GetValueType: unsafe extern "system" fn(value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr,
	pub CreateMemoryInfo: unsafe extern "system" fn(
		name: *const core::ffi::c_char,
		type_: OrtAllocatorType,
		id: core::ffi::c_int,
		mem_type: OrtMemType,
		out: *mut *mut OrtMemoryInfo
	) -> OrtStatusPtr,
	pub CreateCpuMemoryInfo: unsafe extern "system" fn(type_: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr,
	pub CompareMemoryInfo: unsafe extern "system" fn(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub MemoryInfoGetName: unsafe extern "system" fn(ptr: *const OrtMemoryInfo, out: *mut *const core::ffi::c_char) -> OrtStatusPtr,
	pub MemoryInfoGetId: unsafe extern "system" fn(ptr: *const OrtMemoryInfo, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub MemoryInfoGetMemType: unsafe extern "system" fn(ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr,
	pub MemoryInfoGetType: unsafe extern "system" fn(ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr,
	pub AllocatorAlloc: unsafe extern "system" fn(ort_allocator: *mut OrtAllocator, size: usize, out: *mut *mut core::ffi::c_void) -> OrtStatusPtr,
	pub AllocatorFree: unsafe extern "system" fn(ort_allocator: *mut OrtAllocator, p: *mut core::ffi::c_void) -> OrtStatusPtr,
	pub AllocatorGetInfo: unsafe extern "system" fn(ort_allocator: *const OrtAllocator, out: *mut *const OrtMemoryInfo) -> OrtStatusPtr,
	pub GetAllocatorWithDefaultOptions: unsafe extern "system" fn(out: *mut *mut OrtAllocator) -> OrtStatusPtr,
	pub AddFreeDimensionOverride:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, dim_denotation: *const core::ffi::c_char, dim_value: i64) -> OrtStatusPtr,
	pub GetValue:
		unsafe extern "system" fn(value: *const OrtValue, index: core::ffi::c_int, allocator: *mut OrtAllocator, out: *mut *mut OrtValue) -> OrtStatusPtr,
	pub GetValueCount: unsafe extern "system" fn(value: *const OrtValue, out: *mut usize) -> OrtStatusPtr,
	pub CreateValue: unsafe extern "system" fn(in_: *const *const OrtValue, num_values: usize, value_type: ONNXType, out: *mut *mut OrtValue) -> OrtStatusPtr,
	pub CreateOpaqueValue: unsafe extern "system" fn(
		domain_name: *const core::ffi::c_char,
		type_name: *const core::ffi::c_char,
		data_container: *const core::ffi::c_void,
		data_container_size: usize,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub GetOpaqueValue: unsafe extern "system" fn(
		domain_name: *const core::ffi::c_char,
		type_name: *const core::ffi::c_char,
		in_: *const OrtValue,
		data_container: *mut core::ffi::c_void,
		data_container_size: usize
	) -> OrtStatusPtr,
	pub KernelInfoGetAttribute_float: unsafe extern "system" fn(info: *const OrtKernelInfo, name: *const core::ffi::c_char, out: *mut f32) -> OrtStatusPtr,
	pub KernelInfoGetAttribute_int64: unsafe extern "system" fn(info: *const OrtKernelInfo, name: *const core::ffi::c_char, out: *mut i64) -> OrtStatusPtr,
	pub KernelInfoGetAttribute_string:
		unsafe extern "system" fn(info: *const OrtKernelInfo, name: *const core::ffi::c_char, out: *mut core::ffi::c_char, size: *mut usize) -> OrtStatusPtr,
	pub KernelContext_GetInputCount: unsafe extern "system" fn(context: *const OrtKernelContext, out: *mut usize) -> OrtStatusPtr,
	pub KernelContext_GetOutputCount: unsafe extern "system" fn(context: *const OrtKernelContext, out: *mut usize) -> OrtStatusPtr,
	pub KernelContext_GetInput: unsafe extern "system" fn(context: *const OrtKernelContext, index: usize, out: *mut *const OrtValue) -> OrtStatusPtr,
	pub KernelContext_GetOutput: unsafe extern "system" fn(
		context: *mut OrtKernelContext,
		index: usize,
		dim_values: *const i64,
		dim_count: usize,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub ReleaseEnv: unsafe extern "system" fn(input: *mut OrtEnv),
	pub ReleaseStatus: unsafe extern "system" fn(input: *mut OrtStatus),
	pub ReleaseMemoryInfo: unsafe extern "system" fn(input: *mut OrtMemoryInfo),
	pub ReleaseSession: unsafe extern "system" fn(input: *mut OrtSession),
	pub ReleaseValue: unsafe extern "system" fn(input: *mut OrtValue),
	pub ReleaseRunOptions: unsafe extern "system" fn(input: *mut OrtRunOptions),
	pub ReleaseTypeInfo: unsafe extern "system" fn(input: *mut OrtTypeInfo),
	pub ReleaseTensorTypeAndShapeInfo: unsafe extern "system" fn(input: *mut OrtTensorTypeAndShapeInfo),
	pub ReleaseSessionOptions: unsafe extern "system" fn(input: *mut OrtSessionOptions),
	pub ReleaseCustomOpDomain: unsafe extern "system" fn(input: *mut OrtCustomOpDomain),
	pub GetDenotationFromTypeInfo:
		unsafe extern "system" fn(type_info: *const OrtTypeInfo, denotation: *mut *const core::ffi::c_char, len: *mut usize) -> OrtStatusPtr,
	pub CastTypeInfoToMapTypeInfo: unsafe extern "system" fn(type_info: *const OrtTypeInfo, out: *mut *const OrtMapTypeInfo) -> OrtStatusPtr,
	pub CastTypeInfoToSequenceTypeInfo: unsafe extern "system" fn(type_info: *const OrtTypeInfo, out: *mut *const OrtSequenceTypeInfo) -> OrtStatusPtr,
	pub GetMapKeyType: unsafe extern "system" fn(map_type_info: *const OrtMapTypeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr,
	pub GetMapValueType: unsafe extern "system" fn(map_type_info: *const OrtMapTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub GetSequenceElementType: unsafe extern "system" fn(sequence_type_info: *const OrtSequenceTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub ReleaseMapTypeInfo: unsafe extern "system" fn(input: *mut OrtMapTypeInfo),
	pub ReleaseSequenceTypeInfo: unsafe extern "system" fn(input: *mut OrtSequenceTypeInfo),
	pub SessionEndProfiling:
		unsafe extern "system" fn(session: *mut OrtSession, allocator: *mut OrtAllocator, out: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub SessionGetModelMetadata: unsafe extern "system" fn(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr,
	pub ModelMetadataGetProducerName:
		unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub ModelMetadataGetGraphName:
		unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub ModelMetadataGetDomain:
		unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub ModelMetadataGetDescription:
		unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub ModelMetadataLookupCustomMetadataMap: unsafe extern "system" fn(
		model_metadata: *const OrtModelMetadata,
		allocator: *mut OrtAllocator,
		key: *const core::ffi::c_char,
		value: *mut *mut core::ffi::c_char
	) -> OrtStatusPtr,
	pub ModelMetadataGetVersion: unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, value: *mut i64) -> OrtStatusPtr,
	pub ReleaseModelMetadata: unsafe extern "system" fn(input: *mut OrtModelMetadata),
	pub CreateEnvWithGlobalThreadPools: unsafe extern "system" fn(
		log_severity_level: OrtLoggingLevel,
		logid: *const core::ffi::c_char,
		tp_options: *const OrtThreadingOptions,
		out: *mut *mut OrtEnv
	) -> OrtStatusPtr,
	pub DisablePerSessionThreads: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub CreateThreadingOptions: unsafe extern "system" fn(out: *mut *mut OrtThreadingOptions) -> OrtStatusPtr,
	pub ReleaseThreadingOptions: unsafe extern "system" fn(input: *mut OrtThreadingOptions),
	pub ModelMetadataGetCustomMetadataMapKeys: unsafe extern "system" fn(
		model_metadata: *const OrtModelMetadata,
		allocator: *mut OrtAllocator,
		keys: *mut *mut *mut core::ffi::c_char,
		num_keys: *mut i64
	) -> OrtStatusPtr,
	pub AddFreeDimensionOverrideByName:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, dim_name: *const core::ffi::c_char, dim_value: i64) -> OrtStatusPtr,
	pub GetAvailableProviders: unsafe extern "system" fn(out_ptr: *mut *mut *mut core::ffi::c_char, provider_length: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub ReleaseAvailableProviders: unsafe extern "system" fn(ptr: *mut *mut core::ffi::c_char, providers_length: core::ffi::c_int) -> OrtStatusPtr,
	pub GetStringTensorElementLength: unsafe extern "system" fn(value: *const OrtValue, index: usize, out: *mut usize) -> OrtStatusPtr,
	pub GetStringTensorElement: unsafe extern "system" fn(value: *const OrtValue, s_len: usize, index: usize, s: *mut core::ffi::c_void) -> OrtStatusPtr,
	pub FillStringTensorElement: unsafe extern "system" fn(value: *mut OrtValue, s: *const core::ffi::c_char, index: usize) -> OrtStatusPtr,
	pub AddSessionConfigEntry: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		config_key: *const core::ffi::c_char,
		config_value: *const core::ffi::c_char
	) -> OrtStatusPtr,
	pub CreateAllocator: unsafe extern "system" fn(session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr,
	pub ReleaseAllocator: unsafe extern "system" fn(input: *mut OrtAllocator),
	pub RunWithBinding:
		unsafe extern "system" fn(session: *mut OrtSession, run_options: *const OrtRunOptions, binding_ptr: *const OrtIoBinding) -> OrtStatusPtr,
	pub CreateIoBinding: unsafe extern "system" fn(session: *mut OrtSession, out: *mut *mut OrtIoBinding) -> OrtStatusPtr,
	pub ReleaseIoBinding: unsafe extern "system" fn(input: *mut OrtIoBinding),
	pub BindInput: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding, name: *const core::ffi::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr,
	pub BindOutput: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding, name: *const core::ffi::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr,
	pub BindOutputToDevice:
		unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding, name: *const core::ffi::c_char, mem_info_ptr: *const OrtMemoryInfo) -> OrtStatusPtr,
	pub GetBoundOutputNames: unsafe extern "system" fn(
		binding_ptr: *const OrtIoBinding,
		allocator: *mut OrtAllocator,
		buffer: *mut *mut core::ffi::c_char,
		lengths: *mut *mut usize,
		count: *mut usize
	) -> OrtStatusPtr,
	pub GetBoundOutputValues: unsafe extern "system" fn(
		binding_ptr: *const OrtIoBinding,
		allocator: *mut OrtAllocator,
		output: *mut *mut *mut OrtValue,
		output_count: *mut usize
	) -> OrtStatusPtr,
	#[doc = " \\brief Clears any previously set Inputs for an ::OrtIoBinding"]
	pub ClearBoundInputs: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding),
	#[doc = " \\brief Clears any previously set Outputs for an ::OrtIoBinding"]
	pub ClearBoundOutputs: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding),
	pub TensorAt: unsafe extern "system" fn(
		value: *mut OrtValue,
		location_values: *const i64,
		location_values_count: usize,
		out: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub CreateAndRegisterAllocator: unsafe extern "system" fn(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo, arena_cfg: *const OrtArenaCfg) -> OrtStatusPtr,
	pub SetLanguageProjection: unsafe extern "system" fn(ort_env: *const OrtEnv, projection: OrtLanguageProjection) -> OrtStatusPtr,
	pub SessionGetProfilingStartTimeNs: unsafe extern "system" fn(session: *const OrtSession, out: *mut u64) -> OrtStatusPtr,
	pub SetGlobalIntraOpNumThreads: unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, intra_op_num_threads: core::ffi::c_int) -> OrtStatusPtr,
	pub SetGlobalInterOpNumThreads: unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, inter_op_num_threads: core::ffi::c_int) -> OrtStatusPtr,
	pub SetGlobalSpinControl: unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, allow_spinning: core::ffi::c_int) -> OrtStatusPtr,
	pub AddInitializer: unsafe extern "system" fn(options: *mut OrtSessionOptions, name: *const core::ffi::c_char, val: *const OrtValue) -> OrtStatusPtr,
	pub CreateEnvWithCustomLoggerAndGlobalThreadPools: unsafe extern "system" fn(
		logging_function: OrtLoggingFunction,
		logger_param: *mut core::ffi::c_void,
		log_severity_level: OrtLoggingLevel,
		logid: *const core::ffi::c_char,
		tp_options: *const OrtThreadingOptions,
		out: *mut *mut OrtEnv
	) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_CUDA:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, cuda_options: *const OrtCUDAProviderOptions) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_ROCM:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, rocm_options: *const OrtROCMProviderOptions) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_OpenVINO:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, provider_options: *const OrtOpenVINOProviderOptions) -> OrtStatusPtr,
	pub SetGlobalDenormalAsZero: unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions) -> OrtStatusPtr,
	pub CreateArenaCfg: unsafe extern "system" fn(
		max_mem: usize,
		arena_extend_strategy: core::ffi::c_int,
		initial_chunk_size_bytes: core::ffi::c_int,
		max_dead_bytes_per_chunk: core::ffi::c_int,
		out: *mut *mut OrtArenaCfg
	) -> OrtStatusPtr,
	pub ReleaseArenaCfg: unsafe extern "system" fn(input: *mut OrtArenaCfg),
	pub ModelMetadataGetGraphDescription:
		unsafe extern "system" fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_TensorRT:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, tensorrt_options: *const OrtTensorRTProviderOptions) -> OrtStatusPtr,
	pub SetCurrentGpuDeviceId: unsafe extern "system" fn(device_id: core::ffi::c_int) -> OrtStatusPtr,
	pub GetCurrentGpuDeviceId: unsafe extern "system" fn(device_id: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub KernelInfoGetAttributeArray_float:
		unsafe extern "system" fn(info: *const OrtKernelInfo, name: *const core::ffi::c_char, out: *mut f32, size: *mut usize) -> OrtStatusPtr,
	pub KernelInfoGetAttributeArray_int64:
		unsafe extern "system" fn(info: *const OrtKernelInfo, name: *const core::ffi::c_char, out: *mut i64, size: *mut usize) -> OrtStatusPtr,
	pub CreateArenaCfgV2: unsafe extern "system" fn(
		arena_config_keys: *const *const core::ffi::c_char,
		arena_config_values: *const usize,
		num_keys: usize,
		out: *mut *mut OrtArenaCfg
	) -> OrtStatusPtr,
	pub AddRunConfigEntry:
		unsafe extern "system" fn(options: *mut OrtRunOptions, config_key: *const core::ffi::c_char, config_value: *const core::ffi::c_char) -> OrtStatusPtr,
	pub CreatePrepackedWeightsContainer: unsafe extern "system" fn(out: *mut *mut OrtPrepackedWeightsContainer) -> OrtStatusPtr,
	pub ReleasePrepackedWeightsContainer: unsafe extern "system" fn(input: *mut OrtPrepackedWeightsContainer),
	pub CreateSessionWithPrepackedWeightsContainer: unsafe extern "system" fn(
		env: *const OrtEnv,
		model_path: *const ortchar,
		options: *const OrtSessionOptions,
		prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
		out: *mut *mut OrtSession
	) -> OrtStatusPtr,
	pub CreateSessionFromArrayWithPrepackedWeightsContainer: unsafe extern "system" fn(
		env: *const OrtEnv,
		model_data: *const core::ffi::c_void,
		model_data_length: usize,
		options: *const OrtSessionOptions,
		prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
		out: *mut *mut OrtSession
	) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_TensorRT_V2:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, tensorrt_options: *const OrtTensorRTProviderOptionsV2) -> OrtStatusPtr,
	pub CreateTensorRTProviderOptions: unsafe extern "system" fn(out: *mut *mut OrtTensorRTProviderOptionsV2) -> OrtStatusPtr,
	pub UpdateTensorRTProviderOptions: unsafe extern "system" fn(
		tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub GetTensorRTProviderOptionsAsString: unsafe extern "system" fn(
		tensorrt_options: *const OrtTensorRTProviderOptionsV2,
		allocator: *mut OrtAllocator,
		ptr: *mut *mut core::ffi::c_char
	) -> OrtStatusPtr,
	#[doc = " \\brief Release an ::OrtTensorRTProviderOptionsV2\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does"]
	pub ReleaseTensorRTProviderOptions: unsafe extern "system" fn(input: *mut OrtTensorRTProviderOptionsV2),
	pub EnableOrtCustomOps: unsafe extern "system" fn(options: *mut OrtSessionOptions) -> OrtStatusPtr,
	pub RegisterAllocator: unsafe extern "system" fn(env: *mut OrtEnv, allocator: *mut OrtAllocator) -> OrtStatusPtr,
	pub UnregisterAllocator: unsafe extern "system" fn(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo) -> OrtStatusPtr,
	pub IsSparseTensor: unsafe extern "system" fn(value: *const OrtValue, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub CreateSparseTensorAsOrtValue: unsafe extern "system" fn(
		allocator: *mut OrtAllocator,
		dense_shape: *const i64,
		dense_shape_len: usize,
		type_: ONNXTensorElementDataType,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub FillSparseTensorCoo: unsafe extern "system" fn(
		ort_value: *mut OrtValue,
		data_mem_info: *const OrtMemoryInfo,
		values_shape: *const i64,
		values_shape_len: usize,
		values: *const core::ffi::c_void,
		indices_data: *const i64,
		indices_num: usize
	) -> OrtStatusPtr,
	pub FillSparseTensorCsr: unsafe extern "system" fn(
		ort_value: *mut OrtValue,
		data_mem_info: *const OrtMemoryInfo,
		values_shape: *const i64,
		values_shape_len: usize,
		values: *const core::ffi::c_void,
		inner_indices_data: *const i64,
		inner_indices_num: usize,
		outer_indices_data: *const i64,
		outer_indices_num: usize
	) -> OrtStatusPtr,
	pub FillSparseTensorBlockSparse: unsafe extern "system" fn(
		ort_value: *mut OrtValue,
		data_mem_info: *const OrtMemoryInfo,
		values_shape: *const i64,
		values_shape_len: usize,
		values: *const core::ffi::c_void,
		indices_shape_data: *const i64,
		indices_shape_len: usize,
		indices_data: *const i32
	) -> OrtStatusPtr,
	pub CreateSparseTensorWithValuesAsOrtValue: unsafe extern "system" fn(
		info: *const OrtMemoryInfo,
		p_data: *mut core::ffi::c_void,
		dense_shape: *const i64,
		dense_shape_len: usize,
		values_shape: *const i64,
		values_shape_len: usize,
		type_: ONNXTensorElementDataType,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub UseCooIndices: unsafe extern "system" fn(ort_value: *mut OrtValue, indices_data: *mut i64, indices_num: usize) -> OrtStatusPtr,
	pub UseCsrIndices:
		unsafe extern "system" fn(ort_value: *mut OrtValue, inner_data: *mut i64, inner_num: usize, outer_data: *mut i64, outer_num: usize) -> OrtStatusPtr,
	pub UseBlockSparseIndices:
		unsafe extern "system" fn(ort_value: *mut OrtValue, indices_shape: *const i64, indices_shape_len: usize, indices_data: *mut i32) -> OrtStatusPtr,
	pub GetSparseTensorFormat: unsafe extern "system" fn(ort_value: *const OrtValue, out: *mut OrtSparseFormat) -> OrtStatusPtr,
	pub GetSparseTensorValuesTypeAndShape: unsafe extern "system" fn(ort_value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub GetSparseTensorValues: unsafe extern "system" fn(ort_value: *const OrtValue, out: *mut *const core::ffi::c_void) -> OrtStatusPtr,
	pub GetSparseTensorIndicesTypeShape:
		unsafe extern "system" fn(ort_value: *const OrtValue, indices_format: OrtSparseIndicesFormat, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub GetSparseTensorIndices: unsafe extern "system" fn(
		ort_value: *const OrtValue,
		indices_format: OrtSparseIndicesFormat,
		num_indices: *mut usize,
		indices: *mut *const core::ffi::c_void
	) -> OrtStatusPtr,
	pub HasValue: unsafe extern "system" fn(value: *const OrtValue, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub KernelContext_GetGPUComputeStream: unsafe extern "system" fn(context: *const OrtKernelContext, out: *mut *mut core::ffi::c_void) -> OrtStatusPtr,
	pub GetTensorMemoryInfo: unsafe extern "system" fn(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr,
	pub GetExecutionProviderApi:
		unsafe extern "system" fn(provider_name: *const core::ffi::c_char, version: u32, provider_api: *mut *const core::ffi::c_void) -> OrtStatusPtr,
	pub SessionOptionsSetCustomCreateThreadFn:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr,
	pub SessionOptionsSetCustomThreadCreationOptions:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, ort_custom_thread_creation_options: *mut core::ffi::c_void) -> OrtStatusPtr,
	pub SessionOptionsSetCustomJoinThreadFn:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr,
	pub SetGlobalCustomCreateThreadFn:
		unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr,
	pub SetGlobalCustomThreadCreationOptions:
		unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, ort_custom_thread_creation_options: *mut core::ffi::c_void) -> OrtStatusPtr,
	pub SetGlobalCustomJoinThreadFn:
		unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr,
	pub SynchronizeBoundInputs: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr,
	pub SynchronizeBoundOutputs: unsafe extern "system" fn(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_CUDA_V2:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, cuda_options: *const OrtCUDAProviderOptionsV2) -> OrtStatusPtr,
	pub CreateCUDAProviderOptions: unsafe extern "system" fn(out: *mut *mut OrtCUDAProviderOptionsV2) -> OrtStatusPtr,
	pub UpdateCUDAProviderOptions: unsafe extern "system" fn(
		cuda_options: *mut OrtCUDAProviderOptionsV2,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub GetCUDAProviderOptionsAsString: unsafe extern "system" fn(
		cuda_options: *const OrtCUDAProviderOptionsV2,
		allocator: *mut OrtAllocator,
		ptr: *mut *mut core::ffi::c_char
	) -> OrtStatusPtr,
	#[doc = " \\brief Release an ::OrtCUDAProviderOptionsV2\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does\n\n \\since Version 1.11."]
	pub ReleaseCUDAProviderOptions: unsafe extern "system" fn(input: *mut OrtCUDAProviderOptionsV2),
	pub SessionOptionsAppendExecutionProvider_MIGraphX:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, migraphx_options: *const OrtMIGraphXProviderOptions) -> OrtStatusPtr,
	pub AddExternalInitializers: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		initializer_names: *const *const core::ffi::c_char,
		initializers: *const *const OrtValue,
		initializers_num: usize
	) -> OrtStatusPtr,
	pub CreateOpAttr: unsafe extern "system" fn(
		name: *const core::ffi::c_char,
		data: *const core::ffi::c_void,
		len: core::ffi::c_int,
		type_: OrtOpAttrType,
		op_attr: *mut *mut OrtOpAttr
	) -> OrtStatusPtr,
	pub ReleaseOpAttr: unsafe extern "system" fn(input: *mut OrtOpAttr),
	pub CreateOp: unsafe extern "system" fn(
		info: *const OrtKernelInfo,
		op_name: *const core::ffi::c_char,
		domain: *const core::ffi::c_char,
		version: core::ffi::c_int,
		type_constraint_names: *mut *const core::ffi::c_char,
		type_constraint_values: *const ONNXTensorElementDataType,
		type_constraint_count: core::ffi::c_int,
		attr_values: *const *const OrtOpAttr,
		attr_count: core::ffi::c_int,
		input_count: core::ffi::c_int,
		output_count: core::ffi::c_int,
		ort_op: *mut *mut OrtOp
	) -> OrtStatusPtr,
	pub InvokeOp: unsafe extern "system" fn(
		context: *const OrtKernelContext,
		ort_op: *const OrtOp,
		input_values: *const *const OrtValue,
		input_count: core::ffi::c_int,
		output_values: *const *mut OrtValue,
		output_count: core::ffi::c_int
	) -> OrtStatusPtr,
	pub ReleaseOp: unsafe extern "system" fn(input: *mut OrtOp),
	pub SessionOptionsAppendExecutionProvider: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		provider_name: *const core::ffi::c_char,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub CopyKernelInfo: unsafe extern "system" fn(info: *const OrtKernelInfo, info_copy: *mut *mut OrtKernelInfo) -> OrtStatusPtr,
	pub ReleaseKernelInfo: unsafe extern "system" fn(input: *mut OrtKernelInfo),
	#[doc = " \\name Ort Training\n @{\n** \\brief Gets the Training C Api struct\n*\n* Call this function to access the ::OrtTrainingApi structure that holds pointers to functions that enable\n* training with onnxruntime.\n* \\note A NULL pointer will be returned and no error message will be printed if the training api\n* is not supported with this build. A NULL pointer will be returned and an error message will be\n* printed if the provided version is unsupported, for example when using a runtime older than the\n* version created with this header file.\n*\n* \\param[in] version Must be ::ORT_API_VERSION\n* \\return The ::OrtTrainingApi struct for the version requested.\n*\n* \\since Version 1.13\n*/"]
	pub GetTrainingApi: unsafe extern "system" fn(version: u32) -> *const OrtTrainingApi,
	pub SessionOptionsAppendExecutionProvider_CANN:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, cann_options: *const OrtCANNProviderOptions) -> OrtStatusPtr,
	pub CreateCANNProviderOptions: unsafe extern "system" fn(out: *mut *mut OrtCANNProviderOptions) -> OrtStatusPtr,
	pub UpdateCANNProviderOptions: unsafe extern "system" fn(
		cann_options: *mut OrtCANNProviderOptions,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub GetCANNProviderOptionsAsString:
		unsafe extern "system" fn(cann_options: *const OrtCANNProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	#[doc = " \\brief Release an OrtCANNProviderOptions\n\n \\param[in] the pointer of OrtCANNProviderOptions which will been deleted\n\n \\since Version 1.13."]
	pub ReleaseCANNProviderOptions: unsafe extern "system" fn(input: *mut OrtCANNProviderOptions),
	pub MemoryInfoGetDeviceType: unsafe extern "system" fn(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType),
	pub UpdateEnvWithCustomLogLevel: unsafe extern "system" fn(ort_env: *mut OrtEnv, log_severity_level: OrtLoggingLevel) -> OrtStatusPtr,
	pub SetGlobalIntraOpThreadAffinity:
		unsafe extern "system" fn(tp_options: *mut OrtThreadingOptions, affinity_string: *const core::ffi::c_char) -> OrtStatusPtr,
	pub RegisterCustomOpsLibrary_V2: unsafe extern "system" fn(options: *mut OrtSessionOptions, library_name: *const ortchar) -> OrtStatusPtr,
	pub RegisterCustomOpsUsingFunction:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, registration_func_name: *const core::ffi::c_char) -> OrtStatusPtr,
	pub KernelInfo_GetInputCount: unsafe extern "system" fn(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr,
	pub KernelInfo_GetOutputCount: unsafe extern "system" fn(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr,
	pub KernelInfo_GetInputName:
		unsafe extern "system" fn(info: *const OrtKernelInfo, index: usize, out: *mut core::ffi::c_char, size: *mut usize) -> OrtStatusPtr,
	pub KernelInfo_GetOutputName:
		unsafe extern "system" fn(info: *const OrtKernelInfo, index: usize, out: *mut core::ffi::c_char, size: *mut usize) -> OrtStatusPtr,
	pub KernelInfo_GetInputTypeInfo: unsafe extern "system" fn(info: *const OrtKernelInfo, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub KernelInfo_GetOutputTypeInfo: unsafe extern "system" fn(info: *const OrtKernelInfo, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub KernelInfoGetAttribute_tensor: unsafe extern "system" fn(
		info: *const OrtKernelInfo,
		name: *const core::ffi::c_char,
		allocator: *mut OrtAllocator,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub HasSessionConfigEntry:
		unsafe extern "system" fn(options: *const OrtSessionOptions, config_key: *const core::ffi::c_char, out: *mut core::ffi::c_int) -> OrtStatusPtr,
	pub GetSessionConfigEntry: unsafe extern "system" fn(
		options: *const OrtSessionOptions,
		config_key: *const core::ffi::c_char,
		config_value: *mut core::ffi::c_char,
		size: *mut usize
	) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_Dnnl:
		unsafe extern "system" fn(options: *mut OrtSessionOptions, dnnl_options: *const OrtDnnlProviderOptions) -> OrtStatusPtr,
	pub CreateDnnlProviderOptions: unsafe extern "system" fn(out: *mut *mut OrtDnnlProviderOptions) -> OrtStatusPtr,
	pub UpdateDnnlProviderOptions: unsafe extern "system" fn(
		dnnl_options: *mut OrtDnnlProviderOptions,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub GetDnnlProviderOptionsAsString:
		unsafe extern "system" fn(dnnl_options: *const OrtDnnlProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	#[doc = " \\brief Release an ::OrtDnnlProviderOptions\n\n \\since Version 1.15."]
	pub ReleaseDnnlProviderOptions: unsafe extern "system" fn(input: *mut OrtDnnlProviderOptions),
	pub KernelInfo_GetNodeName: unsafe extern "system" fn(info: *const OrtKernelInfo, out: *mut core::ffi::c_char, size: *mut usize) -> OrtStatusPtr,
	pub KernelInfo_GetLogger: unsafe extern "system" fn(info: *const OrtKernelInfo, logger: *mut *const OrtLogger) -> OrtStatusPtr,
	pub KernelContext_GetLogger: unsafe extern "system" fn(context: *const OrtKernelContext, logger: *mut *const OrtLogger) -> OrtStatusPtr,
	pub Logger_LogMessage: unsafe extern "system" fn(
		logger: *const OrtLogger,
		log_severity_level: OrtLoggingLevel,
		message: *const core::ffi::c_char,
		file_path: *const ortchar,
		line_number: core::ffi::c_int,
		func_name: *const core::ffi::c_char
	) -> OrtStatusPtr,
	pub Logger_GetLoggingSeverityLevel: unsafe extern "system" fn(logger: *const OrtLogger, out: *mut OrtLoggingLevel) -> OrtStatusPtr,
	pub KernelInfoGetConstantInput_tensor:
		unsafe extern "system" fn(info: *const OrtKernelInfo, index: usize, is_constant: *mut core::ffi::c_int, out: *mut *const OrtValue) -> OrtStatusPtr,
	pub CastTypeInfoToOptionalTypeInfo: unsafe extern "system" fn(type_info: *const OrtTypeInfo, out: *mut *const OrtOptionalTypeInfo) -> OrtStatusPtr,
	pub GetOptionalContainedTypeInfo: unsafe extern "system" fn(optional_type_info: *const OrtOptionalTypeInfo, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr,
	pub GetResizedStringTensorElementBuffer:
		unsafe extern "system" fn(value: *mut OrtValue, index: usize, length_in_bytes: usize, buffer: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	pub KernelContext_GetAllocator:
		unsafe extern "system" fn(context: *const OrtKernelContext, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr,
	#[doc = " \\brief Returns a null terminated string of the build info including git info and cxx flags\n\n \\return UTF-8 encoded version string. Do not deallocate the returned buffer.\n\n \\since Version 1.15."]
	pub GetBuildInfoString: unsafe extern "system" fn() -> *const core::ffi::c_char,
	pub CreateROCMProviderOptions: unsafe extern "system" fn(out: *mut *mut OrtROCMProviderOptions) -> OrtStatusPtr,
	pub UpdateROCMProviderOptions: unsafe extern "system" fn(
		rocm_options: *mut OrtROCMProviderOptions,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub GetROCMProviderOptionsAsString:
		unsafe extern "system" fn(rocm_options: *const OrtROCMProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut core::ffi::c_char) -> OrtStatusPtr,
	#[doc = " \\brief Release an ::OrtROCMProviderOptions\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does\n\n \\since Version 1.16."]
	pub ReleaseROCMProviderOptions: unsafe extern "system" fn(input: *mut OrtROCMProviderOptions),
	pub CreateAndRegisterAllocatorV2: unsafe extern "system" fn(
		env: *mut OrtEnv,
		provider_type: *const core::ffi::c_char,
		mem_info: *const OrtMemoryInfo,
		arena_cfg: *const OrtArenaCfg,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub RunAsync: unsafe extern "system" fn(
		session: *mut OrtSession,
		run_options: *const OrtRunOptions,
		input_names: *const *const core::ffi::c_char,
		input: *const *const OrtValue,
		input_len: usize,
		output_names: *const *const core::ffi::c_char,
		output_names_len: usize,
		output: *mut *mut OrtValue,
		run_async_callback: RunAsyncCallbackFn,
		user_data: *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub UpdateTensorRTProviderOptionsWithValue: unsafe extern "system" fn(
		tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
		key: *const core::ffi::c_char,
		value: *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub GetTensorRTProviderOptionsByName: unsafe extern "system" fn(
		tensorrt_options: *const OrtTensorRTProviderOptionsV2,
		key: *const core::ffi::c_char,
		ptr: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub UpdateCUDAProviderOptionsWithValue:
		unsafe extern "system" fn(cuda_options: *mut OrtCUDAProviderOptionsV2, key: *const core::ffi::c_char, value: *mut core::ffi::c_void) -> OrtStatusPtr,
	pub GetCUDAProviderOptionsByName: unsafe extern "system" fn(
		cuda_options: *const OrtCUDAProviderOptionsV2,
		key: *const core::ffi::c_char,
		ptr: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub KernelContext_GetResource: unsafe extern "system" fn(
		context: *const OrtKernelContext,
		resouce_version: core::ffi::c_int,
		resource_id: core::ffi::c_int,
		resource: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub SetUserLoggingFunction: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		user_logging_function: OrtLoggingFunction,
		user_logging_param: *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub ShapeInferContext_GetInputCount: unsafe extern "system" fn(context: *const OrtShapeInferContext, out: *mut usize) -> OrtStatusPtr,
	pub ShapeInferContext_GetInputTypeShape:
		unsafe extern "system" fn(context: *const OrtShapeInferContext, index: usize, info: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub ShapeInferContext_GetAttribute:
		unsafe extern "system" fn(context: *const OrtShapeInferContext, attr_name: *const core::ffi::c_char, attr: *mut *const OrtOpAttr) -> OrtStatusPtr,
	pub ShapeInferContext_SetOutputTypeShape:
		unsafe extern "system" fn(context: *const OrtShapeInferContext, index: usize, info: *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr,
	pub SetSymbolicDimensions:
		unsafe extern "system" fn(info: *mut OrtTensorTypeAndShapeInfo, dim_params: *mut *const core::ffi::c_char, dim_params_length: usize) -> OrtStatusPtr,
	pub ReadOpAttr:
		unsafe extern "system" fn(op_attr: *const OrtOpAttr, type_: OrtOpAttrType, data: *mut core::ffi::c_void, len: usize, out: *mut usize) -> OrtStatusPtr,
	pub SetDeterministicCompute: unsafe extern "system" fn(options: *mut OrtSessionOptions, value: bool) -> OrtStatusPtr,
	pub KernelContext_ParallelFor: unsafe extern "system" fn(
		context: *const OrtKernelContext,
		fn_: unsafe extern "system" fn(arg1: *mut core::ffi::c_void, arg2: usize),
		total: usize,
		num_batch: usize,
		usr_data: *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_OpenVINO_V2: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_VitisAI: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		provider_options_keys: *const *const core::ffi::c_char,
		provider_options_values: *const *const core::ffi::c_char,
		num_keys: usize
	) -> OrtStatusPtr,
	pub KernelContext_GetScratchBuffer: unsafe extern "system" fn(
		context: *const OrtKernelContext,
		mem_info: *const OrtMemoryInfo,
		count_or_bytes: usize,
		out: *mut *mut core::ffi::c_void
	) -> OrtStatusPtr,
	pub KernelInfoGetAllocator: unsafe extern "system" fn(info: *const OrtKernelInfo, mem_type: OrtMemType, out: *mut *mut OrtAllocator) -> OrtStatusPtr,
	pub AddExternalInitializersFromMemory: unsafe extern "system" fn(
		options: *mut OrtSessionOptions,
		external_initializer_file_names: *const *const ortchar,
		external_initializer_file_buffer_array: *const *mut core::ffi::c_char,
		external_initializer_file_lengths: *const usize,
		num_external_initializer_files: usize
	) -> OrtStatusPtr,
	pub CreateLoraAdapter:
		unsafe extern "system" fn(adapter_file_path: *const ortchar, allocator: *mut OrtAllocator, out: *mut *mut OrtLoraAdapter) -> OrtStatusPtr,
	pub CreateLoraAdapterFromArray: unsafe extern "system" fn(
		bytes: *const core::ffi::c_void,
		num_bytes: usize,
		allocator: *mut OrtAllocator,
		out: *mut *mut OrtLoraAdapter
	) -> OrtStatusPtr,
	pub ReleaseLoraAdapter: unsafe extern "system" fn(input: *mut OrtLoraAdapter),
	pub RunOptionsAddActiveLoraAdapter: unsafe extern "system" fn(options: *mut OrtRunOptions, adapter: *const OrtLoraAdapter) -> OrtStatusPtr,
	pub SetEpDynamicOptions: unsafe extern "system" fn(
		sess: *mut OrtSession,
		keys: *const *const core::ffi::c_char,
		values: *const *const core::ffi::c_char,
		kv_len: usize
	) -> OrtStatusPtr,
	pub ReleaseValueInfo: unsafe extern "system" fn(input: *mut OrtValueInfo),
	pub ReleaseNode: unsafe extern "system" fn(input: *mut OrtNode),
	pub ReleaseGraph: unsafe extern "system" fn(input: *mut OrtGraph),
	pub ReleaseModel: unsafe extern "system" fn(input: *mut OrtModel),
	pub GetValueInfoName: unsafe extern "system" fn(value_info: *const OrtValueInfo, name: *mut *const c_char) -> OrtStatusPtr,
	pub GetValueInfoTypeInfo: unsafe extern "system" fn(value_info: *const OrtValueInfo, type_info: *mut *const OrtTypeInfo) -> OrtStatusPtr,
	pub GetModelEditorApi: unsafe extern "system" fn() -> *const OrtModelEditorApi,
	pub CreateTensorWithDataAndDeleterAsOrtValue: unsafe extern "system" fn(
		deleter: *mut OrtAllocator,
		p_data: *mut c_void,
		p_data_len: usize,
		shape: *const i64,
		shape_len: usize,
		r#type: ONNXTensorElementDataType,
		out: *mut *mut OrtValue
	) -> OrtStatusPtr,
	pub SessionOptionsSetLoadCancellationFlag: unsafe extern "system" fn(options: *mut OrtSessionOptions, cancel: bool) -> OrtStatusPtr,
	pub GetCompileApi: unsafe extern "system" fn() -> *const OrtCompileApi,
	pub CreateKeyValuePairs: unsafe extern "system" fn(out: *mut *mut OrtKeyValuePairs),
	pub AddKeyValuePair: unsafe extern "system" fn(kvps: *mut OrtKeyValuePairs, key: *const c_char, value: *const c_char),
	pub GetKeyValue: unsafe extern "system" fn(kvps: *const OrtKeyValuePairs, key: *const c_char) -> *const c_char,
	pub GetKeyValuePairs:
		unsafe extern "system" fn(kvps: *const OrtKeyValuePairs, keys: *mut *const *const c_char, values: *mut *const *const c_char, num_entries: *mut usize),
	pub RemoveKeyValuePair: unsafe extern "system" fn(kvps: *mut OrtKeyValuePairs, key: *const c_char),
	pub ReleaseKeyValuePairs: unsafe extern "system" fn(input: *mut OrtKeyValuePairs),
	pub RegisterExecutionProviderLibrary: unsafe extern "system" fn(env: *mut OrtEnv, registration_name: *const c_char, path: *const ortchar) -> OrtStatusPtr,
	pub UnregisterExecutionProviderLibrary: unsafe extern "system" fn(env: *mut OrtEnv, registration_name: *const c_char) -> OrtStatusPtr,
	pub GetEpDevices: unsafe extern "system" fn(env: *const OrtEnv, ep_devices: *mut *const *const OrtEpDevice, num_ep_devices: *mut usize) -> OrtStatusPtr,
	pub SessionOptionsAppendExecutionProvider_V2: unsafe extern "system" fn(
		session_options: *mut OrtSessionOptions,
		env: *mut OrtEnv,
		ep_devices: *const *const OrtEpDevice,
		num_ep_devices: usize,
		ep_option_keys: *const *const c_char,
		ep_option_vals: *const *const c_char,
		num_ep_options: usize
	) -> OrtStatusPtr,
	pub SessionOptionsSetEpSelectionPolicy:
		unsafe extern "system" fn(session_options: *mut OrtSessionOptions, policy: OrtExecutionProviderDevicePolicy) -> OrtStatusPtr,
	pub SessionOptionsSetEpSelectionPolicyDelegate:
		unsafe extern "system" fn(session_options: *mut OrtSessionOptions, delegate: EpSelectionDelegate, delegate_state: *mut c_void) -> OrtStatusPtr,
	pub HardwareDevice_Type: unsafe extern "system" fn(device: *const OrtHardwareDevice) -> OrtHardwareDeviceType,
	pub HardwareDevice_VendorId: unsafe extern "system" fn(device: *const OrtHardwareDevice) -> u32,
	pub HardwareDevice_Vendor: unsafe extern "system" fn(device: *const OrtHardwareDevice) -> *const c_char,
	pub HardwareDevice_DeviceId: unsafe extern "system" fn(device: *const OrtHardwareDevice) -> u32,
	pub HardwareDevice_Metadata: unsafe extern "system" fn(device: *const OrtHardwareDevice) -> *const OrtKeyValuePairs,
	pub EpDevice_EpName: unsafe extern "system" fn(ep_device: *const OrtEpDevice) -> *const c_char,
	pub EpDevice_EpVendor: unsafe extern "system" fn(ep_device: *const OrtEpDevice) -> *const c_char,
	pub EpDevice_EpMetadata: unsafe extern "system" fn(ep_device: *const OrtEpDevice) -> *const OrtKeyValuePairs,
	pub EpDevice_EpOptions: unsafe extern "system" fn(ep_device: *const OrtEpDevice) -> *const OrtKeyValuePairs,
	pub EpDevice_Device: unsafe extern "system" fn(ep_device: *const OrtEpDevice) -> *const OrtHardwareDevice,
	pub GetEpApi: unsafe extern "system" fn() -> *const OrtEpApi
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum OrtCustomOpInputOutputCharacteristic {
	INPUT_OUTPUT_REQUIRED = 0,
	INPUT_OUTPUT_OPTIONAL = 1,
	INPUT_OUTPUT_VARIADIC = 2
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCustomOp {
	pub version: u32,
	pub CreateKernel: Option<unsafe extern "system" fn(op: *const OrtCustomOp, api: *const OrtApi, info: *const OrtKernelInfo) -> *mut core::ffi::c_void>,
	pub GetName: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> *const core::ffi::c_char>,
	pub GetExecutionProviderType: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> *const core::ffi::c_char>,
	pub GetInputType: Option<unsafe extern "system" fn(op: *const OrtCustomOp, index: usize) -> ONNXTensorElementDataType>,
	pub GetInputTypeCount: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> usize>,
	pub GetOutputType: Option<unsafe extern "system" fn(op: *const OrtCustomOp, index: usize) -> ONNXTensorElementDataType>,
	pub GetOutputTypeCount: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> usize>,
	pub KernelCompute: Option<unsafe extern "system" fn(op_kernel: *mut core::ffi::c_void, context: *mut OrtKernelContext)>,
	pub KernelDestroy: Option<unsafe extern "system" fn(op_kernel: *mut core::ffi::c_void)>,
	pub GetInputCharacteristic: Option<unsafe extern "system" fn(op: *const OrtCustomOp, index: usize) -> OrtCustomOpInputOutputCharacteristic>,
	pub GetOutputCharacteristic: Option<unsafe extern "system" fn(op: *const OrtCustomOp, index: usize) -> OrtCustomOpInputOutputCharacteristic>,
	pub GetInputMemoryType: Option<unsafe extern "system" fn(op: *const OrtCustomOp, index: usize) -> OrtMemType>,
	pub GetVariadicInputMinArity: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub GetVariadicInputHomogeneity: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub GetVariadicOutputMinArity: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub GetVariadicOutputHomogeneity: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub CreateKernelV2: Option<
		unsafe extern "system" fn(op: *const OrtCustomOp, api: *const OrtApi, info: *const OrtKernelInfo, kernel: *mut *mut core::ffi::c_void) -> OrtStatusPtr
	>,
	pub KernelComputeV2: Option<unsafe extern "system" fn(op_kernel: *mut core::ffi::c_void, context: *mut OrtKernelContext) -> OrtStatusPtr>,
	pub InferOutputShapeFn: Option<unsafe extern "system" fn(op: *const OrtCustomOp, arg1: *mut OrtShapeInferContext) -> OrtStatusPtr>,
	pub GetStartVersion: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub GetEndVersion: Option<unsafe extern "system" fn(op: *const OrtCustomOp) -> core::ffi::c_int>,
	pub GetMayInplace: Option<unsafe extern "system" fn(input_index: *mut *mut core::ffi::c_int, output_index: *mut *mut core::ffi::c_int) -> usize>,
	pub ReleaseMayInplace: Option<unsafe extern "system" fn(input_index: *mut core::ffi::c_int, output_index: *mut *mut core::ffi::c_int)>,
	pub GetAliasMap: Option<unsafe extern "system" fn(input_index: *mut *mut core::ffi::c_int, output_index: *mut *mut core::ffi::c_int) -> usize>,
	pub ReleaseAliasMap: Option<unsafe extern "system" fn(input_index: *mut core::ffi::c_int, output_index: *mut *mut core::ffi::c_int)>
}
extern "system" {
	pub fn OrtSessionOptionsAppendExecutionProvider_CUDA(options: *mut OrtSessionOptions, device_id: core::ffi::c_int) -> OrtStatusPtr;
	pub fn OrtSessionOptionsAppendExecutionProvider_ROCM(options: *mut OrtSessionOptions, device_id: core::ffi::c_int) -> OrtStatusPtr;
	pub fn OrtSessionOptionsAppendExecutionProvider_MIGraphX(options: *mut OrtSessionOptions, device_id: core::ffi::c_int) -> OrtStatusPtr;
	pub fn OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut OrtSessionOptions, use_arena: core::ffi::c_int) -> OrtStatusPtr;
	pub fn OrtSessionOptionsAppendExecutionProvider_Tensorrt(options: *mut OrtSessionOptions, device_id: core::ffi::c_int) -> OrtStatusPtr;
}
