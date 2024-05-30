#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
// Disable clippy and `u128` not being FFI-safe
#![allow(clippy::all)]
#![allow(improper_ctypes)]
// bindgen-rs generates test code that dereferences null pointers
#![allow(deref_nullptr)]

#[doc(hidden)]
pub mod internal;

pub const ORT_API_VERSION: u32 = 18;

pub use std::ffi::{c_char, c_int, c_ulong, c_ulonglong, c_ushort, c_void};

#[cfg(target_os = "windows")]
pub type ortchar = c_ushort;
#[cfg(not(target_os = "windows"))]
pub type ortchar = c_char;

#[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "wasm32"))]
pub type size_t = usize;
#[cfg(all(target_arch = "aarch64", target_os = "windows"))]
pub type size_t = c_ulonglong;
#[cfg(all(any(target_arch = "aarch64", target_arch = "arm"), not(target_os = "windows")))]
pub type size_t = c_ulong;

#[cfg(not(all(target_arch = "x86", target_os = "windows")))]
macro_rules! _system {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}
#[cfg(all(target_arch = "x86", target_os = "windows"))]
macro_rules! _system {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "stdcall" fn $($tt)*);
}

#[cfg(not(all(target_arch = "x86", target_os = "windows")))]
macro_rules! _system_block {
	($(#[$meta:meta])* fn $($tt:tt)*) => (extern "C" { $(#[$meta])* fn $($tt)* });
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => (extern "C" { $(#[$meta])* $vis fn $($tt)* });
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => (extern "C" { $(#[$meta])* unsafe fn $($tt)* });
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => (extern "C" { $(#[$meta])* $vis unsafe fn $($tt)* });
}
#[cfg(all(target_arch = "x86", target_os = "windows"))]
macro_rules! _system_block {
	($(#[$meta:meta])* fn $($tt:tt)*) => (extern "stdcall" { $(#[$meta])* fn $($tt)* });
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => (extern "stdcall" { $(#[$meta])* $vis fn $($tt)* });
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => (extern "stdcall" { $(#[$meta])* unsafe fn $($tt)* });
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => (extern "stdcall" { $(#[$meta])* $vis unsafe fn $($tt)* });
}

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
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ = 20
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
pub type OrtStatusPtr = *mut OrtStatus;
#[doc = " \\brief Memory allocation interface\n\n Structure of function pointers that defines a memory allocator. This can be created and filled in by the user for custom allocators.\n\n When an allocator is passed to any function, be sure that the allocator object is not destroyed until the last allocated object using it is freed."]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtAllocator {
	#[doc = "< Must be initialized to ORT_API_VERSION"]
	pub version: u32,
	#[doc = "< Returns a pointer to an allocated block of `size` bytes"]
	pub Alloc: ::std::option::Option<_system!(unsafe fn(this_: *mut OrtAllocator, size: size_t) -> *mut ::std::os::raw::c_void)>,
	#[doc = "< Free a block of memory previously allocated with OrtAllocator::Alloc"]
	pub Free: ::std::option::Option<_system!(unsafe fn(this_: *mut OrtAllocator, p: *mut ::std::os::raw::c_void))>,
	#[doc = "< Return a pointer to an ::OrtMemoryInfo that describes this allocator"]
	pub Info: ::std::option::Option<_system!(unsafe fn(this_: *const OrtAllocator) -> *const OrtMemoryInfo)>,
	pub Reserve: ::std::option::Option<_system!(unsafe fn(this_: *const OrtAllocator, size: size_t) -> *mut ::std::os::raw::c_void)>
}
#[test]
fn bindgen_test_layout_OrtAllocator() {
	const UNINIT: ::std::mem::MaybeUninit<OrtAllocator> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtAllocator>(), 32usize, concat!("Size of: ", stringify!(OrtAllocator)));
	assert_eq!(::std::mem::align_of::<OrtAllocator>(), 8usize, concat!("Alignment of ", stringify!(OrtAllocator)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).version) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtAllocator), "::", stringify!(version))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Alloc) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtAllocator), "::", stringify!(Alloc))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Free) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtAllocator), "::", stringify!(Free))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Info) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtAllocator), "::", stringify!(Info))
	);
}
pub type OrtLoggingFunction = ::std::option::Option<
	_system!(
		unsafe fn(
			param: *mut ::std::os::raw::c_void,
			severity: OrtLoggingLevel,
			category: *const ::std::os::raw::c_char,
			logid: *const ::std::os::raw::c_char,
			code_location: *const ::std::os::raw::c_char,
			message: *const ::std::os::raw::c_char
		)
	)
>;
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
	pub device_id: ::std::os::raw::c_int,
	#[doc = " \\brief CUDA Convolution algorithm search configuration.\n   See enum OrtCudnnConvAlgoSearch for more details.\n   Defaults to OrtCudnnConvAlgoSearchExhaustive."]
	pub cudnn_conv_algo_search: OrtCudnnConvAlgoSearch,
	#[doc = " \\brief CUDA memory limit (To use all possible memory pass in maximum size_t)\n   Defaults to SIZE_MAX.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub gpu_mem_limit: size_t,
	#[doc = " \\brief Strategy used to grow the memory arena\n   0 = kNextPowerOfTwo<br>\n   1 = kSameAsRequested<br>\n   Defaults to 0.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub arena_extend_strategy: ::std::os::raw::c_int,
	#[doc = " \\brief Flag indicating if copying needs to take place on the same stream as the compute stream in the CUDA EP\n   0 = Use separate streams for copying and compute.\n   1 = Use the same stream for copying and compute.\n   Defaults to 1.\n   WARNING: Setting this to 0 may result in data races for some models.\n   Please see issue #4829 for more details."]
	pub do_copy_in_default_stream: ::std::os::raw::c_int,
	#[doc = " \\brief Flag indicating if there is a user provided compute stream\n   Defaults to 0."]
	pub has_user_compute_stream: ::std::os::raw::c_int,
	#[doc = " \\brief User provided compute stream.\n   If provided, please set `has_user_compute_stream` to 1."]
	pub user_compute_stream: *mut ::std::os::raw::c_void,
	#[doc = " \\brief CUDA memory arena configuration parameters"]
	pub default_memory_arena_cfg: *mut OrtArenaCfg,
	#[doc = " \\brief Enable TunableOp for using.\n   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_ENABLE."]
	pub tunable_op_enable: ::std::os::raw::c_int,
	#[doc = " \\brief Enable TunableOp for tuning.\n   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_CUDA_TUNABLE_OP_TUNING_ENABLE."]
	pub tunable_op_tuning_enable: ::std::os::raw::c_int,
	#[doc = " \\brief Max tuning duration time limit for each instance of TunableOp.\n   Defaults to 0 to disable the limit."]
	pub tunable_op_max_tuning_duration_ms: ::std::os::raw::c_int
}
#[test]
fn bindgen_test_layout_OrtCUDAProviderOptions() {
	const UNINIT: ::std::mem::MaybeUninit<OrtCUDAProviderOptions> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtCUDAProviderOptions>(), 64usize, concat!("Size of: ", stringify!(OrtCUDAProviderOptions)));
	assert_eq!(::std::mem::align_of::<OrtCUDAProviderOptions>(), 8usize, concat!("Alignment of ", stringify!(OrtCUDAProviderOptions)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_id) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(device_id))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).cudnn_conv_algo_search) as usize - ptr as usize },
		4usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(cudnn_conv_algo_search))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).gpu_mem_limit) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(gpu_mem_limit))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).arena_extend_strategy) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(arena_extend_strategy))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).do_copy_in_default_stream) as usize - ptr as usize },
		20usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(do_copy_in_default_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).has_user_compute_stream) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(has_user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).user_compute_stream) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).default_memory_arena_cfg) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(default_memory_arena_cfg))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_enable) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(tunable_op_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_tuning_enable) as usize - ptr as usize },
		52usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(tunable_op_tuning_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_max_tuning_duration_ms) as usize - ptr as usize },
		56usize,
		concat!("Offset of field: ", stringify!(OrtCUDAProviderOptions), "::", stringify!(tunable_op_max_tuning_duration_ms))
	);
}
#[doc = " \\brief ROCM Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_ROCM"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtROCMProviderOptions {
	#[doc = " \\brief ROCM device Id\n   Defaults to 0."]
	pub device_id: ::std::os::raw::c_int,
	#[doc = " \\brief ROCM MIOpen Convolution algorithm exaustive search option.\n   Defaults to 0 (false)."]
	pub miopen_conv_exhaustive_search: ::std::os::raw::c_int,
	#[doc = " \\brief ROCM memory limit (To use all possible memory pass in maximum size_t)\n   Defaults to SIZE_MAX.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub gpu_mem_limit: size_t,
	#[doc = " \\brief Strategy used to grow the memory arena\n   0 = kNextPowerOfTwo<br>\n   1 = kSameAsRequested<br>\n   Defaults to 0.\n   \\note If a ::OrtArenaCfg has been applied, it will override this field"]
	pub arena_extend_strategy: ::std::os::raw::c_int,
	#[doc = " \\brief Flag indicating if copying needs to take place on the same stream as the compute stream in the ROCM EP\n   0 = Use separate streams for copying and compute.\n   1 = Use the same stream for copying and compute.\n   Defaults to 1.\n   WARNING: Setting this to 0 may result in data races for some models.\n   Please see issue #4829 for more details."]
	pub do_copy_in_default_stream: ::std::os::raw::c_int,
	#[doc = " \\brief Flag indicating if there is a user provided compute stream\n   Defaults to 0."]
	pub has_user_compute_stream: ::std::os::raw::c_int,
	#[doc = " \\brief User provided compute stream.\n   If provided, please set `has_user_compute_stream` to 1."]
	pub user_compute_stream: *mut ::std::os::raw::c_void,
	#[doc = " \\brief ROCM memory arena configuration parameters"]
	pub default_memory_arena_cfg: *mut OrtArenaCfg,
	pub enable_hip_graph: ::std::os::raw::c_int,
	#[doc = " \\brief Enable TunableOp for using.\n   Set it to 1/0 to enable/disable TunableOp. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_ENABLE."]
	pub tunable_op_enable: ::std::os::raw::c_int,
	#[doc = " \\brief Enable TunableOp for tuning.\n   Set it to 1/0 to enable/disable TunableOp tuning. Otherwise, it is disabled by default.\n   This option can be overriden by environment variable ORT_ROCM_TUNABLE_OP_TUNING_ENABLE."]
	pub tunable_op_tuning_enable: ::std::os::raw::c_int,
	#[doc = " \\brief Max tuning duration time limit for each instance of TunableOp.\n   Defaults to 0 to disable the limit."]
	pub tunable_op_max_tuning_duration_ms: ::std::os::raw::c_int
}
#[test]
fn bindgen_test_layout_OrtROCMProviderOptions() {
	const UNINIT: ::std::mem::MaybeUninit<OrtROCMProviderOptions> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtROCMProviderOptions>(), 64usize, concat!("Size of: ", stringify!(OrtROCMProviderOptions)));
	assert_eq!(::std::mem::align_of::<OrtROCMProviderOptions>(), 8usize, concat!("Alignment of ", stringify!(OrtROCMProviderOptions)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_id) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(device_id))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).miopen_conv_exhaustive_search) as usize - ptr as usize },
		4usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(miopen_conv_exhaustive_search))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).gpu_mem_limit) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(gpu_mem_limit))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).arena_extend_strategy) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(arena_extend_strategy))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).do_copy_in_default_stream) as usize - ptr as usize },
		20usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(do_copy_in_default_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).has_user_compute_stream) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(has_user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).user_compute_stream) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).default_memory_arena_cfg) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(default_memory_arena_cfg))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_enable) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(tunable_op_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_tuning_enable) as usize - ptr as usize },
		52usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(tunable_op_tuning_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).tunable_op_max_tuning_duration_ms) as usize - ptr as usize },
		56usize,
		concat!("Offset of field: ", stringify!(OrtROCMProviderOptions), "::", stringify!(tunable_op_max_tuning_duration_ms))
	);
}
#[doc = " \\brief TensorRT Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_TensorRT"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtTensorRTProviderOptions {
	#[doc = "< CUDA device id (0 = default device)"]
	pub device_id: ::std::os::raw::c_int,
	pub has_user_compute_stream: ::std::os::raw::c_int,
	pub user_compute_stream: *mut ::std::os::raw::c_void,
	pub trt_max_partition_iterations: ::std::os::raw::c_int,
	pub trt_min_subgraph_size: ::std::os::raw::c_int,
	pub trt_max_workspace_size: size_t,
	pub trt_fp16_enable: ::std::os::raw::c_int,
	pub trt_int8_enable: ::std::os::raw::c_int,
	pub trt_int8_calibration_table_name: *const ::std::os::raw::c_char,
	pub trt_int8_use_native_calibration_table: ::std::os::raw::c_int,
	pub trt_dla_enable: ::std::os::raw::c_int,
	pub trt_dla_core: ::std::os::raw::c_int,
	pub trt_dump_subgraphs: ::std::os::raw::c_int,
	pub trt_engine_cache_enable: ::std::os::raw::c_int,
	pub trt_engine_cache_path: *const ::std::os::raw::c_char,
	pub trt_engine_decryption_enable: ::std::os::raw::c_int,
	pub trt_engine_decryption_lib_path: *const ::std::os::raw::c_char,
	pub trt_force_sequential_engine_build: ::std::os::raw::c_int
}
#[test]
fn bindgen_test_layout_OrtTensorRTProviderOptions() {
	const UNINIT: ::std::mem::MaybeUninit<OrtTensorRTProviderOptions> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtTensorRTProviderOptions>(), 104usize, concat!("Size of: ", stringify!(OrtTensorRTProviderOptions)));
	assert_eq!(::std::mem::align_of::<OrtTensorRTProviderOptions>(), 8usize, concat!("Alignment of ", stringify!(OrtTensorRTProviderOptions)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_id) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(device_id))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).has_user_compute_stream) as usize - ptr as usize },
		4usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(has_user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).user_compute_stream) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(user_compute_stream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_max_partition_iterations) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_max_partition_iterations))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_min_subgraph_size) as usize - ptr as usize },
		20usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_min_subgraph_size))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_max_workspace_size) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_max_workspace_size))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_fp16_enable) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_fp16_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_int8_enable) as usize - ptr as usize },
		36usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_int8_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_int8_calibration_table_name) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_int8_calibration_table_name))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_int8_use_native_calibration_table) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_int8_use_native_calibration_table))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_dla_enable) as usize - ptr as usize },
		52usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_dla_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_dla_core) as usize - ptr as usize },
		56usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_dla_core))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_dump_subgraphs) as usize - ptr as usize },
		60usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_dump_subgraphs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_engine_cache_enable) as usize - ptr as usize },
		64usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_engine_cache_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_engine_cache_path) as usize - ptr as usize },
		72usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_engine_cache_path))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_engine_decryption_enable) as usize - ptr as usize },
		80usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_engine_decryption_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_engine_decryption_lib_path) as usize - ptr as usize },
		88usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_engine_decryption_lib_path))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).trt_force_sequential_engine_build) as usize - ptr as usize },
		96usize,
		concat!("Offset of field: ", stringify!(OrtTensorRTProviderOptions), "::", stringify!(trt_force_sequential_engine_build))
	);
}
#[doc = " \\brief MIGraphX Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtMIGraphXProviderOptions {
	pub device_id: ::std::os::raw::c_int,
	pub migraphx_fp16_enable: ::std::os::raw::c_int,
	pub migraphx_int8_enable: ::std::os::raw::c_int,
	pub migraphx_use_native_calibration_table: ::std::os::raw::c_int,
	pub migraphx_int8_calibration_table_name: *const ::std::os::raw::c_char
}
#[test]
fn bindgen_test_layout_OrtMIGraphXProviderOptions() {
	const UNINIT: ::std::mem::MaybeUninit<OrtMIGraphXProviderOptions> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtMIGraphXProviderOptions>(), 24usize, concat!("Size of: ", stringify!(OrtMIGraphXProviderOptions)));
	assert_eq!(::std::mem::align_of::<OrtMIGraphXProviderOptions>(), 8usize, concat!("Alignment of ", stringify!(OrtMIGraphXProviderOptions)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_id) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtMIGraphXProviderOptions), "::", stringify!(device_id))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).migraphx_fp16_enable) as usize - ptr as usize },
		4usize,
		concat!("Offset of field: ", stringify!(OrtMIGraphXProviderOptions), "::", stringify!(migraphx_fp16_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).migraphx_int8_enable) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtMIGraphXProviderOptions), "::", stringify!(migraphx_int8_enable))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).migraphx_use_native_calibration_table) as usize - ptr as usize },
		12usize,
		concat!("Offset of field: ", stringify!(OrtMIGraphXProviderOptions), "::", stringify!(migraphx_use_native_calibration_table))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).migraphx_int8_calibration_table_name) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtMIGraphXProviderOptions), "::", stringify!(migraphx_int8_calibration_table_name))
	);
}
#[doc = " \\brief OpenVINO Provider Options\n\n \\see OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtOpenVINOProviderOptions {
	#[doc = " \\brief Device type string\n\n Valid settings are one of: \"CPU_FP32\", \"CPU_FP16\", \"GPU_FP32\", \"GPU_FP16\""]
	pub device_type: *const ::std::os::raw::c_char,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_npu_fast_compile: ::std::os::raw::c_uchar,
	pub device_id: *const ::std::os::raw::c_char,
	#[doc = "< 0 = Use default number of threads"]
	pub num_of_threads: size_t,
	pub cache_dir: *const ::std::os::raw::c_char,
	pub context: *mut ::std::os::raw::c_void,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_opencl_throttling: ::std::os::raw::c_uchar,
	#[doc = "< 0 = disabled, nonzero = enabled"]
	pub enable_dynamic_shapes: ::std::os::raw::c_uchar
}
#[test]
fn bindgen_test_layout_OrtOpenVINOProviderOptions() {
	const UNINIT: ::std::mem::MaybeUninit<OrtOpenVINOProviderOptions> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtOpenVINOProviderOptions>(), 56usize, concat!("Size of: ", stringify!(OrtOpenVINOProviderOptions)));
	assert_eq!(::std::mem::align_of::<OrtOpenVINOProviderOptions>(), 8usize, concat!("Alignment of ", stringify!(OrtOpenVINOProviderOptions)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_type) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(device_type))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).enable_npu_fast_compile) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(enable_npu_fast_compile))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).device_id) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(device_id))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).num_of_threads) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(num_of_threads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).cache_dir) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(cache_dir))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).context) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(context))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).enable_opencl_throttling) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(enable_opencl_throttling))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).enable_dynamic_shapes) as usize - ptr as usize },
		49usize,
		concat!("Offset of field: ", stringify!(OrtOpenVINOProviderOptions), "::", stringify!(enable_dynamic_shapes))
	);
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
	pub LoadCheckpoint:
		::std::option::Option<_system!(unsafe fn(checkpoint_path: *const ortchar, checkpoint_state: *mut *mut OrtCheckpointState) -> OrtStatusPtr)>,
	pub SaveCheckpoint: ::std::option::Option<
		_system!(unsafe fn(checkpoint_state: *mut OrtCheckpointState, checkpoint_path: *const ortchar, include_optimizer_state: bool) -> OrtStatusPtr)
	>,
	pub CreateTrainingSession: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *const OrtEnv,
				options: *const OrtSessionOptions,
				checkpoint_state: *mut OrtCheckpointState,
				train_model_path: *const ortchar,
				eval_model_path: *const ortchar,
				optimizer_model_path: *const ortchar,
				out: *mut *mut OrtTrainingSession
			) -> OrtStatusPtr
		)
	>,
	pub CreateTrainingSessionFromBuffer: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *const OrtEnv,
				options: *const OrtSessionOptions,
				checkpoint_state: *mut OrtCheckpointState,
				train_model_data: *const (),
				train_data_length: size_t,
				eval_model_data: *const (),
				eval_data_length: size_t,
				optimizer_model_data: *const (),
				optimizer_data_length: size_t,
				out: *mut *mut OrtTrainingSession
			) -> OrtStatusPtr
		)
	>,
	pub TrainingSessionGetTrainingModelOutputCount:
		::std::option::Option<_system!(unsafe fn(sess: *const OrtTrainingSession, out: *mut size_t) -> OrtStatusPtr)>,
	pub TrainingSessionGetEvalModelOutputCount: ::std::option::Option<_system!(unsafe fn(sess: *const OrtTrainingSession, out: *mut size_t) -> OrtStatusPtr)>,
	pub TrainingSessionGetTrainingModelOutputName: ::std::option::Option<
		_system!(unsafe fn(sess: *const OrtTrainingSession, index: size_t, allocator: *mut OrtAllocator, output: *mut *mut c_char) -> OrtStatusPtr)
	>,
	pub TrainingSessionGetEvalModelOutputName: ::std::option::Option<
		_system!(unsafe fn(sess: *const OrtTrainingSession, index: size_t, allocator: *mut OrtAllocator, output: *mut *mut c_char) -> OrtStatusPtr)
	>,
	pub LazyResetGrad: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession) -> OrtStatusPtr)>,
	pub TrainStep: ::std::option::Option<
		_system!(
			unsafe fn(
				session: *mut OrtTrainingSession,
				run_options: *const OrtRunOptions,
				inputs_len: size_t,
				inputs: *const *const OrtValue,
				outputs_len: size_t,
				outputs: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub EvalStep: ::std::option::Option<
		_system!(
			unsafe fn(
				session: *mut OrtTrainingSession,
				run_options: *const OrtRunOptions,
				inputs_len: size_t,
				inputs: *const *const OrtValue,
				outputs_len: size_t,
				outputs: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub SetLearningRate: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, learning_rate: f32) -> OrtStatusPtr)>,
	pub GetLearningRate: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, learning_rate: *mut f32) -> OrtStatusPtr)>,
	pub OptimizerStep: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, run_options: *const OrtRunOptions) -> OrtStatusPtr)>,
	pub RegisterLinearLRScheduler: ::std::option::Option<
		_system!(unsafe fn(session: *mut OrtTrainingSession, warmup_step_count: i64, total_step_count: i64, initial_lr: f32) -> OrtStatusPtr)
	>,
	pub SchedulerStep: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession) -> OrtStatusPtr)>,
	pub GetParametersSize: ::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, out: *mut size_t, trainable_only: bool) -> OrtStatusPtr)>,
	pub CopyParametersToBuffer:
		::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, parameters_buffer: *mut OrtValue, trainable_only: bool) -> OrtStatusPtr)>,
	pub CopyBufferToParameters:
		::std::option::Option<_system!(unsafe fn(session: *mut OrtTrainingSession, parameters_buffer: *mut OrtValue, trainable_only: bool) -> OrtStatusPtr)>,
	pub ReleaseTrainingSession: ::std::option::Option<_system!(unsafe fn(input: *mut OrtTrainingSession))>,
	pub ReleaseCheckpointState: ::std::option::Option<_system!(unsafe fn(input: *mut OrtCheckpointState))>,
	pub ExportModelForInferencing: ::std::option::Option<
		_system!(
			unsafe fn(
				session: *mut OrtTrainingSession,
				inference_model_path: *const ortchar,
				graph_outputs_len: usize,
				graph_output_names: *const *const c_char
			) -> OrtStatusPtr
		)
	>
}
#[doc = " \\brief The helper interface to get the right version of OrtApi\n\n Get a pointer to this structure through ::OrtGetApiBase"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtApiBase {
	#[doc = " \\brief Get a pointer to the requested version of the ::OrtApi\n\n \\param[in] version Must be ::ORT_API_VERSION\n \\return The ::OrtApi for the version requested, nullptr will be returned if this version is unsupported, for example when using a runtime\n   older than the version created with this header file.\n\n One can call GetVersionString() to get the version of the Onnxruntime library for logging\n and error reporting purposes."]
	pub GetApi: ::std::option::Option<_system!(unsafe fn(version: u32) -> *const OrtApi)>,
	#[doc = " \\brief Returns a null terminated string of the version of the Onnxruntime library (eg: \"1.8.1\")\n\n  \\return UTF-8 encoded version string. Do not deallocate the returned buffer."]
	pub GetVersionString: ::std::option::Option<_system!(unsafe fn() -> *const ::std::os::raw::c_char)>
}
#[test]
fn bindgen_test_layout_OrtApiBase() {
	const UNINIT: ::std::mem::MaybeUninit<OrtApiBase> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtApiBase>(), 16usize, concat!("Size of: ", stringify!(OrtApiBase)));
	assert_eq!(::std::mem::align_of::<OrtApiBase>(), 8usize, concat!("Alignment of ", stringify!(OrtApiBase)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetApi) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtApiBase), "::", stringify!(GetApi))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetVersionString) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtApiBase), "::", stringify!(GetVersionString))
	);
}
_system_block! {
	#[doc = " \\brief The Onnxruntime library's entry point to access the C API\n\n Call this to get the a pointer to an ::OrtApiBase"]
	pub fn OrtGetApiBase() -> *const OrtApiBase;
}
#[doc = " \\brief Thread work loop function\n\n Onnxruntime will provide the working loop on custom thread creation\n Argument is an onnxruntime built-in type which will be provided when thread pool calls OrtCustomCreateThreadFn"]
pub type OrtThreadWorkerFn = ::std::option::Option<_system!(unsafe fn(ort_worker_fn_param: *mut ::std::os::raw::c_void))>;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtCustomHandleType {
	pub __place_holder: ::std::os::raw::c_char
}
#[test]
fn bindgen_test_layout_OrtCustomHandleType() {
	const UNINIT: ::std::mem::MaybeUninit<OrtCustomHandleType> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtCustomHandleType>(), 1usize, concat!("Size of: ", stringify!(OrtCustomHandleType)));
	assert_eq!(::std::mem::align_of::<OrtCustomHandleType>(), 1usize, concat!("Alignment of ", stringify!(OrtCustomHandleType)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).__place_holder) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtCustomHandleType), "::", stringify!(__place_holder))
	);
}
pub type OrtCustomThreadHandle = *const OrtCustomHandleType;
#[doc = " \\brief Ort custom thread creation function\n\n The function should return a thread handle to be used in onnxruntime thread pools\n Onnxruntime will throw exception on return value of nullptr or 0, indicating that the function failed to create a thread"]
pub type OrtCustomCreateThreadFn = ::std::option::Option<
	_system!(
		unsafe fn(
			ort_custom_thread_creation_options: *mut ::std::os::raw::c_void,
			ort_thread_worker_fn: OrtThreadWorkerFn,
			ort_worker_fn_param: *mut ::std::os::raw::c_void
		) -> OrtCustomThreadHandle
	)
>;
#[doc = " \\brief Custom thread join function\n\n Onnxruntime thread pool destructor will call the function to join a custom thread.\n Argument ort_custom_thread_handle is the value returned by OrtCustomCreateThreadFn"]
pub type OrtCustomJoinThreadFn = ::std::option::Option<_system!(unsafe fn(ort_custom_thread_handle: OrtCustomThreadHandle))>;
#[doc = " \\brief Callback function for RunAsync\n\n \\param[in] user_data User specific data that passed back to the callback\n \\param[out] outputs On succeed, outputs host inference results, on error, the value will be nullptr\n \\param[out] num_outputs Number of outputs, on error, the value will be zero\n \\param[out] status On error, status will provide details"]
pub type RunAsyncCallbackFn =
	::std::option::Option<_system!(unsafe fn(user_data: *mut ::std::os::raw::c_void, outputs: *mut *mut OrtValue, num_outputs: size_t, status: OrtStatusPtr))>;
#[doc = " \\brief The C API\n\n All C API functions are defined inside this structure as pointers to functions.\n Call OrtApiBase::GetApi to get a pointer to it\n\n \\nosubgrouping"]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OrtApi {
	#[doc = " \\brief Create an OrtStatus from a null terminated string\n\n \\param[in] code\n \\param[in] msg A null-terminated string. Its contents will be copied.\n \\return A new OrtStatus object, must be destroyed with OrtApi::ReleaseStatus"]
	pub CreateStatus: ::std::option::Option<_system!(unsafe fn(code: OrtErrorCode, msg: *const ::std::os::raw::c_char) -> *mut OrtStatus)>,
	#[doc = " \\brief Get OrtErrorCode from OrtStatus\n\n \\param[in] status\n \\return OrtErrorCode that \\p status was created with"]
	pub GetErrorCode: ::std::option::Option<_system!(unsafe fn(status: *const OrtStatus) -> OrtErrorCode)>,
	#[doc = " \\brief Get error string from OrtStatus\n\n \\param[in] status\n \\return The error message inside the `status`. Do not free the returned value."]
	pub GetErrorMessage: ::std::option::Option<_system!(unsafe fn(status: *const OrtStatus) -> *const ::std::os::raw::c_char)>,
	pub CreateEnv: ::std::option::Option<
		_system!(unsafe fn(log_severity_level: OrtLoggingLevel, logid: *const ::std::os::raw::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr)
	>,
	pub CreateEnvWithCustomLogger: ::std::option::Option<
		_system!(
			unsafe fn(
				logging_function: OrtLoggingFunction,
				logger_param: *mut ::std::os::raw::c_void,
				log_severity_level: OrtLoggingLevel,
				logid: *const ::std::os::raw::c_char,
				out: *mut *mut OrtEnv
			) -> OrtStatusPtr
		)
	>,
	pub EnableTelemetryEvents: ::std::option::Option<_system!(unsafe fn(env: *const OrtEnv) -> OrtStatusPtr)>,
	pub DisableTelemetryEvents: ::std::option::Option<_system!(unsafe fn(env: *const OrtEnv) -> OrtStatusPtr)>,
	pub CreateSession: ::std::option::Option<
		_system!(unsafe fn(env: *const OrtEnv, model_path: *const ortchar, options: *const OrtSessionOptions, out: *mut *mut OrtSession) -> OrtStatusPtr)
	>,
	pub CreateSessionFromArray: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *const OrtEnv,
				model_data: *const ::std::os::raw::c_void,
				model_data_length: size_t,
				options: *const OrtSessionOptions,
				out: *mut *mut OrtSession
			) -> OrtStatusPtr
		)
	>,
	pub Run: ::std::option::Option<
		_system!(
			unsafe fn(
				session: *mut OrtSession,
				run_options: *const OrtRunOptions,
				input_names: *const *const ::std::os::raw::c_char,
				inputs: *const *const OrtValue,
				input_len: size_t,
				output_names: *const *const ::std::os::raw::c_char,
				output_names_len: size_t,
				outputs: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub CreateSessionOptions: ::std::option::Option<_system!(unsafe fn(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub SetOptimizedModelFilePath:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, optimized_model_filepath: *const ortchar) -> OrtStatusPtr)>,
	pub CloneSessionOptions:
		::std::option::Option<_system!(unsafe fn(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub SetSessionExecutionMode: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr)>,
	pub EnableProfiling: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, profile_file_prefix: *const ortchar) -> OrtStatusPtr)>,
	pub DisableProfiling: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub EnableMemPattern: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub DisableMemPattern: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub EnableCpuMemArena: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub DisableCpuMemArena: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub SetSessionLogId: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, logid: *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub SetSessionLogVerbosityLevel:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, session_log_verbosity_level: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub SetSessionLogSeverityLevel:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, session_log_severity_level: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub SetSessionGraphOptimizationLevel:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr)>,
	pub SetIntraOpNumThreads:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, intra_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub SetInterOpNumThreads:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, inter_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub CreateCustomOpDomain:
		::std::option::Option<_system!(unsafe fn(domain: *const ::std::os::raw::c_char, out: *mut *mut OrtCustomOpDomain) -> OrtStatusPtr)>,
	pub CustomOpDomain_Add: ::std::option::Option<_system!(unsafe fn(custom_op_domain: *mut OrtCustomOpDomain, op: *const OrtCustomOp) -> OrtStatusPtr)>,
	pub AddCustomOpDomain:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, custom_op_domain: *mut OrtCustomOpDomain) -> OrtStatusPtr)>,
	pub RegisterCustomOpsLibrary: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				library_path: *const ::std::os::raw::c_char,
				library_handle: *mut *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub SessionGetInputCount: ::std::option::Option<_system!(unsafe fn(session: *const OrtSession, out: *mut size_t) -> OrtStatusPtr)>,
	pub SessionGetOutputCount: ::std::option::Option<_system!(unsafe fn(session: *const OrtSession, out: *mut size_t) -> OrtStatusPtr)>,
	pub SessionGetOverridableInitializerCount: ::std::option::Option<_system!(unsafe fn(session: *const OrtSession, out: *mut size_t) -> OrtStatusPtr)>,
	pub SessionGetInputTypeInfo:
		::std::option::Option<_system!(unsafe fn(session: *const OrtSession, index: size_t, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub SessionGetOutputTypeInfo:
		::std::option::Option<_system!(unsafe fn(session: *const OrtSession, index: size_t, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub SessionGetOverridableInitializerTypeInfo:
		::std::option::Option<_system!(unsafe fn(session: *const OrtSession, index: size_t, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub SessionGetInputName: ::std::option::Option<
		_system!(unsafe fn(session: *const OrtSession, index: size_t, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub SessionGetOutputName: ::std::option::Option<
		_system!(unsafe fn(session: *const OrtSession, index: size_t, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub SessionGetOverridableInitializerName: ::std::option::Option<
		_system!(unsafe fn(session: *const OrtSession, index: size_t, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub CreateRunOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtRunOptions) -> OrtStatusPtr)>,
	pub RunOptionsSetRunLogVerbosityLevel:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtRunOptions, log_verbosity_level: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub RunOptionsSetRunLogSeverityLevel:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtRunOptions, log_severity_level: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub RunOptionsSetRunTag: ::std::option::Option<_system!(unsafe fn(options: *mut OrtRunOptions, run_tag: *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub RunOptionsGetRunLogVerbosityLevel:
		::std::option::Option<_system!(unsafe fn(options: *const OrtRunOptions, log_verbosity_level: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub RunOptionsGetRunLogSeverityLevel:
		::std::option::Option<_system!(unsafe fn(options: *const OrtRunOptions, log_severity_level: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub RunOptionsGetRunTag:
		::std::option::Option<_system!(unsafe fn(options: *const OrtRunOptions, run_tag: *mut *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub RunOptionsSetTerminate: ::std::option::Option<_system!(unsafe fn(options: *mut OrtRunOptions) -> OrtStatusPtr)>,
	pub RunOptionsUnsetTerminate: ::std::option::Option<_system!(unsafe fn(options: *mut OrtRunOptions) -> OrtStatusPtr)>,
	pub CreateTensorAsOrtValue: ::std::option::Option<
		_system!(
			unsafe fn(
				allocator: *mut OrtAllocator,
				shape: *const i64,
				shape_len: size_t,
				type_: ONNXTensorElementDataType,
				out: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub CreateTensorWithDataAsOrtValue: ::std::option::Option<
		_system!(
			unsafe fn(
				info: *const OrtMemoryInfo,
				p_data: *mut ::std::os::raw::c_void,
				p_data_len: size_t,
				shape: *const i64,
				shape_len: size_t,
				type_: ONNXTensorElementDataType,
				out: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub IsTensor: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub GetTensorMutableData: ::std::option::Option<_system!(unsafe fn(value: *mut OrtValue, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub FillStringTensor:
		::std::option::Option<_system!(unsafe fn(value: *mut OrtValue, s: *const *const ::std::os::raw::c_char, s_len: size_t) -> OrtStatusPtr)>,
	pub GetStringTensorDataLength: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, len: *mut size_t) -> OrtStatusPtr)>,
	pub GetStringTensorContent: ::std::option::Option<
		_system!(unsafe fn(value: *const OrtValue, s: *mut ::std::os::raw::c_void, s_len: size_t, offsets: *mut size_t, offsets_len: size_t) -> OrtStatusPtr)
	>,
	pub CastTypeInfoToTensorInfo:
		::std::option::Option<_system!(unsafe fn(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)>,
	pub GetOnnxTypeFromTypeInfo: ::std::option::Option<_system!(unsafe fn(type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr)>,
	pub CreateTensorTypeAndShapeInfo: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)>,
	pub SetTensorElementType:
		::std::option::Option<_system!(unsafe fn(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr)>,
	pub SetDimensions:
		::std::option::Option<_system!(unsafe fn(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: size_t) -> OrtStatusPtr)>,
	pub GetTensorElementType:
		::std::option::Option<_system!(unsafe fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr)>,
	pub GetDimensionsCount: ::std::option::Option<_system!(unsafe fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut size_t) -> OrtStatusPtr)>,
	pub GetDimensions:
		::std::option::Option<_system!(unsafe fn(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: size_t) -> OrtStatusPtr)>,
	pub GetSymbolicDimensions: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtTensorTypeAndShapeInfo, dim_params: *mut *const ::std::os::raw::c_char, dim_params_length: size_t) -> OrtStatusPtr)
	>,
	pub GetTensorShapeElementCount: ::std::option::Option<_system!(unsafe fn(info: *const OrtTensorTypeAndShapeInfo, out: *mut size_t) -> OrtStatusPtr)>,
	pub GetTensorTypeAndShape: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)>,
	pub GetTypeInfo: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub GetValueType: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr)>,
	pub CreateMemoryInfo: ::std::option::Option<
		_system!(
			unsafe fn(
				name: *const ::std::os::raw::c_char,
				type_: OrtAllocatorType,
				id: ::std::os::raw::c_int,
				mem_type: OrtMemType,
				out: *mut *mut OrtMemoryInfo
			) -> OrtStatusPtr
		)
	>,
	pub CreateCpuMemoryInfo:
		::std::option::Option<_system!(unsafe fn(type_: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr)>,
	pub CompareMemoryInfo:
		::std::option::Option<_system!(unsafe fn(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub MemoryInfoGetName: ::std::option::Option<_system!(unsafe fn(ptr: *const OrtMemoryInfo, out: *mut *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub MemoryInfoGetId: ::std::option::Option<_system!(unsafe fn(ptr: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub MemoryInfoGetMemType: ::std::option::Option<_system!(unsafe fn(ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr)>,
	pub MemoryInfoGetType: ::std::option::Option<_system!(unsafe fn(ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr)>,
	pub AllocatorAlloc:
		::std::option::Option<_system!(unsafe fn(ort_allocator: *mut OrtAllocator, size: size_t, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub AllocatorFree: ::std::option::Option<_system!(unsafe fn(ort_allocator: *mut OrtAllocator, p: *mut ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub AllocatorGetInfo: ::std::option::Option<_system!(unsafe fn(ort_allocator: *const OrtAllocator, out: *mut *const OrtMemoryInfo) -> OrtStatusPtr)>,
	pub GetAllocatorWithDefaultOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtAllocator) -> OrtStatusPtr)>,
	pub AddFreeDimensionOverride: ::std::option::Option<
		_system!(unsafe fn(options: *mut OrtSessionOptions, dim_denotation: *const ::std::os::raw::c_char, dim_value: i64) -> OrtStatusPtr)
	>,
	pub GetValue: ::std::option::Option<
		_system!(unsafe fn(value: *const OrtValue, index: ::std::os::raw::c_int, allocator: *mut OrtAllocator, out: *mut *mut OrtValue) -> OrtStatusPtr)
	>,
	pub GetValueCount: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut size_t) -> OrtStatusPtr)>,
	pub CreateValue: ::std::option::Option<
		_system!(unsafe fn(in_: *const *const OrtValue, num_values: size_t, value_type: ONNXType, out: *mut *mut OrtValue) -> OrtStatusPtr)
	>,
	pub CreateOpaqueValue: ::std::option::Option<
		_system!(
			unsafe fn(
				domain_name: *const ::std::os::raw::c_char,
				type_name: *const ::std::os::raw::c_char,
				data_container: *const ::std::os::raw::c_void,
				data_container_size: size_t,
				out: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub GetOpaqueValue: ::std::option::Option<
		_system!(
			unsafe fn(
				domain_name: *const ::std::os::raw::c_char,
				type_name: *const ::std::os::raw::c_char,
				in_: *const OrtValue,
				data_container: *mut ::std::os::raw::c_void,
				data_container_size: size_t
			) -> OrtStatusPtr
		)
	>,
	pub KernelInfoGetAttribute_float:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut f32) -> OrtStatusPtr)>,
	pub KernelInfoGetAttribute_int64:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut i64) -> OrtStatusPtr)>,
	pub KernelInfoGetAttribute_string: ::std::option::Option<
		_system!(
			unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut ::std::os::raw::c_char, size: *mut size_t) -> OrtStatusPtr
		)
	>,
	pub KernelContext_GetInputCount: ::std::option::Option<_system!(unsafe fn(context: *const OrtKernelContext, out: *mut size_t) -> OrtStatusPtr)>,
	pub KernelContext_GetOutputCount: ::std::option::Option<_system!(unsafe fn(context: *const OrtKernelContext, out: *mut size_t) -> OrtStatusPtr)>,
	pub KernelContext_GetInput:
		::std::option::Option<_system!(unsafe fn(context: *const OrtKernelContext, index: size_t, out: *mut *const OrtValue) -> OrtStatusPtr)>,
	pub KernelContext_GetOutput: ::std::option::Option<
		_system!(unsafe fn(context: *mut OrtKernelContext, index: size_t, dim_values: *const i64, dim_count: size_t, out: *mut *mut OrtValue) -> OrtStatusPtr)
	>,
	pub ReleaseEnv: ::std::option::Option<_system!(unsafe fn(input: *mut OrtEnv))>,
	pub ReleaseStatus: ::std::option::Option<_system!(unsafe fn(input: *mut OrtStatus))>,
	pub ReleaseMemoryInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtMemoryInfo))>,
	pub ReleaseSession: ::std::option::Option<_system!(unsafe fn(input: *mut OrtSession))>,
	pub ReleaseValue: ::std::option::Option<_system!(unsafe fn(input: *mut OrtValue))>,
	pub ReleaseRunOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtRunOptions))>,
	pub ReleaseTypeInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtTypeInfo))>,
	pub ReleaseTensorTypeAndShapeInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtTensorTypeAndShapeInfo))>,
	pub ReleaseSessionOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtSessionOptions))>,
	pub ReleaseCustomOpDomain: ::std::option::Option<_system!(unsafe fn(input: *mut OrtCustomOpDomain))>,
	pub GetDenotationFromTypeInfo: ::std::option::Option<
		_system!(unsafe fn(type_info: *const OrtTypeInfo, denotation: *mut *const ::std::os::raw::c_char, len: *mut size_t) -> OrtStatusPtr)
	>,
	pub CastTypeInfoToMapTypeInfo: ::std::option::Option<_system!(unsafe fn(type_info: *const OrtTypeInfo, out: *mut *const OrtMapTypeInfo) -> OrtStatusPtr)>,
	pub CastTypeInfoToSequenceTypeInfo:
		::std::option::Option<_system!(unsafe fn(type_info: *const OrtTypeInfo, out: *mut *const OrtSequenceTypeInfo) -> OrtStatusPtr)>,
	pub GetMapKeyType: ::std::option::Option<_system!(unsafe fn(map_type_info: *const OrtMapTypeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr)>,
	pub GetMapValueType: ::std::option::Option<_system!(unsafe fn(map_type_info: *const OrtMapTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub GetSequenceElementType:
		::std::option::Option<_system!(unsafe fn(sequence_type_info: *const OrtSequenceTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub ReleaseMapTypeInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtMapTypeInfo))>,
	pub ReleaseSequenceTypeInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtSequenceTypeInfo))>,
	pub SessionEndProfiling: ::std::option::Option<
		_system!(unsafe fn(session: *mut OrtSession, allocator: *mut OrtAllocator, out: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub SessionGetModelMetadata: ::std::option::Option<_system!(unsafe fn(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr)>,
	pub ModelMetadataGetProducerName: ::std::option::Option<
		_system!(unsafe fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub ModelMetadataGetGraphName: ::std::option::Option<
		_system!(unsafe fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub ModelMetadataGetDomain: ::std::option::Option<
		_system!(unsafe fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub ModelMetadataGetDescription: ::std::option::Option<
		_system!(unsafe fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub ModelMetadataLookupCustomMetadataMap: ::std::option::Option<
		_system!(
			unsafe fn(
				model_metadata: *const OrtModelMetadata,
				allocator: *mut OrtAllocator,
				key: *const ::std::os::raw::c_char,
				value: *mut *mut ::std::os::raw::c_char
			) -> OrtStatusPtr
		)
	>,
	pub ModelMetadataGetVersion: ::std::option::Option<_system!(unsafe fn(model_metadata: *const OrtModelMetadata, value: *mut i64) -> OrtStatusPtr)>,
	pub ReleaseModelMetadata: ::std::option::Option<_system!(unsafe fn(input: *mut OrtModelMetadata))>,
	pub CreateEnvWithGlobalThreadPools: ::std::option::Option<
		_system!(
			unsafe fn(
				log_severity_level: OrtLoggingLevel,
				logid: *const ::std::os::raw::c_char,
				tp_options: *const OrtThreadingOptions,
				out: *mut *mut OrtEnv
			) -> OrtStatusPtr
		)
	>,
	pub DisablePerSessionThreads: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub CreateThreadingOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtThreadingOptions) -> OrtStatusPtr)>,
	pub ReleaseThreadingOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtThreadingOptions))>,
	pub ModelMetadataGetCustomMetadataMapKeys: ::std::option::Option<
		_system!(
			unsafe fn(
				model_metadata: *const OrtModelMetadata,
				allocator: *mut OrtAllocator,
				keys: *mut *mut *mut ::std::os::raw::c_char,
				num_keys: *mut i64
			) -> OrtStatusPtr
		)
	>,
	pub AddFreeDimensionOverrideByName:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, dim_name: *const ::std::os::raw::c_char, dim_value: i64) -> OrtStatusPtr)>,
	pub GetAvailableProviders:
		::std::option::Option<_system!(unsafe fn(out_ptr: *mut *mut *mut ::std::os::raw::c_char, provider_length: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub ReleaseAvailableProviders:
		::std::option::Option<_system!(unsafe fn(ptr: *mut *mut ::std::os::raw::c_char, providers_length: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub GetStringTensorElementLength: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, index: size_t, out: *mut size_t) -> OrtStatusPtr)>,
	pub GetStringTensorElement:
		::std::option::Option<_system!(unsafe fn(value: *const OrtValue, s_len: size_t, index: size_t, s: *mut ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub FillStringTensorElement:
		::std::option::Option<_system!(unsafe fn(value: *mut OrtValue, s: *const ::std::os::raw::c_char, index: size_t) -> OrtStatusPtr)>,
	pub AddSessionConfigEntry: ::std::option::Option<
		_system!(
			unsafe fn(options: *mut OrtSessionOptions, config_key: *const ::std::os::raw::c_char, config_value: *const ::std::os::raw::c_char) -> OrtStatusPtr
		)
	>,
	pub CreateAllocator:
		::std::option::Option<_system!(unsafe fn(session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr)>,
	pub ReleaseAllocator: ::std::option::Option<_system!(unsafe fn(input: *mut OrtAllocator))>,
	pub RunWithBinding: ::std::option::Option<
		_system!(unsafe fn(session: *mut OrtSession, run_options: *const OrtRunOptions, binding_ptr: *const OrtIoBinding) -> OrtStatusPtr)
	>,
	pub CreateIoBinding: ::std::option::Option<_system!(unsafe fn(session: *mut OrtSession, out: *mut *mut OrtIoBinding) -> OrtStatusPtr)>,
	pub ReleaseIoBinding: ::std::option::Option<_system!(unsafe fn(input: *mut OrtIoBinding))>,
	pub BindInput: ::std::option::Option<
		_system!(unsafe fn(binding_ptr: *mut OrtIoBinding, name: *const ::std::os::raw::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr)
	>,
	pub BindOutput: ::std::option::Option<
		_system!(unsafe fn(binding_ptr: *mut OrtIoBinding, name: *const ::std::os::raw::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr)
	>,
	pub BindOutputToDevice: ::std::option::Option<
		_system!(unsafe fn(binding_ptr: *mut OrtIoBinding, name: *const ::std::os::raw::c_char, mem_info_ptr: *const OrtMemoryInfo) -> OrtStatusPtr)
	>,
	pub GetBoundOutputNames: ::std::option::Option<
		_system!(
			unsafe fn(
				binding_ptr: *const OrtIoBinding,
				allocator: *mut OrtAllocator,
				buffer: *mut *mut ::std::os::raw::c_char,
				lengths: *mut *mut size_t,
				count: *mut size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetBoundOutputValues: ::std::option::Option<
		_system!(
			unsafe fn(
				binding_ptr: *const OrtIoBinding,
				allocator: *mut OrtAllocator,
				output: *mut *mut *mut OrtValue,
				output_count: *mut size_t
			) -> OrtStatusPtr
		)
	>,
	#[doc = " \\brief Clears any previously set Inputs for an ::OrtIoBinding"]
	pub ClearBoundInputs: ::std::option::Option<_system!(unsafe fn(binding_ptr: *mut OrtIoBinding))>,
	#[doc = " \\brief Clears any previously set Outputs for an ::OrtIoBinding"]
	pub ClearBoundOutputs: ::std::option::Option<_system!(unsafe fn(binding_ptr: *mut OrtIoBinding))>,
	pub TensorAt: ::std::option::Option<
		_system!(
			unsafe fn(value: *mut OrtValue, location_values: *const i64, location_values_count: size_t, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr
		)
	>,
	pub CreateAndRegisterAllocator:
		::std::option::Option<_system!(unsafe fn(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo, arena_cfg: *const OrtArenaCfg) -> OrtStatusPtr)>,
	pub SetLanguageProjection: ::std::option::Option<_system!(unsafe fn(ort_env: *const OrtEnv, projection: OrtLanguageProjection) -> OrtStatusPtr)>,
	pub SessionGetProfilingStartTimeNs: ::std::option::Option<_system!(unsafe fn(session: *const OrtSession, out: *mut u64) -> OrtStatusPtr)>,
	pub SetGlobalIntraOpNumThreads:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, intra_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub SetGlobalInterOpNumThreads:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, inter_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub SetGlobalSpinControl:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, allow_spinning: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub AddInitializer:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, name: *const ::std::os::raw::c_char, val: *const OrtValue) -> OrtStatusPtr)>,
	pub CreateEnvWithCustomLoggerAndGlobalThreadPools: ::std::option::Option<
		_system!(
			unsafe fn(
				logging_function: OrtLoggingFunction,
				logger_param: *mut ::std::os::raw::c_void,
				log_severity_level: OrtLoggingLevel,
				logid: *const ::std::os::raw::c_char,
				tp_options: *const OrtThreadingOptions,
				out: *mut *mut OrtEnv
			) -> OrtStatusPtr
		)
	>,
	pub SessionOptionsAppendExecutionProvider_CUDA:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, cuda_options: *const OrtCUDAProviderOptions) -> OrtStatusPtr)>,
	pub SessionOptionsAppendExecutionProvider_ROCM:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, rocm_options: *const OrtROCMProviderOptions) -> OrtStatusPtr)>,
	pub SessionOptionsAppendExecutionProvider_OpenVINO:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, provider_options: *const OrtOpenVINOProviderOptions) -> OrtStatusPtr)>,
	pub SetGlobalDenormalAsZero: ::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions) -> OrtStatusPtr)>,
	pub CreateArenaCfg: ::std::option::Option<
		_system!(
			unsafe fn(
				max_mem: size_t,
				arena_extend_strategy: ::std::os::raw::c_int,
				initial_chunk_size_bytes: ::std::os::raw::c_int,
				max_dead_bytes_per_chunk: ::std::os::raw::c_int,
				out: *mut *mut OrtArenaCfg
			) -> OrtStatusPtr
		)
	>,
	pub ReleaseArenaCfg: ::std::option::Option<_system!(unsafe fn(input: *mut OrtArenaCfg))>,
	pub ModelMetadataGetGraphDescription: ::std::option::Option<
		_system!(unsafe fn(model_metadata: *const OrtModelMetadata, allocator: *mut OrtAllocator, value: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub SessionOptionsAppendExecutionProvider_TensorRT:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, tensorrt_options: *const OrtTensorRTProviderOptions) -> OrtStatusPtr)>,
	pub SetCurrentGpuDeviceId: ::std::option::Option<_system!(unsafe fn(device_id: ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub GetCurrentGpuDeviceId: ::std::option::Option<_system!(unsafe fn(device_id: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub KernelInfoGetAttributeArray_float: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut f32, size: *mut size_t) -> OrtStatusPtr)
	>,
	pub KernelInfoGetAttributeArray_int64: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut i64, size: *mut size_t) -> OrtStatusPtr)
	>,
	pub CreateArenaCfgV2: ::std::option::Option<
		_system!(
			unsafe fn(
				arena_config_keys: *const *const ::std::os::raw::c_char,
				arena_config_values: *const size_t,
				num_keys: size_t,
				out: *mut *mut OrtArenaCfg
			) -> OrtStatusPtr
		)
	>,
	pub AddRunConfigEntry: ::std::option::Option<
		_system!(
			unsafe fn(options: *mut OrtRunOptions, config_key: *const ::std::os::raw::c_char, config_value: *const ::std::os::raw::c_char) -> OrtStatusPtr
		)
	>,
	pub CreatePrepackedWeightsContainer: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtPrepackedWeightsContainer) -> OrtStatusPtr)>,
	pub ReleasePrepackedWeightsContainer: ::std::option::Option<_system!(unsafe fn(input: *mut OrtPrepackedWeightsContainer))>,
	pub CreateSessionWithPrepackedWeightsContainer: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *const OrtEnv,
				model_path: *const ortchar,
				options: *const OrtSessionOptions,
				prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
				out: *mut *mut OrtSession
			) -> OrtStatusPtr
		)
	>,
	pub CreateSessionFromArrayWithPrepackedWeightsContainer: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *const OrtEnv,
				model_data: *const ::std::os::raw::c_void,
				model_data_length: size_t,
				options: *const OrtSessionOptions,
				prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
				out: *mut *mut OrtSession
			) -> OrtStatusPtr
		)
	>,
	pub SessionOptionsAppendExecutionProvider_TensorRT_V2:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, tensorrt_options: *const OrtTensorRTProviderOptionsV2) -> OrtStatusPtr)>,
	pub CreateTensorRTProviderOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtTensorRTProviderOptionsV2) -> OrtStatusPtr)>,
	pub UpdateTensorRTProviderOptions: ::std::option::Option<
		_system!(
			unsafe fn(
				tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetTensorRTProviderOptionsAsString: ::std::option::Option<
		_system!(
			unsafe fn(
				tensorrt_options: *const OrtTensorRTProviderOptionsV2,
				allocator: *mut OrtAllocator,
				ptr: *mut *mut ::std::os::raw::c_char
			) -> OrtStatusPtr
		)
	>,
	#[doc = " \\brief Release an ::OrtTensorRTProviderOptionsV2\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does"]
	pub ReleaseTensorRTProviderOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtTensorRTProviderOptionsV2))>,
	pub EnableOrtCustomOps: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions) -> OrtStatusPtr)>,
	pub RegisterAllocator: ::std::option::Option<_system!(unsafe fn(env: *mut OrtEnv, allocator: *mut OrtAllocator) -> OrtStatusPtr)>,
	pub UnregisterAllocator: ::std::option::Option<_system!(unsafe fn(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo) -> OrtStatusPtr)>,
	pub IsSparseTensor: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub CreateSparseTensorAsOrtValue: ::std::option::Option<
		_system!(
			unsafe fn(
				allocator: *mut OrtAllocator,
				dense_shape: *const i64,
				dense_shape_len: size_t,
				type_: ONNXTensorElementDataType,
				out: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub FillSparseTensorCoo: ::std::option::Option<
		_system!(
			unsafe fn(
				ort_value: *mut OrtValue,
				data_mem_info: *const OrtMemoryInfo,
				values_shape: *const i64,
				values_shape_len: size_t,
				values: *const ::std::os::raw::c_void,
				indices_data: *const i64,
				indices_num: size_t
			) -> OrtStatusPtr
		)
	>,
	pub FillSparseTensorCsr: ::std::option::Option<
		_system!(
			unsafe fn(
				ort_value: *mut OrtValue,
				data_mem_info: *const OrtMemoryInfo,
				values_shape: *const i64,
				values_shape_len: size_t,
				values: *const ::std::os::raw::c_void,
				inner_indices_data: *const i64,
				inner_indices_num: size_t,
				outer_indices_data: *const i64,
				outer_indices_num: size_t
			) -> OrtStatusPtr
		)
	>,
	pub FillSparseTensorBlockSparse: ::std::option::Option<
		_system!(
			unsafe fn(
				ort_value: *mut OrtValue,
				data_mem_info: *const OrtMemoryInfo,
				values_shape: *const i64,
				values_shape_len: size_t,
				values: *const ::std::os::raw::c_void,
				indices_shape_data: *const i64,
				indices_shape_len: size_t,
				indices_data: *const i32
			) -> OrtStatusPtr
		)
	>,
	pub CreateSparseTensorWithValuesAsOrtValue: ::std::option::Option<
		_system!(
			unsafe fn(
				info: *const OrtMemoryInfo,
				p_data: *mut ::std::os::raw::c_void,
				dense_shape: *const i64,
				dense_shape_len: size_t,
				values_shape: *const i64,
				values_shape_len: size_t,
				type_: ONNXTensorElementDataType,
				out: *mut *mut OrtValue
			) -> OrtStatusPtr
		)
	>,
	pub UseCooIndices: ::std::option::Option<_system!(unsafe fn(ort_value: *mut OrtValue, indices_data: *mut i64, indices_num: size_t) -> OrtStatusPtr)>,
	pub UseCsrIndices: ::std::option::Option<
		_system!(unsafe fn(ort_value: *mut OrtValue, inner_data: *mut i64, inner_num: size_t, outer_data: *mut i64, outer_num: size_t) -> OrtStatusPtr)
	>,
	pub UseBlockSparseIndices: ::std::option::Option<
		_system!(unsafe fn(ort_value: *mut OrtValue, indices_shape: *const i64, indices_shape_len: size_t, indices_data: *mut i32) -> OrtStatusPtr)
	>,
	pub GetSparseTensorFormat: ::std::option::Option<_system!(unsafe fn(ort_value: *const OrtValue, out: *mut OrtSparseFormat) -> OrtStatusPtr)>,
	pub GetSparseTensorValuesTypeAndShape:
		::std::option::Option<_system!(unsafe fn(ort_value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)>,
	pub GetSparseTensorValues: ::std::option::Option<_system!(unsafe fn(ort_value: *const OrtValue, out: *mut *const ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub GetSparseTensorIndicesTypeShape: ::std::option::Option<
		_system!(unsafe fn(ort_value: *const OrtValue, indices_format: OrtSparseIndicesFormat, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)
	>,
	pub GetSparseTensorIndices: ::std::option::Option<
		_system!(
			unsafe fn(
				ort_value: *const OrtValue,
				indices_format: OrtSparseIndicesFormat,
				num_indices: *mut size_t,
				indices: *mut *const ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub HasValue: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)>,
	pub KernelContext_GetGPUComputeStream:
		::std::option::Option<_system!(unsafe fn(context: *const OrtKernelContext, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr)>,
	pub GetTensorMemoryInfo: ::std::option::Option<_system!(unsafe fn(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr)>,
	pub GetExecutionProviderApi: ::std::option::Option<
		_system!(unsafe fn(provider_name: *const ::std::os::raw::c_char, version: u32, provider_api: *mut *const ::std::os::raw::c_void) -> OrtStatusPtr)
	>,
	pub SessionOptionsSetCustomCreateThreadFn:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr)>,
	pub SessionOptionsSetCustomThreadCreationOptions: ::std::option::Option<
		_system!(unsafe fn(options: *mut OrtSessionOptions, ort_custom_thread_creation_options: *mut ::std::os::raw::c_void) -> OrtStatusPtr)
	>,
	pub SessionOptionsSetCustomJoinThreadFn:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr)>,
	pub SetGlobalCustomCreateThreadFn:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr)>,
	pub SetGlobalCustomThreadCreationOptions: ::std::option::Option<
		_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, ort_custom_thread_creation_options: *mut ::std::os::raw::c_void) -> OrtStatusPtr)
	>,
	pub SetGlobalCustomJoinThreadFn:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr)>,
	pub SynchronizeBoundInputs: ::std::option::Option<_system!(unsafe fn(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr)>,
	pub SynchronizeBoundOutputs: ::std::option::Option<_system!(unsafe fn(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr)>,
	pub SessionOptionsAppendExecutionProvider_CUDA_V2:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, cuda_options: *const OrtCUDAProviderOptionsV2) -> OrtStatusPtr)>,
	pub CreateCUDAProviderOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtCUDAProviderOptionsV2) -> OrtStatusPtr)>,
	pub UpdateCUDAProviderOptions: ::std::option::Option<
		_system!(
			unsafe fn(
				cuda_options: *mut OrtCUDAProviderOptionsV2,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetCUDAProviderOptionsAsString: ::std::option::Option<
		_system!(unsafe fn(cuda_options: *const OrtCUDAProviderOptionsV2, allocator: *mut OrtAllocator, ptr: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	#[doc = " \\brief Release an ::OrtCUDAProviderOptionsV2\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does\n\n \\since Version 1.11."]
	pub ReleaseCUDAProviderOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtCUDAProviderOptionsV2))>,
	pub SessionOptionsAppendExecutionProvider_MIGraphX:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, migraphx_options: *const OrtMIGraphXProviderOptions) -> OrtStatusPtr)>,
	pub AddExternalInitializers: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				initializer_names: *const *const ::std::os::raw::c_char,
				initializers: *const *const OrtValue,
				initializers_num: size_t
			) -> OrtStatusPtr
		)
	>,
	pub CreateOpAttr: ::std::option::Option<
		_system!(
			unsafe fn(
				name: *const ::std::os::raw::c_char,
				data: *const ::std::os::raw::c_void,
				len: ::std::os::raw::c_int,
				type_: OrtOpAttrType,
				op_attr: *mut *mut OrtOpAttr
			) -> OrtStatusPtr
		)
	>,
	pub ReleaseOpAttr: ::std::option::Option<_system!(unsafe fn(input: *mut OrtOpAttr))>,
	pub CreateOp: ::std::option::Option<
		_system!(
			unsafe fn(
				info: *const OrtKernelInfo,
				op_name: *const ::std::os::raw::c_char,
				domain: *const ::std::os::raw::c_char,
				version: ::std::os::raw::c_int,
				type_constraint_names: *mut *const ::std::os::raw::c_char,
				type_constraint_values: *const ONNXTensorElementDataType,
				type_constraint_count: ::std::os::raw::c_int,
				attr_values: *const *const OrtOpAttr,
				attr_count: ::std::os::raw::c_int,
				input_count: ::std::os::raw::c_int,
				output_count: ::std::os::raw::c_int,
				ort_op: *mut *mut OrtOp
			) -> OrtStatusPtr
		)
	>,
	pub InvokeOp: ::std::option::Option<
		_system!(
			unsafe fn(
				context: *const OrtKernelContext,
				ort_op: *const OrtOp,
				input_values: *const *const OrtValue,
				input_count: ::std::os::raw::c_int,
				output_values: *const *mut OrtValue,
				output_count: ::std::os::raw::c_int
			) -> OrtStatusPtr
		)
	>,
	pub ReleaseOp: ::std::option::Option<_system!(unsafe fn(input: *mut OrtOp))>,
	pub SessionOptionsAppendExecutionProvider: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				provider_name: *const ::std::os::raw::c_char,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub CopyKernelInfo: ::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, info_copy: *mut *mut OrtKernelInfo) -> OrtStatusPtr)>,
	pub ReleaseKernelInfo: ::std::option::Option<_system!(unsafe fn(input: *mut OrtKernelInfo))>,
	#[doc = " \\name Ort Training\n @{\n** \\brief Gets the Training C Api struct\n*\n* Call this function to access the ::OrtTrainingApi structure that holds pointers to functions that enable\n* training with onnxruntime.\n* \\note A NULL pointer will be returned and no error message will be printed if the training api\n* is not supported with this build. A NULL pointer will be returned and an error message will be\n* printed if the provided version is unsupported, for example when using a runtime older than the\n* version created with this header file.\n*\n* \\param[in] version Must be ::ORT_API_VERSION\n* \\return The ::OrtTrainingApi struct for the version requested.\n*\n* \\since Version 1.13\n*/"]
	pub GetTrainingApi: ::std::option::Option<_system!(unsafe fn(version: u32) -> *const OrtTrainingApi)>,
	pub SessionOptionsAppendExecutionProvider_CANN:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, cann_options: *const OrtCANNProviderOptions) -> OrtStatusPtr)>,
	pub CreateCANNProviderOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtCANNProviderOptions) -> OrtStatusPtr)>,
	pub UpdateCANNProviderOptions: ::std::option::Option<
		_system!(
			unsafe fn(
				cann_options: *mut OrtCANNProviderOptions,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetCANNProviderOptionsAsString: ::std::option::Option<
		_system!(unsafe fn(cann_options: *const OrtCANNProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	#[doc = " \\brief Release an OrtCANNProviderOptions\n\n \\param[in] the pointer of OrtCANNProviderOptions which will been deleted\n\n \\since Version 1.13."]
	pub ReleaseCANNProviderOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtCANNProviderOptions))>,
	pub MemoryInfoGetDeviceType: ::std::option::Option<_system!(unsafe fn(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType))>,
	pub UpdateEnvWithCustomLogLevel: ::std::option::Option<_system!(unsafe fn(ort_env: *mut OrtEnv, log_severity_level: OrtLoggingLevel) -> OrtStatusPtr)>,
	pub SetGlobalIntraOpThreadAffinity:
		::std::option::Option<_system!(unsafe fn(tp_options: *mut OrtThreadingOptions, affinity_string: *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub RegisterCustomOpsLibrary_V2: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, library_name: *const ortchar) -> OrtStatusPtr)>,
	pub RegisterCustomOpsUsingFunction:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, registration_func_name: *const ::std::os::raw::c_char) -> OrtStatusPtr)>,
	pub KernelInfo_GetInputCount: ::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, out: *mut size_t) -> OrtStatusPtr)>,
	pub KernelInfo_GetOutputCount: ::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, out: *mut size_t) -> OrtStatusPtr)>,
	pub KernelInfo_GetInputName: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtKernelInfo, index: size_t, out: *mut ::std::os::raw::c_char, size: *mut size_t) -> OrtStatusPtr)
	>,
	pub KernelInfo_GetOutputName: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtKernelInfo, index: size_t, out: *mut ::std::os::raw::c_char, size: *mut size_t) -> OrtStatusPtr)
	>,
	pub KernelInfo_GetInputTypeInfo:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, index: size_t, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub KernelInfo_GetOutputTypeInfo:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, index: size_t, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub KernelInfoGetAttribute_tensor: ::std::option::Option<
		_system!(
			unsafe fn(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, allocator: *mut OrtAllocator, out: *mut *mut OrtValue) -> OrtStatusPtr
		)
	>,
	pub HasSessionConfigEntry: ::std::option::Option<
		_system!(unsafe fn(options: *const OrtSessionOptions, config_key: *const ::std::os::raw::c_char, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr)
	>,
	pub GetSessionConfigEntry: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *const OrtSessionOptions,
				config_key: *const ::std::os::raw::c_char,
				config_value: *mut ::std::os::raw::c_char,
				size: *mut size_t
			) -> OrtStatusPtr
		)
	>,
	pub SessionOptionsAppendExecutionProvider_Dnnl:
		::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, dnnl_options: *const OrtDnnlProviderOptions) -> OrtStatusPtr)>,
	pub CreateDnnlProviderOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtDnnlProviderOptions) -> OrtStatusPtr)>,
	pub UpdateDnnlProviderOptions: ::std::option::Option<
		_system!(
			unsafe fn(
				dnnl_options: *mut OrtDnnlProviderOptions,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetDnnlProviderOptionsAsString: ::std::option::Option<
		_system!(unsafe fn(dnnl_options: *const OrtDnnlProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	#[doc = " \\brief Release an ::OrtDnnlProviderOptions\n\n \\since Version 1.15."]
	pub ReleaseDnnlProviderOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtDnnlProviderOptions))>,
	pub KernelInfo_GetNodeName:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, out: *mut ::std::os::raw::c_char, size: *mut size_t) -> OrtStatusPtr)>,
	pub KernelInfo_GetLogger: ::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, logger: *mut *const OrtLogger) -> OrtStatusPtr)>,
	pub KernelContext_GetLogger: ::std::option::Option<_system!(unsafe fn(context: *const OrtKernelContext, logger: *mut *const OrtLogger) -> OrtStatusPtr)>,
	pub Logger_LogMessage: ::std::option::Option<
		_system!(
			unsafe fn(
				logger: *const OrtLogger,
				log_severity_level: OrtLoggingLevel,
				message: *const ::std::os::raw::c_char,
				file_path: *const ortchar,
				line_number: ::std::os::raw::c_int,
				func_name: *const ::std::os::raw::c_char
			) -> OrtStatusPtr
		)
	>,
	pub Logger_GetLoggingSeverityLevel: ::std::option::Option<_system!(unsafe fn(logger: *const OrtLogger, out: *mut OrtLoggingLevel) -> OrtStatusPtr)>,
	pub KernelInfoGetConstantInput_tensor: ::std::option::Option<
		_system!(unsafe fn(info: *const OrtKernelInfo, index: size_t, is_constant: *mut ::std::os::raw::c_int, out: *mut *const OrtValue) -> OrtStatusPtr)
	>,
	pub CastTypeInfoToOptionalTypeInfo:
		::std::option::Option<_system!(unsafe fn(type_info: *const OrtTypeInfo, out: *mut *const OrtOptionalTypeInfo) -> OrtStatusPtr)>,
	pub GetOptionalContainedTypeInfo:
		::std::option::Option<_system!(unsafe fn(optional_type_info: *const OrtOptionalTypeInfo, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr)>,
	pub GetResizedStringTensorElementBuffer: ::std::option::Option<
		_system!(unsafe fn(value: *mut OrtValue, index: size_t, length_in_bytes: size_t, buffer: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	pub KernelContext_GetAllocator: ::std::option::Option<
		_system!(unsafe fn(context: *const OrtKernelContext, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr)
	>,
	#[doc = " \\brief Returns a null terminated string of the build info including git info and cxx flags\n\n \\return UTF-8 encoded version string. Do not deallocate the returned buffer.\n\n \\since Version 1.15."]
	pub GetBuildInfoString: ::std::option::Option<_system!(unsafe fn() -> *const ::std::os::raw::c_char)>,
	pub CreateROCMProviderOptions: ::std::option::Option<_system!(unsafe fn(out: *mut *mut OrtROCMProviderOptions) -> OrtStatusPtr)>,
	pub UpdateROCMProviderOptions: ::std::option::Option<
		_system!(
			unsafe fn(
				rocm_options: *mut OrtROCMProviderOptions,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub GetROCMProviderOptionsAsString: ::std::option::Option<
		_system!(unsafe fn(rocm_options: *const OrtROCMProviderOptions, allocator: *mut OrtAllocator, ptr: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr)
	>,
	#[doc = " \\brief Release an ::OrtROCMProviderOptions\n\n \\note This is an exception in the naming convention of other Release* functions, as the name of the method does not have the V2 suffix, but the type does\n\n \\since Version 1.16."]
	pub ReleaseROCMProviderOptions: ::std::option::Option<_system!(unsafe fn(input: *mut OrtROCMProviderOptions))>,
	pub CreateAndRegisterAllocatorV2: ::std::option::Option<
		_system!(
			unsafe fn(
				env: *mut OrtEnv,
				provider_type: *const ::std::os::raw::c_char,
				mem_info: *const OrtMemoryInfo,
				arena_cfg: *const OrtArenaCfg,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub RunAsync: ::std::option::Option<
		_system!(
			unsafe fn(
				session: *mut OrtSession,
				run_options: *const OrtRunOptions,
				input_names: *const *const ::std::os::raw::c_char,
				input: *const *const OrtValue,
				input_len: size_t,
				output_names: *const *const ::std::os::raw::c_char,
				output_names_len: size_t,
				output: *mut *mut OrtValue,
				run_async_callback: RunAsyncCallbackFn,
				user_data: *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub UpdateTensorRTProviderOptionsWithValue: ::std::option::Option<
		_system!(
			unsafe fn(
				tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
				key: *const ::std::os::raw::c_char,
				value: *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub GetTensorRTProviderOptionsByName: ::std::option::Option<
		_system!(
			unsafe fn(
				tensorrt_options: *const OrtTensorRTProviderOptionsV2,
				key: *const ::std::os::raw::c_char,
				ptr: *mut *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub UpdateCUDAProviderOptionsWithValue: ::std::option::Option<
		_system!(
			unsafe fn(cuda_options: *mut OrtCUDAProviderOptionsV2, key: *const ::std::os::raw::c_char, value: *mut ::std::os::raw::c_void) -> OrtStatusPtr
		)
	>,
	pub GetCUDAProviderOptionsByName: ::std::option::Option<
		_system!(
			unsafe fn(cuda_options: *const OrtCUDAProviderOptionsV2, key: *const ::std::os::raw::c_char, ptr: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr
		)
	>,
	pub KernelContext_GetResource: ::std::option::Option<
		_system!(
			unsafe fn(
				context: *const OrtKernelContext,
				resouce_version: ::std::os::raw::c_int,
				resource_id: ::std::os::raw::c_int,
				resource: *mut *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub SetUserLoggingFunction: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				user_logging_function: OrtLoggingFunction,
				user_logging_param: *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub ShapeInferContext_GetInputCount: ::std::option::Option<_system!(unsafe fn(context: *const OrtShapeInferContext, out: *mut size_t) -> OrtStatusPtr)>,
	pub ShapeInferContext_GetInputTypeShape: ::std::option::Option<
		_system!(unsafe fn(context: *const OrtShapeInferContext, index: size_t, info: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)
	>,
	pub ShapeInferContext_GetAttribute: ::std::option::Option<
		_system!(unsafe fn(context: *const OrtShapeInferContext, attr_name: *const ::std::os::raw::c_char, attr: *mut *const OrtOpAttr) -> OrtStatusPtr)
	>,
	pub ShapeInferContext_SetOutputTypeShape:
		::std::option::Option<_system!(unsafe fn(context: *const OrtShapeInferContext, index: size_t, info: *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr)>,
	pub SetSymbolicDimensions: ::std::option::Option<
		_system!(unsafe fn(info: *mut OrtTensorTypeAndShapeInfo, dim_params: *mut *const ::std::os::raw::c_char, dim_params_length: size_t) -> OrtStatusPtr)
	>,
	pub ReadOpAttr: ::std::option::Option<
		_system!(unsafe fn(op_attr: *const OrtOpAttr, type_: OrtOpAttrType, data: *mut ::std::os::raw::c_void, len: size_t, out: *mut size_t) -> OrtStatusPtr)
	>,
	pub SetDeterministicCompute: ::std::option::Option<_system!(unsafe fn(options: *mut OrtSessionOptions, value: bool) -> OrtStatusPtr)>,
	pub KernelContext_ParallelFor: ::std::option::Option<
		_system!(
			unsafe fn(
				context: *const OrtKernelContext,
				fn_: ::std::option::Option<_system!(unsafe fn(arg1: *mut ::std::os::raw::c_void, arg2: size_t))>,
				total: size_t,
				num_batch: size_t,
				usr_data: *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub SessionOptionsAppendExecutionProvider_OpenVINO_V2: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub SessionOptionsAppendExecutionProvider_VitisAI: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				provider_options_keys: *const *const ::std::os::raw::c_char,
				provider_options_values: *const *const ::std::os::raw::c_char,
				num_keys: size_t
			) -> OrtStatusPtr
		)
	>,
	pub KernelContext_GetScratchBuffer: ::std::option::Option<
		_system!(
			unsafe fn(
				context: *const OrtKernelContext,
				mem_info: *const OrtMemoryInfo,
				count_or_bytes: size_t,
				out: *mut *mut ::std::os::raw::c_void
			) -> OrtStatusPtr
		)
	>,
	pub KernelInfoGetAllocator:
		::std::option::Option<_system!(unsafe fn(info: *const OrtKernelInfo, mem_type: OrtMemType, out: *mut *mut OrtAllocator) -> OrtStatusPtr)>,
	pub AddExternalInitializersFromMemory: ::std::option::Option<
		_system!(
			unsafe fn(
				options: *mut OrtSessionOptions,
				external_initializer_file_names: *const *const ortchar,
				external_initializer_file_buffer_array: *const *mut ::std::os::raw::c_char,
				external_initializer_file_lengths: *const size_t,
				num_external_initializer_files: size_t
			) -> OrtStatusPtr
		)
	>
}
#[test]
fn bindgen_test_layout_OrtApi() {
	const UNINIT: ::std::mem::MaybeUninit<OrtApi> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtApi>(), 2208usize, concat!("Size of: ", stringify!(OrtApi)));
	assert_eq!(::std::mem::align_of::<OrtApi>(), 8usize, concat!("Alignment of ", stringify!(OrtApi)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateStatus) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateStatus))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetErrorCode) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetErrorCode))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetErrorMessage) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetErrorMessage))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateEnv) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateEnv))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateEnvWithCustomLogger) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateEnvWithCustomLogger))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).EnableTelemetryEvents) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(EnableTelemetryEvents))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).DisableTelemetryEvents) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(DisableTelemetryEvents))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSession) as usize - ptr as usize },
		56usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSession))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSessionFromArray) as usize - ptr as usize },
		64usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSessionFromArray))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Run) as usize - ptr as usize },
		72usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(Run))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSessionOptions) as usize - ptr as usize },
		80usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSessionOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetOptimizedModelFilePath) as usize - ptr as usize },
		88usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetOptimizedModelFilePath))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CloneSessionOptions) as usize - ptr as usize },
		96usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CloneSessionOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSessionExecutionMode) as usize - ptr as usize },
		104usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSessionExecutionMode))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).EnableProfiling) as usize - ptr as usize },
		112usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(EnableProfiling))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).DisableProfiling) as usize - ptr as usize },
		120usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(DisableProfiling))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).EnableMemPattern) as usize - ptr as usize },
		128usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(EnableMemPattern))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).DisableMemPattern) as usize - ptr as usize },
		136usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(DisableMemPattern))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).EnableCpuMemArena) as usize - ptr as usize },
		144usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(EnableCpuMemArena))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).DisableCpuMemArena) as usize - ptr as usize },
		152usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(DisableCpuMemArena))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSessionLogId) as usize - ptr as usize },
		160usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSessionLogId))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSessionLogVerbosityLevel) as usize - ptr as usize },
		168usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSessionLogVerbosityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSessionLogSeverityLevel) as usize - ptr as usize },
		176usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSessionLogSeverityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSessionGraphOptimizationLevel) as usize - ptr as usize },
		184usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSessionGraphOptimizationLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetIntraOpNumThreads) as usize - ptr as usize },
		192usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetIntraOpNumThreads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetInterOpNumThreads) as usize - ptr as usize },
		200usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetInterOpNumThreads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateCustomOpDomain) as usize - ptr as usize },
		208usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateCustomOpDomain))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CustomOpDomain_Add) as usize - ptr as usize },
		216usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CustomOpDomain_Add))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddCustomOpDomain) as usize - ptr as usize },
		224usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddCustomOpDomain))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RegisterCustomOpsLibrary) as usize - ptr as usize },
		232usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RegisterCustomOpsLibrary))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetInputCount) as usize - ptr as usize },
		240usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetInputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOutputCount) as usize - ptr as usize },
		248usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOutputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOverridableInitializerCount) as usize - ptr as usize },
		256usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOverridableInitializerCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetInputTypeInfo) as usize - ptr as usize },
		264usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetInputTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOutputTypeInfo) as usize - ptr as usize },
		272usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOutputTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOverridableInitializerTypeInfo) as usize - ptr as usize },
		280usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOverridableInitializerTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetInputName) as usize - ptr as usize },
		288usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetInputName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOutputName) as usize - ptr as usize },
		296usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOutputName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetOverridableInitializerName) as usize - ptr as usize },
		304usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetOverridableInitializerName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateRunOptions) as usize - ptr as usize },
		312usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateRunOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsSetRunLogVerbosityLevel) as usize - ptr as usize },
		320usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsSetRunLogVerbosityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsSetRunLogSeverityLevel) as usize - ptr as usize },
		328usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsSetRunLogSeverityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsSetRunTag) as usize - ptr as usize },
		336usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsSetRunTag))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsGetRunLogVerbosityLevel) as usize - ptr as usize },
		344usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsGetRunLogVerbosityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsGetRunLogSeverityLevel) as usize - ptr as usize },
		352usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsGetRunLogSeverityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsGetRunTag) as usize - ptr as usize },
		360usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsGetRunTag))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsSetTerminate) as usize - ptr as usize },
		368usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsSetTerminate))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunOptionsUnsetTerminate) as usize - ptr as usize },
		376usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunOptionsUnsetTerminate))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateTensorAsOrtValue) as usize - ptr as usize },
		384usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateTensorAsOrtValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateTensorWithDataAsOrtValue) as usize - ptr as usize },
		392usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateTensorWithDataAsOrtValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).IsTensor) as usize - ptr as usize },
		400usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(IsTensor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorMutableData) as usize - ptr as usize },
		408usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorMutableData))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).FillStringTensor) as usize - ptr as usize },
		416usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(FillStringTensor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetStringTensorDataLength) as usize - ptr as usize },
		424usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetStringTensorDataLength))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetStringTensorContent) as usize - ptr as usize },
		432usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetStringTensorContent))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CastTypeInfoToTensorInfo) as usize - ptr as usize },
		440usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CastTypeInfoToTensorInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOnnxTypeFromTypeInfo) as usize - ptr as usize },
		448usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetOnnxTypeFromTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateTensorTypeAndShapeInfo) as usize - ptr as usize },
		456usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateTensorTypeAndShapeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetTensorElementType) as usize - ptr as usize },
		464usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetTensorElementType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetDimensions) as usize - ptr as usize },
		472usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetDimensions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorElementType) as usize - ptr as usize },
		480usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorElementType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetDimensionsCount) as usize - ptr as usize },
		488usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetDimensionsCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetDimensions) as usize - ptr as usize },
		496usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetDimensions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSymbolicDimensions) as usize - ptr as usize },
		504usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSymbolicDimensions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorShapeElementCount) as usize - ptr as usize },
		512usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorShapeElementCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorTypeAndShape) as usize - ptr as usize },
		520usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorTypeAndShape))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTypeInfo) as usize - ptr as usize },
		528usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetValueType) as usize - ptr as usize },
		536usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetValueType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateMemoryInfo) as usize - ptr as usize },
		544usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateMemoryInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateCpuMemoryInfo) as usize - ptr as usize },
		552usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateCpuMemoryInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CompareMemoryInfo) as usize - ptr as usize },
		560usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CompareMemoryInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).MemoryInfoGetName) as usize - ptr as usize },
		568usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(MemoryInfoGetName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).MemoryInfoGetId) as usize - ptr as usize },
		576usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(MemoryInfoGetId))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).MemoryInfoGetMemType) as usize - ptr as usize },
		584usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(MemoryInfoGetMemType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).MemoryInfoGetType) as usize - ptr as usize },
		592usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(MemoryInfoGetType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AllocatorAlloc) as usize - ptr as usize },
		600usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AllocatorAlloc))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AllocatorFree) as usize - ptr as usize },
		608usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AllocatorFree))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AllocatorGetInfo) as usize - ptr as usize },
		616usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AllocatorGetInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetAllocatorWithDefaultOptions) as usize - ptr as usize },
		624usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetAllocatorWithDefaultOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddFreeDimensionOverride) as usize - ptr as usize },
		632usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddFreeDimensionOverride))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetValue) as usize - ptr as usize },
		640usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetValueCount) as usize - ptr as usize },
		648usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetValueCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateValue) as usize - ptr as usize },
		656usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateOpaqueValue) as usize - ptr as usize },
		664usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateOpaqueValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOpaqueValue) as usize - ptr as usize },
		672usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetOpaqueValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttribute_float) as usize - ptr as usize },
		680usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttribute_float))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttribute_int64) as usize - ptr as usize },
		688usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttribute_int64))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttribute_string) as usize - ptr as usize },
		696usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttribute_string))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetInputCount) as usize - ptr as usize },
		704usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetInputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetOutputCount) as usize - ptr as usize },
		712usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetOutputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetInput) as usize - ptr as usize },
		720usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetInput))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetOutput) as usize - ptr as usize },
		728usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetOutput))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseEnv) as usize - ptr as usize },
		736usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseEnv))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseStatus) as usize - ptr as usize },
		744usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseStatus))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseMemoryInfo) as usize - ptr as usize },
		752usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseMemoryInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseSession) as usize - ptr as usize },
		760usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseSession))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseValue) as usize - ptr as usize },
		768usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseRunOptions) as usize - ptr as usize },
		776usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseRunOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseTypeInfo) as usize - ptr as usize },
		784usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseTensorTypeAndShapeInfo) as usize - ptr as usize },
		792usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseTensorTypeAndShapeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseSessionOptions) as usize - ptr as usize },
		800usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseSessionOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseCustomOpDomain) as usize - ptr as usize },
		808usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseCustomOpDomain))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetDenotationFromTypeInfo) as usize - ptr as usize },
		816usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetDenotationFromTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CastTypeInfoToMapTypeInfo) as usize - ptr as usize },
		824usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CastTypeInfoToMapTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CastTypeInfoToSequenceTypeInfo) as usize - ptr as usize },
		832usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CastTypeInfoToSequenceTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetMapKeyType) as usize - ptr as usize },
		840usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetMapKeyType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetMapValueType) as usize - ptr as usize },
		848usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetMapValueType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSequenceElementType) as usize - ptr as usize },
		856usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSequenceElementType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseMapTypeInfo) as usize - ptr as usize },
		864usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseMapTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseSequenceTypeInfo) as usize - ptr as usize },
		872usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseSequenceTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionEndProfiling) as usize - ptr as usize },
		880usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionEndProfiling))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetModelMetadata) as usize - ptr as usize },
		888usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetModelMetadata))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetProducerName) as usize - ptr as usize },
		896usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetProducerName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetGraphName) as usize - ptr as usize },
		904usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetGraphName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetDomain) as usize - ptr as usize },
		912usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetDomain))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetDescription) as usize - ptr as usize },
		920usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetDescription))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataLookupCustomMetadataMap) as usize - ptr as usize },
		928usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataLookupCustomMetadataMap))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetVersion) as usize - ptr as usize },
		936usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetVersion))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseModelMetadata) as usize - ptr as usize },
		944usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseModelMetadata))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateEnvWithGlobalThreadPools) as usize - ptr as usize },
		952usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateEnvWithGlobalThreadPools))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).DisablePerSessionThreads) as usize - ptr as usize },
		960usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(DisablePerSessionThreads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateThreadingOptions) as usize - ptr as usize },
		968usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateThreadingOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseThreadingOptions) as usize - ptr as usize },
		976usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseThreadingOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetCustomMetadataMapKeys) as usize - ptr as usize },
		984usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetCustomMetadataMapKeys))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddFreeDimensionOverrideByName) as usize - ptr as usize },
		992usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddFreeDimensionOverrideByName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetAvailableProviders) as usize - ptr as usize },
		1000usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetAvailableProviders))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseAvailableProviders) as usize - ptr as usize },
		1008usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseAvailableProviders))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetStringTensorElementLength) as usize - ptr as usize },
		1016usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetStringTensorElementLength))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetStringTensorElement) as usize - ptr as usize },
		1024usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetStringTensorElement))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).FillStringTensorElement) as usize - ptr as usize },
		1032usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(FillStringTensorElement))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddSessionConfigEntry) as usize - ptr as usize },
		1040usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddSessionConfigEntry))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateAllocator) as usize - ptr as usize },
		1048usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseAllocator) as usize - ptr as usize },
		1056usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunWithBinding) as usize - ptr as usize },
		1064usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunWithBinding))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateIoBinding) as usize - ptr as usize },
		1072usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateIoBinding))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseIoBinding) as usize - ptr as usize },
		1080usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseIoBinding))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).BindInput) as usize - ptr as usize },
		1088usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(BindInput))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).BindOutput) as usize - ptr as usize },
		1096usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(BindOutput))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).BindOutputToDevice) as usize - ptr as usize },
		1104usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(BindOutputToDevice))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetBoundOutputNames) as usize - ptr as usize },
		1112usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetBoundOutputNames))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetBoundOutputValues) as usize - ptr as usize },
		1120usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetBoundOutputValues))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ClearBoundInputs) as usize - ptr as usize },
		1128usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ClearBoundInputs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ClearBoundOutputs) as usize - ptr as usize },
		1136usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ClearBoundOutputs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).TensorAt) as usize - ptr as usize },
		1144usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(TensorAt))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateAndRegisterAllocator) as usize - ptr as usize },
		1152usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateAndRegisterAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetLanguageProjection) as usize - ptr as usize },
		1160usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetLanguageProjection))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionGetProfilingStartTimeNs) as usize - ptr as usize },
		1168usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionGetProfilingStartTimeNs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalIntraOpNumThreads) as usize - ptr as usize },
		1176usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalIntraOpNumThreads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalInterOpNumThreads) as usize - ptr as usize },
		1184usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalInterOpNumThreads))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalSpinControl) as usize - ptr as usize },
		1192usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalSpinControl))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddInitializer) as usize - ptr as usize },
		1200usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddInitializer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateEnvWithCustomLoggerAndGlobalThreadPools) as usize - ptr as usize },
		1208usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateEnvWithCustomLoggerAndGlobalThreadPools))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_CUDA) as usize - ptr as usize },
		1216usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_CUDA))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_ROCM) as usize - ptr as usize },
		1224usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_ROCM))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_OpenVINO) as usize - ptr as usize },
		1232usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_OpenVINO))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalDenormalAsZero) as usize - ptr as usize },
		1240usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalDenormalAsZero))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateArenaCfg) as usize - ptr as usize },
		1248usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateArenaCfg))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseArenaCfg) as usize - ptr as usize },
		1256usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseArenaCfg))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ModelMetadataGetGraphDescription) as usize - ptr as usize },
		1264usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ModelMetadataGetGraphDescription))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_TensorRT) as usize - ptr as usize },
		1272usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_TensorRT))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetCurrentGpuDeviceId) as usize - ptr as usize },
		1280usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetCurrentGpuDeviceId))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetCurrentGpuDeviceId) as usize - ptr as usize },
		1288usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetCurrentGpuDeviceId))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttributeArray_float) as usize - ptr as usize },
		1296usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttributeArray_float))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttributeArray_int64) as usize - ptr as usize },
		1304usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttributeArray_int64))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateArenaCfgV2) as usize - ptr as usize },
		1312usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateArenaCfgV2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddRunConfigEntry) as usize - ptr as usize },
		1320usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddRunConfigEntry))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreatePrepackedWeightsContainer) as usize - ptr as usize },
		1328usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreatePrepackedWeightsContainer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleasePrepackedWeightsContainer) as usize - ptr as usize },
		1336usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleasePrepackedWeightsContainer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSessionWithPrepackedWeightsContainer) as usize - ptr as usize },
		1344usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSessionWithPrepackedWeightsContainer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSessionFromArrayWithPrepackedWeightsContainer) as usize - ptr as usize },
		1352usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSessionFromArrayWithPrepackedWeightsContainer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_TensorRT_V2) as usize - ptr as usize },
		1360usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_TensorRT_V2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateTensorRTProviderOptions) as usize - ptr as usize },
		1368usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateTensorRTProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateTensorRTProviderOptions) as usize - ptr as usize },
		1376usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateTensorRTProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorRTProviderOptionsAsString) as usize - ptr as usize },
		1384usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorRTProviderOptionsAsString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseTensorRTProviderOptions) as usize - ptr as usize },
		1392usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseTensorRTProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).EnableOrtCustomOps) as usize - ptr as usize },
		1400usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(EnableOrtCustomOps))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RegisterAllocator) as usize - ptr as usize },
		1408usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RegisterAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UnregisterAllocator) as usize - ptr as usize },
		1416usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UnregisterAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).IsSparseTensor) as usize - ptr as usize },
		1424usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(IsSparseTensor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSparseTensorAsOrtValue) as usize - ptr as usize },
		1432usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSparseTensorAsOrtValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).FillSparseTensorCoo) as usize - ptr as usize },
		1440usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(FillSparseTensorCoo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).FillSparseTensorCsr) as usize - ptr as usize },
		1448usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(FillSparseTensorCsr))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).FillSparseTensorBlockSparse) as usize - ptr as usize },
		1456usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(FillSparseTensorBlockSparse))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateSparseTensorWithValuesAsOrtValue) as usize - ptr as usize },
		1464usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateSparseTensorWithValuesAsOrtValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UseCooIndices) as usize - ptr as usize },
		1472usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UseCooIndices))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UseCsrIndices) as usize - ptr as usize },
		1480usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UseCsrIndices))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UseBlockSparseIndices) as usize - ptr as usize },
		1488usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UseBlockSparseIndices))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSparseTensorFormat) as usize - ptr as usize },
		1496usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSparseTensorFormat))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSparseTensorValuesTypeAndShape) as usize - ptr as usize },
		1504usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSparseTensorValuesTypeAndShape))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSparseTensorValues) as usize - ptr as usize },
		1512usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSparseTensorValues))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSparseTensorIndicesTypeShape) as usize - ptr as usize },
		1520usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSparseTensorIndicesTypeShape))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSparseTensorIndices) as usize - ptr as usize },
		1528usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSparseTensorIndices))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).HasValue) as usize - ptr as usize },
		1536usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(HasValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetGPUComputeStream) as usize - ptr as usize },
		1544usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetGPUComputeStream))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorMemoryInfo) as usize - ptr as usize },
		1552usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorMemoryInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetExecutionProviderApi) as usize - ptr as usize },
		1560usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetExecutionProviderApi))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsSetCustomCreateThreadFn) as usize - ptr as usize },
		1568usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsSetCustomCreateThreadFn))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsSetCustomThreadCreationOptions) as usize - ptr as usize },
		1576usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsSetCustomThreadCreationOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsSetCustomJoinThreadFn) as usize - ptr as usize },
		1584usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsSetCustomJoinThreadFn))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalCustomCreateThreadFn) as usize - ptr as usize },
		1592usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalCustomCreateThreadFn))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalCustomThreadCreationOptions) as usize - ptr as usize },
		1600usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalCustomThreadCreationOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalCustomJoinThreadFn) as usize - ptr as usize },
		1608usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalCustomJoinThreadFn))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SynchronizeBoundInputs) as usize - ptr as usize },
		1616usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SynchronizeBoundInputs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SynchronizeBoundOutputs) as usize - ptr as usize },
		1624usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SynchronizeBoundOutputs))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_CUDA_V2) as usize - ptr as usize },
		1632usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_CUDA_V2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateCUDAProviderOptions) as usize - ptr as usize },
		1640usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateCUDAProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateCUDAProviderOptions) as usize - ptr as usize },
		1648usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateCUDAProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetCUDAProviderOptionsAsString) as usize - ptr as usize },
		1656usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetCUDAProviderOptionsAsString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseCUDAProviderOptions) as usize - ptr as usize },
		1664usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseCUDAProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_MIGraphX) as usize - ptr as usize },
		1672usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_MIGraphX))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).AddExternalInitializers) as usize - ptr as usize },
		1680usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(AddExternalInitializers))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateOpAttr) as usize - ptr as usize },
		1688usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateOpAttr))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseOpAttr) as usize - ptr as usize },
		1696usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseOpAttr))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateOp) as usize - ptr as usize },
		1704usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateOp))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).InvokeOp) as usize - ptr as usize },
		1712usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(InvokeOp))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseOp) as usize - ptr as usize },
		1720usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseOp))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider) as usize - ptr as usize },
		1728usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CopyKernelInfo) as usize - ptr as usize },
		1736usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CopyKernelInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseKernelInfo) as usize - ptr as usize },
		1744usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseKernelInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTrainingApi) as usize - ptr as usize },
		1752usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTrainingApi))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_CANN) as usize - ptr as usize },
		1760usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_CANN))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateCANNProviderOptions) as usize - ptr as usize },
		1768usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateCANNProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateCANNProviderOptions) as usize - ptr as usize },
		1776usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateCANNProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetCANNProviderOptionsAsString) as usize - ptr as usize },
		1784usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetCANNProviderOptionsAsString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseCANNProviderOptions) as usize - ptr as usize },
		1792usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseCANNProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).MemoryInfoGetDeviceType) as usize - ptr as usize },
		1800usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(MemoryInfoGetDeviceType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateEnvWithCustomLogLevel) as usize - ptr as usize },
		1808usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateEnvWithCustomLogLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetGlobalIntraOpThreadAffinity) as usize - ptr as usize },
		1816usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetGlobalIntraOpThreadAffinity))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RegisterCustomOpsLibrary_V2) as usize - ptr as usize },
		1824usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RegisterCustomOpsLibrary_V2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RegisterCustomOpsUsingFunction) as usize - ptr as usize },
		1832usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RegisterCustomOpsUsingFunction))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetInputCount) as usize - ptr as usize },
		1840usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetInputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetOutputCount) as usize - ptr as usize },
		1848usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetOutputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetInputName) as usize - ptr as usize },
		1856usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetInputName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetOutputName) as usize - ptr as usize },
		1864usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetOutputName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetInputTypeInfo) as usize - ptr as usize },
		1872usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetInputTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetOutputTypeInfo) as usize - ptr as usize },
		1880usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetOutputTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetAttribute_tensor) as usize - ptr as usize },
		1888usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetAttribute_tensor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).HasSessionConfigEntry) as usize - ptr as usize },
		1896usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(HasSessionConfigEntry))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetSessionConfigEntry) as usize - ptr as usize },
		1904usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetSessionConfigEntry))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_Dnnl) as usize - ptr as usize },
		1912usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_Dnnl))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateDnnlProviderOptions) as usize - ptr as usize },
		1920usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateDnnlProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateDnnlProviderOptions) as usize - ptr as usize },
		1928usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateDnnlProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetDnnlProviderOptionsAsString) as usize - ptr as usize },
		1936usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetDnnlProviderOptionsAsString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseDnnlProviderOptions) as usize - ptr as usize },
		1944usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseDnnlProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetNodeName) as usize - ptr as usize },
		1952usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetNodeName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfo_GetLogger) as usize - ptr as usize },
		1960usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfo_GetLogger))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetLogger) as usize - ptr as usize },
		1968usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetLogger))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Logger_LogMessage) as usize - ptr as usize },
		1976usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(Logger_LogMessage))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).Logger_GetLoggingSeverityLevel) as usize - ptr as usize },
		1984usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(Logger_GetLoggingSeverityLevel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelInfoGetConstantInput_tensor) as usize - ptr as usize },
		1992usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelInfoGetConstantInput_tensor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CastTypeInfoToOptionalTypeInfo) as usize - ptr as usize },
		2000usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CastTypeInfoToOptionalTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOptionalContainedTypeInfo) as usize - ptr as usize },
		2008usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetOptionalContainedTypeInfo))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetResizedStringTensorElementBuffer) as usize - ptr as usize },
		2016usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetResizedStringTensorElementBuffer))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetAllocator) as usize - ptr as usize },
		2024usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetAllocator))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetBuildInfoString) as usize - ptr as usize },
		2032usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetBuildInfoString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateROCMProviderOptions) as usize - ptr as usize },
		2040usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateROCMProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateROCMProviderOptions) as usize - ptr as usize },
		2048usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateROCMProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetROCMProviderOptionsAsString) as usize - ptr as usize },
		2056usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetROCMProviderOptionsAsString))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReleaseROCMProviderOptions) as usize - ptr as usize },
		2064usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReleaseROCMProviderOptions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateAndRegisterAllocatorV2) as usize - ptr as usize },
		2072usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(CreateAndRegisterAllocatorV2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).RunAsync) as usize - ptr as usize },
		2080usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(RunAsync))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateTensorRTProviderOptionsWithValue) as usize - ptr as usize },
		2088usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateTensorRTProviderOptionsWithValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetTensorRTProviderOptionsByName) as usize - ptr as usize },
		2096usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetTensorRTProviderOptionsByName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).UpdateCUDAProviderOptionsWithValue) as usize - ptr as usize },
		2104usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(UpdateCUDAProviderOptionsWithValue))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetCUDAProviderOptionsByName) as usize - ptr as usize },
		2112usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(GetCUDAProviderOptionsByName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_GetResource) as usize - ptr as usize },
		2120usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_GetResource))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetUserLoggingFunction) as usize - ptr as usize },
		2128usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetUserLoggingFunction))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ShapeInferContext_GetInputCount) as usize - ptr as usize },
		2136usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ShapeInferContext_GetInputCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ShapeInferContext_GetInputTypeShape) as usize - ptr as usize },
		2144usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ShapeInferContext_GetInputTypeShape))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ShapeInferContext_GetAttribute) as usize - ptr as usize },
		2152usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ShapeInferContext_GetAttribute))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ShapeInferContext_SetOutputTypeShape) as usize - ptr as usize },
		2160usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ShapeInferContext_SetOutputTypeShape))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetSymbolicDimensions) as usize - ptr as usize },
		2168usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetSymbolicDimensions))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).ReadOpAttr) as usize - ptr as usize },
		2176usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(ReadOpAttr))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SetDeterministicCompute) as usize - ptr as usize },
		2184usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SetDeterministicCompute))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelContext_ParallelFor) as usize - ptr as usize },
		2192usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(KernelContext_ParallelFor))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).SessionOptionsAppendExecutionProvider_OpenVINO_V2) as usize - ptr as usize },
		2200usize,
		concat!("Offset of field: ", stringify!(OrtApi), "::", stringify!(SessionOptionsAppendExecutionProvider_OpenVINO_V2))
	);
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
	pub CreateKernel:
		::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, api: *const OrtApi, info: *const OrtKernelInfo) -> *mut ::std::os::raw::c_void)>,
	pub GetName: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> *const ::std::os::raw::c_char)>,
	pub GetExecutionProviderType: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> *const ::std::os::raw::c_char)>,
	pub GetInputType: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, index: size_t) -> ONNXTensorElementDataType)>,
	pub GetInputTypeCount: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> size_t)>,
	pub GetOutputType: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, index: size_t) -> ONNXTensorElementDataType)>,
	pub GetOutputTypeCount: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> size_t)>,
	pub KernelCompute: ::std::option::Option<_system!(unsafe fn(op_kernel: *mut ::std::os::raw::c_void, context: *mut OrtKernelContext))>,
	pub KernelDestroy: ::std::option::Option<_system!(unsafe fn(op_kernel: *mut ::std::os::raw::c_void))>,
	pub GetInputCharacteristic: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, index: size_t) -> OrtCustomOpInputOutputCharacteristic)>,
	pub GetOutputCharacteristic: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, index: size_t) -> OrtCustomOpInputOutputCharacteristic)>,
	pub GetInputMemoryType: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, index: size_t) -> OrtMemType)>,
	pub GetVariadicInputMinArity: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub GetVariadicInputHomogeneity: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub GetVariadicOutputMinArity: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub GetVariadicOutputHomogeneity: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub CreateKernelV2: ::std::option::Option<
		_system!(unsafe fn(op: *const OrtCustomOp, api: *const OrtApi, info: *const OrtKernelInfo, kernel: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr)
	>,
	pub KernelComputeV2: ::std::option::Option<_system!(unsafe fn(op_kernel: *mut ::std::os::raw::c_void, context: *mut OrtKernelContext) -> OrtStatusPtr)>,
	pub InferOutputShapeFn: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp, arg1: *mut OrtShapeInferContext) -> OrtStatusPtr)>,
	pub GetStartVersion: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub GetEndVersion: ::std::option::Option<_system!(unsafe fn(op: *const OrtCustomOp) -> ::std::os::raw::c_int)>,
	pub GetMayInplace:
		::std::option::Option<_system!(unsafe fn(input_index: *mut *mut ::std::os::raw::c_int, output_index: *mut *mut ::std::os::raw::c_int) -> size_t)>,
	pub ReleaseMayInplace: ::std::option::Option<_system!(unsafe fn(input_index: *mut ::std::os::raw::c_int, output_index: *mut *mut ::std::os::raw::c_int))>,
	pub GetAliasMap:
		::std::option::Option<_system!(unsafe fn(input_index: *mut *mut ::std::os::raw::c_int, output_index: *mut *mut ::std::os::raw::c_int) -> size_t)>,
	pub ReleaseAliasMap: ::std::option::Option<_system!(unsafe fn(input_index: *mut ::std::os::raw::c_int, output_index: *mut *mut ::std::os::raw::c_int))>
}
#[test]
fn bindgen_test_layout_OrtCustomOp() {
	const UNINIT: ::std::mem::MaybeUninit<OrtCustomOp> = ::std::mem::MaybeUninit::uninit();
	let ptr = UNINIT.as_ptr();
	assert_eq!(::std::mem::size_of::<OrtCustomOp>(), 176usize, concat!("Size of: ", stringify!(OrtCustomOp)));
	assert_eq!(::std::mem::align_of::<OrtCustomOp>(), 8usize, concat!("Alignment of ", stringify!(OrtCustomOp)));
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).version) as usize - ptr as usize },
		0usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(version))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateKernel) as usize - ptr as usize },
		8usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(CreateKernel))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetName) as usize - ptr as usize },
		16usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetName))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetExecutionProviderType) as usize - ptr as usize },
		24usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetExecutionProviderType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetInputType) as usize - ptr as usize },
		32usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetInputType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetInputTypeCount) as usize - ptr as usize },
		40usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetInputTypeCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOutputType) as usize - ptr as usize },
		48usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetOutputType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOutputTypeCount) as usize - ptr as usize },
		56usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetOutputTypeCount))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelCompute) as usize - ptr as usize },
		64usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(KernelCompute))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelDestroy) as usize - ptr as usize },
		72usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(KernelDestroy))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetInputCharacteristic) as usize - ptr as usize },
		80usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetInputCharacteristic))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetOutputCharacteristic) as usize - ptr as usize },
		88usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetOutputCharacteristic))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetInputMemoryType) as usize - ptr as usize },
		96usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetInputMemoryType))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetVariadicInputMinArity) as usize - ptr as usize },
		104usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetVariadicInputMinArity))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetVariadicInputHomogeneity) as usize - ptr as usize },
		112usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetVariadicInputHomogeneity))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetVariadicOutputMinArity) as usize - ptr as usize },
		120usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetVariadicOutputMinArity))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetVariadicOutputHomogeneity) as usize - ptr as usize },
		128usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetVariadicOutputHomogeneity))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).CreateKernelV2) as usize - ptr as usize },
		136usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(CreateKernelV2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).KernelComputeV2) as usize - ptr as usize },
		144usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(KernelComputeV2))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).InferOutputShapeFn) as usize - ptr as usize },
		152usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(InferOutputShapeFn))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetStartVersion) as usize - ptr as usize },
		160usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetStartVersion))
	);
	assert_eq!(
		unsafe { ::std::ptr::addr_of!((*ptr).GetEndVersion) as usize - ptr as usize },
		168usize,
		concat!("Offset of field: ", stringify!(OrtCustomOp), "::", stringify!(GetEndVersion))
	);
}
_system_block! {
	pub fn OrtSessionOptionsAppendExecutionProvider_CUDA(options: *mut OrtSessionOptions, device_id: ::std::os::raw::c_int) -> OrtStatusPtr;
}
_system_block! {
	pub fn OrtSessionOptionsAppendExecutionProvider_ROCM(options: *mut OrtSessionOptions, device_id: ::std::os::raw::c_int) -> OrtStatusPtr;
}
_system_block! {
	pub fn OrtSessionOptionsAppendExecutionProvider_MIGraphX(options: *mut OrtSessionOptions, device_id: ::std::os::raw::c_int) -> OrtStatusPtr;
}
_system_block! {
	pub fn OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut OrtSessionOptions, use_arena: ::std::os::raw::c_int) -> OrtStatusPtr;
}
_system_block! {
	pub fn OrtSessionOptionsAppendExecutionProvider_Tensorrt(options: *mut OrtSessionOptions, device_id: ::std::os::raw::c_int) -> OrtStatusPtr;
}
