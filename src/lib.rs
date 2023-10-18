#![doc = include_str!("../README.md")]

pub mod download;
pub(crate) mod environment;
pub(crate) mod error;
pub(crate) mod execution_providers;
pub(crate) mod io_binding;
pub(crate) mod memory;
pub(crate) mod metadata;
pub(crate) mod session;
pub(crate) mod tensor;
pub(crate) mod value;

use std::{
	ffi::{self, CStr},
	os::raw::c_char,
	ptr,
	sync::{atomic::AtomicPtr, Arc, Mutex}
};

use once_cell::sync::Lazy;
#[doc(hidden)]
pub use smallvec;
use tracing::warn;

pub use self::environment::{Environment, EnvironmentBuilder};
#[cfg(feature = "fetch-models")]
pub use self::error::OrtDownloadError;
pub use self::error::{OrtApiError, OrtError, OrtResult};
pub use self::execution_providers::{
	ACLExecutionProviderOptions, ArenaExtendStrategy, CPUExecutionProviderOptions, CUDAExecutionProviderCuDNNConvAlgoSearch, CUDAExecutionProviderOptions,
	CoreMLExecutionProviderOptions, DirectMLExecutionProviderOptions, ExecutionProvider, NNAPIExecutionProviderOptions, OneDNNExecutionProviderOptions,
	OpenVINOExecutionProviderOptions, QNNExecutionHTPPerformanceMode, QNNExecutionProviderOptions, ROCmExecutionProviderOptions, TVMExecutionProviderOptions,
	TVMExecutorType, TVMTuningType, TensorRTExecutionProviderOptions
};
pub use self::io_binding::IoBinding;
pub use self::memory::{AllocationDevice, MemoryInfo};
pub use self::metadata::ModelMetadata;
pub use self::session::{InMemorySession, Session, SessionBuilder, SessionInputs, SessionOutputs, SharedSessionInner};
pub use self::tensor::{
	ort_owned_tensor::ViewHolder, IntoTensorElementDataType, NdArrayExtensions, OrtOwnedTensor, TensorData, TensorDataToType, TensorElementDataType
};
pub use self::value::Value;

#[cfg(not(all(target_arch = "x86", target_os = "windows")))]
macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}
#[cfg(all(target_arch = "x86", target_os = "windows"))]
macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "stdcall" fn $($tt)*);
}

pub(crate) use extern_system_fn;

#[cfg(feature = "load-dynamic")]
pub(crate) static G_ORT_DYLIB_PATH: Lazy<Arc<String>> = Lazy::new(|| {
	let path = match std::env::var("ORT_DYLIB_PATH") {
		Ok(s) if !s.is_empty() => s,
		#[cfg(target_os = "windows")]
		_ => "onnxruntime.dll".to_owned(),
		#[cfg(any(target_os = "linux", target_os = "android"))]
		_ => "libonnxruntime.so".to_owned(),
		#[cfg(target_os = "macos")]
		_ => "libonnxruntime.dylib".to_owned()
	};
	Arc::new(path)
});
#[cfg(feature = "load-dynamic")]
pub(crate) static G_ORT_LIB: Lazy<Arc<Mutex<AtomicPtr<libloading::Library>>>> = Lazy::new(|| {
	unsafe {
		// resolve path relative to executable
		let path: std::path::PathBuf = (&**G_ORT_DYLIB_PATH).into();
		let absolute_path = if path.is_absolute() {
			path
		} else {
			let relative = std::env::current_exe()
				.expect("could not get current executable path")
				.parent()
				.unwrap()
				.join(&path);
			if relative.exists() { relative } else { path }
		};
		let lib = libloading::Library::new(&absolute_path).unwrap_or_else(|e| panic!("could not load the library at `{}`: {e:?}", absolute_path.display()));
		Arc::new(Mutex::new(AtomicPtr::new(Box::leak(Box::new(lib)) as *mut _)))
	}
});

pub(crate) static G_ORT_API: Lazy<Arc<Mutex<AtomicPtr<ort_sys::OrtApi>>>> = Lazy::new(|| {
	#[cfg(feature = "load-dynamic")]
	unsafe {
		let dylib = *G_ORT_LIB
			.lock()
			.expect("failed to acquire ONNX Runtime dylib lock; another thread panicked?")
			.get_mut();
		let base_getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort_sys::OrtApiBase> = (*dylib)
			.get(b"OrtGetApiBase")
			.expect("`OrtGetApiBase` must be present in ONNX Runtime dylib");
		let base: *const ort_sys::OrtApiBase = base_getter();
		assert_ne!(base, ptr::null());

		let get_version_string: extern_system_fn! { unsafe fn () -> *const ffi::c_char } =
			(*base).GetVersionString.expect("`GetVersionString` must be present in `OrtApiBase`");
		let version_string = get_version_string();
		let version_string = CStr::from_ptr(version_string).to_string_lossy();
		let lib_minor_version = version_string.split('.').nth(1).map(|x| x.parse::<u32>().unwrap_or(0)).unwrap_or(0);
		match lib_minor_version.cmp(&15) {
			std::cmp::Ordering::Less => panic!(
				"ort 2.0 is not compatible with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.15.x', but got '{version_string}'",
				**G_ORT_DYLIB_PATH
			),
			std::cmp::Ordering::Greater => warn!(
				"ort 2.0 may have compatibility issues with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.15.x', but got '{version_string}'",
				**G_ORT_DYLIB_PATH
			),
			std::cmp::Ordering::Equal => {}
		};
		let get_api: extern_system_fn! { unsafe fn(u32) -> *const ort_sys::OrtApi } = (*base).GetApi.expect("`GetApi` must be present in `OrtApiBase`");
		let api: *const ort_sys::OrtApi = get_api(ort_sys::ORT_API_VERSION);
		Arc::new(Mutex::new(AtomicPtr::new(api as *mut ort_sys::OrtApi)))
	}
	#[cfg(not(feature = "load-dynamic"))]
	{
		let base: *const ort_sys::OrtApiBase = unsafe { ort_sys::OrtGetApiBase() };
		assert_ne!(base, ptr::null());
		let get_api: extern_system_fn! { unsafe fn(u32) -> *const ort_sys::OrtApi } = unsafe { (*base).GetApi.unwrap() };
		let api: *const ort_sys::OrtApi = unsafe { get_api(ort_sys::ORT_API_VERSION) };
		Arc::new(Mutex::new(AtomicPtr::new(api as *mut ort_sys::OrtApi)))
	}
});

/// Attempts to acquire the global [`ort_sys::OrtApi`] object.
///
/// # Panics
///
/// Panics if another thread panicked while holding the API lock, or if the ONNX Runtime API could not be initialized.
pub fn ort() -> ort_sys::OrtApi {
	let mut api_ref = G_ORT_API.lock().expect("failed to acquire OrtApi lock; another thread panicked?");
	let api_ref_mut: &mut *mut ort_sys::OrtApi = api_ref.get_mut();
	let api_ptr_mut: *mut ort_sys::OrtApi = *api_ref_mut;

	assert_ne!(api_ptr_mut, ptr::null_mut());

	unsafe { *api_ptr_mut }
}

macro_rules! ortsys {
	($method:ident) => {
		$crate::ort().$method.unwrap()
	};
	(unsafe $method:ident) => {
		unsafe { $crate::ort().$method.unwrap() }
	};
	($method:ident($($n:expr),+ $(,)?)) => {
		$crate::ort().$method.unwrap()($($n),+)
	};
	(unsafe $method:ident($($n:expr),+ $(,)?)) => {
		unsafe { $crate::ort().$method.unwrap()($($n),+) }
	};
	($method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::ort().$method.unwrap()($($n),+);
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		let _x = unsafe { $crate::ort().$method.unwrap()($($n),+) };
		$($crate::error::assert_non_null_pointer($check, stringify!($method)).unwrap();)+
		_x
	}};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result($crate::ort().$method.unwrap()($($n),+)).map_err($err)?;
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result(unsafe { $crate::ort().$method.unwrap()($($n),+) }).map_err($err)?;
	};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::error::status_to_result($crate::ort().$method.unwrap()($($n),+)).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		$crate::error::status_to_result(unsafe { $crate::ort().$method.unwrap()($($n),+) }).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	}};
}

macro_rules! ortfree {
	(unsafe $allocator_ptr:expr, $ptr:expr) => {
		unsafe { (*($allocator_ptr)).Free.unwrap()($allocator_ptr, $ptr as *mut std::ffi::c_void) }
	};
	($allocator_ptr:expr, $ptr:expr) => {
		(*$allocator_ptr).Free.unwrap()($allocator_ptr, $ptr as *mut std::ffi::c_void)
	};
}

pub(crate) use ortfree;
pub(crate) use ortsys;

pub(crate) fn char_p_to_string(raw: *const c_char) -> OrtResult<String> {
	let c_string = unsafe { CStr::from_ptr(raw as *mut c_char).to_owned() };
	match c_string.into_string() {
		Ok(string) => Ok(string),
		Err(e) => Err(OrtApiError::IntoStringError(e))
	}
	.map_err(OrtError::FfiStringConversion)
}

/// ONNX's logger sends the code location where the log occurred, which will be parsed into this struct.
#[derive(Debug)]
struct CodeLocation<'a> {
	file: &'a str,
	line: &'a str,
	function: &'a str
}

impl<'a> From<&'a str> for CodeLocation<'a> {
	fn from(code_location: &'a str) -> Self {
		let mut splitter = code_location.split(' ');
		let file_and_line = splitter.next().unwrap_or("<unknown file>:<unknown line>");
		let function = splitter.next().unwrap_or("<unknown function>");
		let mut file_and_line_splitter = file_and_line.split(':');
		let file = file_and_line_splitter.next().unwrap_or("<unknown file>");
		let line = file_and_line_splitter.next().unwrap_or("<unknown line>");

		CodeLocation { file, line, function }
	}
}

extern_system_fn! {
	/// Callback from C that will handle ONNX logging, forwarding ONNX's logs to the `tracing` crate.
	pub(crate) fn custom_logger(_params: *mut ffi::c_void, severity: ort_sys::OrtLoggingLevel, category: *const c_char, log_id: *const c_char, code_location: *const c_char, message: *const c_char) {
		use tracing::{span, Level, trace, debug, warn, info, error};

		let log_level = match severity {
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => Level::TRACE,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => Level::DEBUG,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => Level::INFO,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => Level::WARN,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => Level::ERROR
		};

		assert_ne!(category, ptr::null());
		let category = unsafe { CStr::from_ptr(category) };
		assert_ne!(code_location, ptr::null());
		let code_location = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("unknown");
		assert_ne!(message, ptr::null());
		let message = unsafe { CStr::from_ptr(message) }.to_str().unwrap_or("<invalid>");
		assert_ne!(log_id, ptr::null());
		let log_id = unsafe { CStr::from_ptr(log_id) };

		let code_location = CodeLocation::from(code_location);
		let span = span!(
			Level::TRACE,
			"ort",
			category = category.to_str().unwrap_or("<unknown>"),
			file = code_location.file,
			line = code_location.line,
			function = code_location.function,
			log_id = log_id.to_str().unwrap_or("<unknown>")
		);
		let _enter = span.enter();

		match log_level {
			Level::TRACE => trace!("{}", message),
			Level::DEBUG => debug!("{}", message),
			Level::INFO => info!("{}", message),
			Level::WARN => warn!("{}", message),
			Level::ERROR => error!("{}", message)
		}
	}
}

/// The minimum logging level. Logs will be handled by the `tracing` crate.
#[derive(Debug)]
pub enum LoggingLevel {
	/// Verbose logging level. This will log *a lot* of messages!
	Verbose,
	/// Info logging level.
	Info,
	/// Warning logging level. Recommended to receive potentially important warnings.
	Warning,
	/// Error logging level.
	Error,
	/// Fatal logging level.
	Fatal
}

impl From<LoggingLevel> for ort_sys::OrtLoggingLevel {
	fn from(logging_level: LoggingLevel) -> Self {
		match logging_level {
			LoggingLevel::Verbose => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
			LoggingLevel::Info => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
			LoggingLevel::Warning => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
			LoggingLevel::Error => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
			LoggingLevel::Fatal => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL
		}
	}
}

/// ONNX Runtime provides various graph optimizations to improve performance. Graph optimizations are essentially
/// graph-level transformations, ranging from small graph simplifications and node eliminations to more complex node
/// fusions and layout optimizations.
///
/// Graph optimizations are divided in several categories (or levels) based on their complexity and functionality. They
/// can be performed either online or offline. In online mode, the optimizations are done before performing the
/// inference, while in offline mode, the runtime saves the optimized graph to disk (most commonly used when converting
/// an ONNX model to an ONNX Runtime model).
///
/// The optimizations belonging to one level are performed after the optimizations of the previous level have been
/// applied (e.g., extended optimizations are applied after basic optimizations have been applied).
///
/// **All optimizations (i.e. [`GraphOptimizationLevel::Level3`]) are enabled by default.**
///
/// # Online/offline mode
/// All optimizations can be performed either online or offline. In online mode, when initializing an inference session,
/// we also apply all enabled graph optimizations before performing model inference. Applying all optimizations each
/// time we initiate a session can add overhead to the model startup time (especially for complex models), which can be
/// critical in production scenarios. This is where the offline mode can bring a lot of benefit. In offline mode, after
/// performing graph optimizations, ONNX Runtime serializes the resulting model to disk. Subsequently, we can reduce
/// startup time by using the already optimized model and disabling all optimizations.
///
/// ## Notes:
/// - When running in offline mode, make sure to use the exact same options (e.g., execution providers, optimization
///   level) and hardware as the target machine that the model inference will run on (e.g., you cannot run a model
///   pre-optimized for a GPU execution provider on a machine that is equipped only with CPU).
/// - When layout optimizations are enabled, the offline mode can only be used on compatible hardware to the environment
///   when the offline model is saved. For example, if model has layout optimized for AVX2, the offline model would
///   require CPUs that support AVX2.
#[derive(Debug)]
pub enum GraphOptimizationLevel {
	/// Disables all graph optimizations.
	Disable,
	/// Level 1 includes semantics-preserving graph rewrites which remove redundant nodes and redundant computation.
	/// They run before graph partitioning and thus apply to all the execution providers. Available basic/level 1 graph
	/// optimizations are as follows:
	///
	/// - Constant Folding: Statically computes parts of the graph that rely only on constant initializers. This
	///   eliminates the need to compute them during runtime.
	/// - Redundant node eliminations: Remove all redundant nodes without changing the graph structure. The following
	///   such optimizations are currently supported:
	///   * Identity Elimination
	///   * Slice Elimination
	///   * Unsqueeze Elimination
	///   * Dropout Elimination
	/// - Semantics-preserving node fusions : Fuse/fold multiple nodes into a single node. For example, Conv Add fusion
	///   folds the Add operator as the bias of the Conv operator. The following such optimizations are currently
	///   supported:
	///   * Conv Add Fusion
	///   * Conv Mul Fusion
	///   * Conv BatchNorm Fusion
	///   * Relu Clip Fusion
	///   * Reshape Fusion
	Level1,
	#[rustfmt::skip]
	/// Level 2 optimizations include complex node fusions. They are run after graph partitioning and are only applied to
	/// the nodes assigned to the CPU or CUDA execution provider. Available extended/level 2 graph optimizations are as follows:
	///
	/// | Optimization                    | EPs       | Comments                                                                       |
	/// |:------------------------------- |:--------- |:------------------------------------------------------------------------------ |
	/// | GEMM Activation Fusion          | CPU       |                                                                                |
	/// | Matmul Add Fusion               | CPU       |                                                                                |
	/// | Conv Activation Fusion          | CPU       |                                                                                |
	/// | GELU Fusion                     | CPU, CUDA |                                                                                |
	/// | Layer Normalization Fusion      | CPU, CUDA |                                                                                |
	/// | BERT Embedding Layer Fusion     | CPU, CUDA | Fuses BERT embedding layers, layer normalization, & attention mask length      |
	/// | Attention Fusion*               | CPU, CUDA |                                                                                |
	/// | Skip Layer Normalization Fusion | CPU, CUDA | Fuse bias of fully connected layers, skip connections, and layer normalization |
	/// | Bias GELU Fusion                | CPU, CUDA | Fuse bias of fully connected layers & GELU activation                          |
	/// | GELU Approximation*             | CUDA      | Disabled by default; enable with `OrtSessionOptions::EnableGeluApproximation`  |
	///
	/// > **NOTE**: To optimize performance of the BERT model, approximation is used in GELU Approximation and Attention
	/// Fusion for the CUDA execution provider. The impact on accuracy is negligible based on our evaluation; F1 score
	/// for a BERT model on SQuAD v1.1 is almost the same (87.05 vs 87.03).
	Level2,
	/// Level 3 optimizations include memory layout optimizations, which may optimize the graph to use the NCHWc memory
	/// layout rather than NCHW to improve spatial locality for some targets.
	Level3
}

impl From<GraphOptimizationLevel> for ort_sys::GraphOptimizationLevel {
	fn from(val: GraphOptimizationLevel) -> Self {
		match val {
			GraphOptimizationLevel::Disable => ort_sys::GraphOptimizationLevel::ORT_DISABLE_ALL,
			GraphOptimizationLevel::Level1 => ort_sys::GraphOptimizationLevel::ORT_ENABLE_BASIC,
			GraphOptimizationLevel::Level2 => ort_sys::GraphOptimizationLevel::ORT_ENABLE_EXTENDED,
			GraphOptimizationLevel::Level3 => ort_sys::GraphOptimizationLevel::ORT_ENABLE_ALL
		}
	}
}

/// Execution provider allocator type.
#[derive(Debug, Copy, Clone)]
pub enum AllocatorType {
	/// Default device-specific allocator.
	Device,
	/// Arena allocator.
	Arena
}

impl From<AllocatorType> for ort_sys::OrtAllocatorType {
	fn from(val: AllocatorType) -> Self {
		match val {
			AllocatorType::Device => ort_sys::OrtAllocatorType::OrtDeviceAllocator,
			AllocatorType::Arena => ort_sys::OrtAllocatorType::OrtArenaAllocator
		}
	}
}

/// Memory types for allocated memory.
#[derive(Debug, Copy, Clone)]
pub enum MemType {
	/// Any CPU memory used by non-CPU execution provider.
	CPUInput,
	/// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED.
	CPUOutput,
	/// The default allocator for an execution provider.
	Default
}

impl MemType {
	/// Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. `CUDA_PINNED`.
	pub const CPU: MemType = MemType::CPUOutput;
}

impl From<MemType> for ort_sys::OrtMemType {
	fn from(val: MemType) -> Self {
		match val {
			MemType::CPUInput => ort_sys::OrtMemType::OrtMemTypeCPUInput,
			MemType::CPUOutput => ort_sys::OrtMemType::OrtMemTypeCPUOutput,
			MemType::Default => ort_sys::OrtMemType::OrtMemTypeDefault
		}
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn test_char_p_to_string() {
		let s = ffi::CString::new("foo").unwrap();
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).unwrap());
	}
}
