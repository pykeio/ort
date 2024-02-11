//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::{
	ffi::CString,
	fmt,
	marker::PhantomData,
	ops::Deref,
	os::raw::c_char,
	path::Path,
	ptr,
	rc::Rc,
	sync::{atomic::Ordering, Arc}
};
#[cfg(feature = "fetch-models")]
use std::{path::PathBuf, time::Duration};

use compact_str::CompactString;

#[cfg(feature = "fetch-models")]
use super::error::FetchModelError;
use super::{
	char_p_to_string,
	environment::get_environment,
	error::{assert_non_null_pointer, assert_null_pointer, status_to_result, Error, ErrorInternal, Result},
	execution_providers::{apply_execution_providers, ExecutionProviderDispatch},
	extern_system_fn,
	io_binding::IoBinding,
	memory::Allocator,
	metadata::ModelMetadata,
	ortsys,
	value::{Value, ValueType},
	GraphOptimizationLevel
};
use crate::{environment::Environment, operator::OperatorDomain, MemoryInfo};

pub(crate) mod input;
pub(crate) mod output;
pub use self::{input::SessionInputs, output::SessionOutputs};

/// Type used to create a session using the _builder pattern_. Once created with [`Session::builder`], you can use the
/// different methods to configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file`](crate::SessionBuilder::with_model_from_file)
/// method to "commit" the builder configuration into a [`Session`].
///
/// # Example
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let mut session = Session::builder()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
	pub(crate) session_options_ptr: *mut ort_sys::OrtSessionOptions,
	memory_info: Option<Rc<MemoryInfo>>,
	#[cfg(feature = "custom-ops")]
	custom_runtime_handles: Vec<*mut std::os::raw::c_void>,
	custom_op_domains: Vec<Rc<OperatorDomain>>,
	execution_providers: Vec<ExecutionProviderDispatch>
}

impl fmt::Debug for SessionBuilder {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		f.debug_struct("SessionBuilder").field("memory_info", &self.memory_info).finish()
	}
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		status_to_result(ortsys![unsafe CloneSessionOptions(self.session_options_ptr, &mut session_options_ptr as *mut _)])
			.expect("error cloning session options");
		Self {
			session_options_ptr,
			memory_info: self.memory_info.clone(),
			#[cfg(feature = "custom-ops")]
			custom_runtime_handles: self.custom_runtime_handles.clone(),
			custom_op_domains: self.custom_op_domains.clone(),
			execution_providers: self.execution_providers.clone()
		}
	}
}

impl Drop for SessionBuilder {
	#[tracing::instrument]
	fn drop(&mut self) {
		#[cfg(feature = "custom-ops")]
		for &handle in self.custom_runtime_handles.iter() {
			close_lib_handle(handle);
		}

		if !self.session_options_ptr.is_null() {
			ortsys![unsafe ReleaseSessionOptions(self.session_options_ptr)];
		}
	}
}

impl SessionBuilder {
	/// Creates a new session builder in the given environment.
	pub fn new() -> Result<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr) -> Error::CreateSessionOptions; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr,
			memory_info: None,
			#[cfg(feature = "custom-ops")]
			custom_runtime_handles: Vec::new(),
			custom_op_domains: Vec::new(),
			execution_providers: Vec::new()
		})
	}

	/// Configures a list of execution providers to attempt to use for the session.
	///
	/// Execution providers are loaded in the order they are provided until a suitable execution provider is found. Most
	/// execution providers will silently fail if they are unavailable or misconfigured (see notes below), however, some
	/// may log to the console, which is sadly unavoidable. The CPU execution provider is always available, so always
	/// put it last in the list (though it is not required).
	///
	/// Execution providers will only work if the corresponding Cargo feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled are currently ignored.
	///
	/// Execution provider options can be specified in the second argument. Refer to ONNX Runtime's
	/// [execution provider docs](https://onnxruntime.ai/docs/execution-providers/) for configuration options. In most
	/// cases, passing `None` to configure with no options is suitable.
	///
	/// It is recommended to enable the `cuda` EP for x86 platforms and the `acl` EP for ARM platforms for the best
	/// performance, though this does mean you'll have to build ONNX Runtime for these targets. Microsoft's prebuilt
	/// binaries are built with CUDA and TensorRT support, if you built `ort` with the `cuda` or `tensorrt` features
	/// enabled.
	///
	/// Supported execution providers:
	/// - `cpu`: Default CPU/MLAS execution provider. Available on all platforms.
	/// - `acl`: Arm Compute Library
	/// - `cuda`: NVIDIA CUDA/cuDNN
	/// - `tensorrt`: NVIDIA TensorRT
	///
	/// ## Notes
	///
	/// - **Use of [`SessionBuilder::with_execution_providers`] in a library is discouraged.** Execution providers
	///   should always be configurable by the user, in case an execution provider is misconfigured and causes the
	///   application to crash (see notes below). Instead, your library should accept an [`Environment`] from the user
	///   rather than creating its own. This way, the user can configure execution providers for **all** modules that
	///   use it.
	/// - Using the CUDA/TensorRT execution providers **can terminate the process if the CUDA/TensorRT installation is
	///   misconfigured**. Configuring the execution provider will seem to work, but when you attempt to run a session,
	///   it will hard crash the process with a "stack buffer overrun" error. This can occur when CUDA/TensorRT is
	///   missing a DLL such as `zlibwapi.dll`. To prevent your app from crashing, you can check to see if you can load
	///   `zlibwapi.dll` before enabling the CUDA/TensorRT execution providers.
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> Result<Self> {
		self.execution_providers = execution_providers.as_ref().to_vec();
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution within nodes. If ONNX Runtime was
	/// built with OpenMP (as is the case with Microsoft's prebuilt binaries), this will have no effect on the number of
	/// threads used. Instead, you can configure the number of threads OpenMP uses via the `OMP_NUM_THREADS` environment
	/// variable.
	///
	/// For configuring the number of threads used when the session execution mode is set to `Parallel`, see
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_intra_threads(self, num_threads: i16) -> Result<Self> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetIntraOpNumThreads(self.session_options_ptr, num_threads) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to disable per-session thread pool, instead using the environment's global thread pool.
	/// This must be used with an environment created with
	/// [`EnvironmentBuilder::with_global_thread_pool`](crate::environment::EnvironmentBuilder::with_global_thread_pool)
	/// enabled.
	pub fn with_disable_per_session_threads(self) -> Result<Self> {
		ortsys![unsafe DisablePerSessionThreads(self.session_options_ptr) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution of the graph. If nodes can be run
	/// in parallel, this sets the maximum number of threads to use to run them in parallel.
	///
	/// This has no effect when the session execution mode is set to `Sequential`.
	///
	/// For configuring the number of threads used to parallelize the execution within nodes, see
	/// [`SessionBuilder::with_intra_threads()`].
	pub fn with_inter_threads(self, num_threads: i16) -> Result<Self> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetInterOpNumThreads(self.session_options_ptr, num_threads) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Enable/disable the parallel execution mode for this session. By default, this is disabled.
	///
	/// Parallel execution can improve performance for models with many branches, at the cost of higher memory usage.
	/// You can configure the amount of threads used to parallelize the execution of the graph via
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_parallel_execution(self, parallel_execution: bool) -> Result<Self> {
		let execution_mode = if parallel_execution {
			ort_sys::ExecutionMode::ORT_PARALLEL
		} else {
			ort_sys::ExecutionMode::ORT_SEQUENTIAL
		};
		ortsys![unsafe SetSessionExecutionMode(self.session_options_ptr, execution_mode) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(self, opt_level: GraphOptimizationLevel) -> Result<Self> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.session_options_ptr, opt_level.into()) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Enables profiling. Profile information will be writen to `profiling_file` after profiling completes.
	/// See [`Session::end_profiling`].
	#[cfg(feature = "profiling")]
	#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
	pub fn with_profiling<S: AsRef<str>>(self, profiling_file: S) -> Result<Self> {
		#[cfg(windows)]
		let profiling_file = widestring::WideCString::from_str(profiling_file.as_ref())?;
		#[cfg(not(windows))]
		let profiling_file = CString::new(profiling_file.as_ref())?;
		ortsys![unsafe EnableProfiling(self.session_options_ptr, profiling_file.as_ptr()) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Enables/disables memory pattern optimization. Disable it if the input size varies, i.e., dynamic batch
	pub fn with_memory_pattern(self, enable: bool) -> Result<Self> {
		if enable {
			ortsys![unsafe EnableMemPattern(self.session_options_ptr) -> Error::CreateSessionOptions];
		} else {
			ortsys![unsafe DisableMemPattern(self.session_options_ptr) -> Error::CreateSessionOptions];
		}
		Ok(self)
	}

	/// Set the session's allocator options from a [`MemoryInfo`].
	///
	/// If not provided, the session is created using ONNX Runtime's default device allocator.
	pub fn with_allocator(mut self, info: MemoryInfo) -> Result<Self> {
		self.memory_info = Some(Rc::new(info));
		Ok(self)
	}

	/// Registers a custom operator library with the given library path in the session.
	#[cfg(feature = "custom-ops")]
	#[cfg_attr(docsrs, doc(cfg(feature = "custom-ops")))]
	pub fn with_custom_ops_lib(mut self, lib_path: impl AsRef<str>) -> Result<Self> {
		let path_cstr = CString::new(lib_path.as_ref())?;

		let mut handle: *mut ::std::os::raw::c_void = std::ptr::null_mut();

		let status = ortsys![unsafe RegisterCustomOpsLibrary(self.session_options_ptr, path_cstr.as_ptr(), &mut handle)];

		// per RegisterCustomOpsLibrary docs, release handle if there was an error and the handle
		// is non-null
		if let Err(e) = status_to_result(status).map_err(Error::CreateSessionOptions) {
			if !handle.is_null() {
				// handle was written to, should release it
				close_lib_handle(handle);
			}

			return Err(e);
		}

		self.custom_runtime_handles.push(handle);

		Ok(self)
	}

	/// Enable custom operators. See onnxruntime-extensions: https://github.com/microsoft/onnxruntime-extensions
	#[cfg(feature = "custom-ops")]
	#[cfg_attr(docsrs, doc(cfg(feature = "custom-ops")))]
	pub fn with_enable_custom_ops(self) -> Result<Self> {
		ortsys![unsafe EnableOrtCustomOps(self.session_options_ptr) -> Error::CreateSessionOptions];
		Ok(self)
	}

	pub fn with_operators(mut self, domain: OperatorDomain) -> Result<Self> {
		ortsys![unsafe AddCustomOpDomain(self.session_options_ptr, domain.ptr.as_ptr()) -> Error::AddCustomOperatorDomain];
		self.custom_op_domains.push(Rc::new(domain));

		Ok(self)
	}

	/// Downloads a pre-trained ONNX model from the given URL and builds the session.
	#[cfg(feature = "fetch-models")]
	#[cfg_attr(docsrs, doc(cfg(feature = "fetch-models")))]
	pub fn with_model_downloaded(self, model_url: impl AsRef<str>) -> Result<Session> {
		let mut download_dir = ort_sys::internal::dirs::cache_dir()
			.expect("could not determine cache directory")
			.join("models");
		if std::fs::create_dir_all(&download_dir).is_err() {
			download_dir = std::env::current_dir().unwrap();
		}

		let url = model_url.as_ref();
		let model_filename = PathBuf::from(url.split('/').last().unwrap());
		let model_filepath = download_dir.join(model_filename);
		let downloaded_path = if model_filepath.exists() {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), "Model already exists, skipping download");
			model_filepath
		} else {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{:?}", url).as_str(), "Downloading model");

			let resp = ureq::get(url).call().map_err(Box::new).map_err(FetchModelError::FetchError)?;

			assert!(resp.has("Content-Length"));
			let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
			tracing::info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_reader();

			let f = std::fs::File::create(&model_filepath).unwrap();
			let mut writer = std::io::BufWriter::new(f);

			let bytes_io_count = std::io::copy(&mut reader, &mut writer).map_err(FetchModelError::IoError)?;
			if bytes_io_count == len as u64 {
				model_filepath
			} else {
				return Err(FetchModelError::CopyError {
					expected: len as u64,
					io: bytes_io_count
				}
				.into());
			}
		};

		self.with_model_from_file(downloaded_path)
	}

	#[cfg(feature = "fetch-models")]
	#[tracing::instrument]
	fn download_to<P>(&self, url: &str, download_dir: P) -> Result<PathBuf>
	where
		P: AsRef<Path> + fmt::Debug
	{
		let model_filename = PathBuf::from(url.split('/').last().unwrap());
		let model_filepath = download_dir.as_ref().join(model_filename);
		if model_filepath.exists() {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), "Model already exists, skipping download");
			Ok(model_filepath)
		} else {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{:?}", url).as_str(), "Downloading model");

			let resp = ureq::get(url)
				.timeout(Duration::from_secs(180))
				.call()
				.map_err(Box::new)
				.map_err(FetchModelError::FetchError)?;

			assert!(resp.has("Content-Length"));
			let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
			tracing::info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_reader();

			let f = std::fs::File::create(&model_filepath).unwrap();
			let mut writer = std::io::BufWriter::new(f);

			let bytes_io_count = std::io::copy(&mut reader, &mut writer).map_err(FetchModelError::IoError)?;
			if bytes_io_count == len as u64 {
				Ok(model_filepath)
			} else {
				Err(FetchModelError::CopyError {
					expected: len as u64,
					io: bytes_io_count
				}
				.into())
			}
		}
	}

	// TODO: Add all functions changing the options.
	//       See all OrtApi methods taking a `options: *mut OrtSessionOptions`.

	/// Loads an ONNX model from a file and builds the session.
	pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> Result<Session>
	where
		P: AsRef<Path>
	{
		let model_filepath = model_filepath_ref.as_ref();
		if !model_filepath.exists() {
			return Err(Error::FileDoesNotExist {
				filename: model_filepath.to_path_buf()
			});
		}

		// Build an OsString, then a vector of bytes to pass to C
		let model_path = std::ffi::OsString::from(model_filepath);
		#[cfg(target_family = "windows")]
		let model_path: Vec<u16> = model_path
            .encode_wide()
            .chain(std::iter::once(0)) // Make sure we have a null terminated string
            .collect();
		#[cfg(not(target_family = "windows"))]
		let model_path: Vec<std::os::raw::c_char> = model_path
            .as_bytes()
            .iter()
            .chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
            .map(|b| *b as std::os::raw::c_char)
            .collect();

		let env = get_environment()?;
		apply_execution_providers(&self, self.execution_providers.iter().chain(&env.execution_providers).cloned());

		let env_ptr = env.env_ptr.load(Ordering::Relaxed);

		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> Error::CreateSession; nonNull(session_ptr)];

		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
				ortsys![unsafe CreateAllocator(session_ptr, info.ptr, &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
				Allocator::from_raw(allocator_ptr)
			}
			None => Allocator::default()
		};

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator.ptr, i))
			.collect::<Result<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator.ptr, i))
			.collect::<Result<Vec<Output>>>()?;

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr,
				allocator,
				_environment: Arc::clone(env)
			}),
			inputs,
			outputs
		})
	}

	/// Load an ONNX graph from memory and commit the session
	/// For `.ort` models, we enable `session.use_ort_model_bytes_directly`.
	/// For more information, check [Load ORT format model from an in-memory byte array](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#load-ort-format-model-from-an-in-memory-byte-array).
	///
	/// If you wish to store the model bytes and the [`InMemorySession`] in the same struct, look for crates that
	/// facilitate creating self-referential structs, such as [`ouroboros`](https://github.com/joshua-maros/ouroboros).
	pub fn with_model_from_memory_directly(self, model_bytes: &[u8]) -> Result<InMemorySession<'_>> {
		let str_to_char = |s: &str| {
			s.as_bytes()
				.iter()
				.chain(std::iter::once(&b'\0')) // Make sure we have a null terminated string
				.map(|b| *b as std::os::raw::c_char)
				.collect::<Vec<std::os::raw::c_char>>()
		};
		// Enable zero-copy deserialization for models in `.ort` format.
		ortsys![unsafe AddSessionConfigEntry(self.session_options_ptr, str_to_char("session.use_ort_model_bytes_directly").as_ptr(), str_to_char("1").as_ptr())];
		ortsys![unsafe AddSessionConfigEntry(self.session_options_ptr, str_to_char("session.use_ort_model_bytes_for_initializers").as_ptr(), str_to_char("1").as_ptr())];

		let session = self.with_model_from_memory(model_bytes)?;

		Ok(InMemorySession { session, phantom: PhantomData })
	}

	/// Load an ONNX graph from memory and commit the session.
	pub fn with_model_from_memory(self, model_bytes: &[u8]) -> Result<Session> {
		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();

		let env = get_environment()?;
		apply_execution_providers(&self, self.execution_providers.iter().chain(&env.execution_providers).cloned());

		let env_ptr = env.env_ptr.load(Ordering::Relaxed);

		let model_data = model_bytes.as_ptr() as *const std::ffi::c_void;
		let model_data_length = model_bytes.len();
		ortsys![
			unsafe CreateSessionFromArray(env_ptr, model_data, model_data_length as _, self.session_options_ptr, &mut session_ptr) -> Error::CreateSession;
			nonNull(session_ptr)
		];

		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
				ortsys![unsafe CreateAllocator(session_ptr, info.ptr, &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
				Allocator::from_raw(allocator_ptr)
			}
			None => Allocator::default()
		};

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator.ptr, i))
			.collect::<Result<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator.ptr, i))
			.collect::<Result<Vec<Output>>>()?;

		let session = Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr,
				allocator,
				_environment: Arc::clone(env)
			}),
			inputs,
			outputs
		};
		Ok(session)
	}
}

/// Holds onto a C session and its allocator. This is wrapped in an [`Arc`] to ensure that [`Value`]s returned by the
/// session keep their memory alive until all references to the session are dropped.
#[derive(Debug)]
pub struct SharedSessionInner {
	pub(crate) session_ptr: *mut ort_sys::OrtSession,
	allocator: Allocator,
	_environment: Arc<Environment>
}

unsafe impl Send for SharedSessionInner {}
unsafe impl Sync for SharedSessionInner {}

impl Drop for SharedSessionInner {
	#[tracing::instrument]
	fn drop(&mut self) {
		tracing::debug!("dropping SharedSessionInner");
		if !self.session_ptr.is_null() {
			tracing::debug!("dropping session ptr");
			ortsys![unsafe ReleaseSession(self.session_ptr)];
		}
		self.session_ptr = std::ptr::null_mut();
	}
}

/// Type storing the session information, built from an [`Environment`](crate::environment::Environment)
#[derive(Debug)]
pub struct Session {
	pub(crate) inner: Arc<SharedSessionInner>,
	/// Information about the ONNX's inputs as stored in loaded file
	pub inputs: Vec<Input>,
	/// Information about the ONNX's outputs as stored in loaded file
	pub outputs: Vec<Output>
}

/// A [`Session`] with data stored in-memory.
pub struct InMemorySession<'s> {
	session: Session,
	phantom: PhantomData<&'s ()>
}

impl<'s> Deref for InMemorySession<'s> {
	type Target = Session;
	fn deref(&self) -> &Self::Target {
		&self.session
	}
}

/// Information about an ONNX's input as stored in loaded file
#[derive(Debug)]
pub struct Input {
	/// Name of the input layer
	pub name: String,
	/// Type of the input layer's elements
	pub input_type: ValueType
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
	/// Name of the output layer
	pub name: String,
	/// Type of the output layer's elements
	pub output_type: ValueType
}

/// ONNX Run Options which is used to terminate/unterminate run(s) in a session
#[derive(Debug)]
pub struct RunOptions {
	pub(crate) run_options_ptr: *mut ort_sys::OrtRunOptions
}

// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ac2a08cac0a657604bd5899e0d1a13675
unsafe impl Send for RunOptions {}
unsafe impl Sync for RunOptions {}

impl RunOptions {
	/// Creates a new [`RunOptions`].
	pub fn new() -> Result<Self> {
		let mut run_options_ptr: *mut ort_sys::OrtRunOptions = std::ptr::null_mut();
		ortsys![unsafe CreateRunOptions(&mut run_options_ptr) -> Error::CreateRunOptions; nonNull(run_options_ptr)];
		Ok(Self { run_options_ptr })
	}

	/// Terminates the runs associated with [`RunOptions`].
	pub fn set_terminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsSetTerminate(self.run_options_ptr) -> Error::RunOptionsSetTerminate];
		Ok(())
	}

	/// Unterminates the runs associated with [`RunOptions`].
	pub fn set_unterminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsUnsetTerminate(self.run_options_ptr) -> Error::RunOptionsUnsetTerminate];
		Ok(())
	}
}

impl Drop for RunOptions {
	fn drop(&mut self) {
		if !self.run_options_ptr.is_null() {
			ortsys![unsafe ReleaseRunOptions(self.run_options_ptr)];
		}
		self.run_options_ptr = std::ptr::null_mut();
	}
}

impl Session {
	pub fn builder() -> Result<SessionBuilder> {
		SessionBuilder::new()
	}

	/// Returns this session's [`Allocator`].
	pub fn allocator(&self) -> &Allocator {
		&self.inner.allocator
	}

	/// Creates a new [`IoBinding`] for this session.
	pub fn create_binding(&self) -> Result<IoBinding> {
		IoBinding::new(self)
	}

	/// Get an [`Arc`] reference to the underlying [`SharedSessionInner`], containing the C session and allocator.
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}

	/// Run the input data through the ONNX graph, performing inference.
	pub fn run<'s, 'i, const N: usize>(&'s self, input_values: impl Into<SessionInputs<'i, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				let outputs = self.run_inner(
					&self
						.inputs
						.iter()
						.map(|input| CompactString::new(input.name.as_str()))
						.collect::<Vec<_>>(),
					input_values,
					None
				)?;
				Ok(outputs)
			}
			SessionInputs::ValueArray(input_values) => {
				let outputs = self.run_inner(
					&self
						.inputs
						.iter()
						.map(|input| CompactString::new(input.name.as_str()))
						.collect::<Vec<_>>(),
					&input_values,
					None
				)?;
				Ok(outputs)
			}
			SessionInputs::ValueMap(input_values) => {
				let (input_names, values): (Vec<CompactString>, Vec<Value>) = input_values.into_iter().unzip();
				self.run_inner(&input_names, &values, None)
			}
		}
	}

	/// Run the input data through the ONNX graph, performing inference.
	pub fn run_with_options<'s, 'i, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, N>>,
		run_options: Arc<RunOptions>
	) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				let outputs = self.run_inner(
					&self
						.inputs
						.iter()
						.map(|input| CompactString::new(input.name.as_str()))
						.collect::<Vec<_>>(),
					input_values,
					Some(run_options)
				)?;
				Ok(outputs)
			}
			SessionInputs::ValueArray(input_values) => {
				let outputs = self.run_inner(
					&self
						.inputs
						.iter()
						.map(|input| CompactString::new(input.name.as_str()))
						.collect::<Vec<_>>(),
					&input_values,
					Some(run_options)
				)?;
				Ok(outputs)
			}
			SessionInputs::ValueMap(input_values) => {
				let (input_names, values): (Vec<CompactString>, Vec<Value>) = input_values.into_iter().unzip();
				self.run_inner(&input_names, &values, Some(run_options))
			}
		}
	}

	fn run_inner(&self, input_names: &[CompactString], input_values: &[Value], run_options: Option<Arc<RunOptions>>) -> Result<SessionOutputs<'_>> {
		let input_names_ptr: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap())
			.map(|n| n.into_raw() as *const c_char)
			.collect();
		let output_names_ptr: Vec<*const c_char> = self
			.outputs
			.iter()
			.map(|output| CString::new(output.name.as_str()).unwrap())
			.map(|n| n.into_raw() as *const c_char)
			.collect();

		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.iter().map(|input_array_ort| input_array_ort.ptr() as *const _).collect();

		let run_options_ptr = if let Some(run_options) = &run_options {
			run_options.run_options_ptr
		} else {
			std::ptr::null_mut()
		};

		ortsys![
			unsafe Run(
				self.inner.session_ptr,
				run_options_ptr,
				input_names_ptr.as_ptr(),
				input_ort_values.as_ptr(),
				input_ort_values.len() as _,
				output_names_ptr.as_ptr(),
				output_names_ptr.len() as _,
				output_tensor_ptrs.as_mut_ptr()
			) -> Error::SessionRun
		];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe { Value::from_raw(tensor_ptr, Arc::clone(&self.inner)) })
			.collect();

		// Reconvert name ptrs to CString so drop impl is called and memory is freed
		drop(
			input_names_ptr
				.into_iter()
				.chain(output_names_ptr.into_iter())
				.map(|p| {
					assert_non_null_pointer(p, "c_char for CString")?;
					unsafe { Ok(CString::from_raw(p as *mut c_char)) }
				})
				.collect::<Result<Vec<_>>>()?
		);

		Ok(SessionOutputs::new(self.outputs.iter().map(|o| o.name.as_str()), outputs))
	}

	/// Gets the session model metadata. See [`ModelMetadata`] for more info.
	pub fn metadata(&self) -> Result<ModelMetadata> {
		let mut metadata_ptr: *mut ort_sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.inner.session_ptr, &mut metadata_ptr) -> Error::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(ModelMetadata::new(metadata_ptr, self.inner.allocator.ptr))
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiing file will be empty.
	#[cfg(feature = "profiling")]
	#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
	pub fn end_profiling(&self) -> Result<String> {
		let mut profiling_name: *mut c_char = std::ptr::null_mut();

		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr, self.inner.allocator.ptr, &mut profiling_name)];
		assert_non_null_pointer(profiling_name, "ProfilingName")?;
		dangerous::raw_pointer_to_string(self.inner.allocator.ptr, profiling_name)
	}
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

#[cfg(all(unix, feature = "custom-ops"))]
fn close_lib_handle(handle: *mut std::os::raw::c_void) {
	unsafe { libc::dlclose(handle) };
}

#[cfg(all(windows, feature = "custom-ops"))]
fn close_lib_handle(handle: *mut std::os::raw::c_void) {
	unsafe { winapi::um::libloaderapi::FreeLibrary(handle as winapi::shared::minwindef::HINSTANCE) };
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
	use super::*;
	use crate::{
		ortfree,
		value::{extract_data_type_from_map_info, extract_data_type_from_sequence_info, extract_data_type_from_tensor_info}
	};

	pub(super) fn extract_inputs_count(session_ptr: *mut ort_sys::OrtSession) -> Result<usize> {
		let f = ortsys![unsafe SessionGetInputCount];
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: *mut ort_sys::OrtSession) -> Result<usize> {
		let f = ortsys![unsafe SessionGetOutputCount];
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const ort_sys::OrtSession, *mut ort_sys::size_t) -> *mut ort_sys::OrtStatus },
		session_ptr: *mut ort_sys::OrtSession
	) -> Result<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr, &mut num_nodes) };
		status_to_result(status).map_err(Error::GetInOutCount)?;
		assert_null_pointer(status, "SessionStatus")?;
		(num_nodes != 0)
			.then_some(())
			.ok_or_else(|| Error::GetInOutCount(ErrorInternal::Msg("No nodes in model".to_owned())))?;
		Ok(num_nodes as _)
	}

	fn extract_input_name(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetInputName];
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	fn extract_output_name(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetOutputName];
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	pub(crate) fn raw_pointer_to_string(allocator_ptr: *mut ort_sys::OrtAllocator, c_str: *mut c_char) -> Result<String> {
		let name = match char_p_to_string(c_str) {
			Ok(name) => name,
			Err(e) => {
				ortfree!(unsafe allocator_ptr, c_str);
				return Err(e);
			}
		};
		ortfree!(unsafe allocator_ptr, c_str);
		Ok(name)
	}

	fn extract_io_name(
		f: extern_system_fn! { unsafe fn(
			*const ort_sys::OrtSession,
			ort_sys::size_t,
			*mut ort_sys::OrtAllocator,
			*mut *mut c_char,
		) -> *mut ort_sys::OrtStatus },
		session_ptr: *mut ort_sys::OrtSession,
		allocator_ptr: *mut ort_sys::OrtAllocator,
		i: ort_sys::size_t
	) -> Result<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
		status_to_result(status).map_err(Error::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		raw_pointer_to_string(allocator_ptr, name_bytes)
	}

	pub(super) fn extract_input(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: usize) -> Result<Input> {
		let input_name = extract_input_name(session_ptr, allocator_ptr, i as _)?;
		let f = ortsys![unsafe SessionGetInputTypeInfo];
		let input_type = extract_io(f, session_ptr, i as _)?;
		Ok(Input { name: input_name, input_type })
	}

	pub(super) fn extract_output(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: usize) -> Result<Output> {
		let output_name = extract_output_name(session_ptr, allocator_ptr, i as _)?;
		let f = ortsys![unsafe SessionGetOutputTypeInfo];
		let output_type = extract_io(f, session_ptr, i as _)?;
		Ok(Output { name: output_name, output_type })
	}

	fn extract_io(
		f: extern_system_fn! { unsafe fn(
			*const ort_sys::OrtSession,
			ort_sys::size_t,
			*mut *mut ort_sys::OrtTypeInfo,
		) -> *mut ort_sys::OrtStatus },
		session_ptr: *mut ort_sys::OrtSession,
		i: ort_sys::size_t
	) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
		status_to_result(status).map_err(Error::GetTypeInfo)?;
		assert_non_null_pointer(typeinfo_ptr, "TypeInfo")?;

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty)];
		status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToSequenceTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_sequence_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr)? }
			}
			_ => unreachable!()
		};

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		Ok(io_type)
	}
}
