//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

#![allow(clippy::tabs_in_doc_comments)]

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
#[cfg(feature = "fetch-models")]
use std::{env, path::PathBuf, time::Duration};
use std::{ffi::CString, fmt, marker::PhantomData, ops::Deref, os::raw::c_char, path::Path, ptr, sync::Arc};

use super::{
	char_p_to_string,
	environment::Environment,
	error::{assert_non_null_pointer, assert_null_pointer, status_to_result, OrtApiError, OrtError, OrtResult},
	execution_providers::{apply_execution_providers, ExecutionProvider},
	extern_system_fn,
	io_binding::IoBinding,
	memory::Allocator,
	metadata::ModelMetadata,
	ort, ortsys,
	tensor::DataType,
	value::Value,
	AllocatorType, GraphOptimizationLevel, MemType
};
#[cfg(feature = "fetch-models")]
use super::{download::ModelUrl, error::OrtDownloadError};

pub(crate) mod input;
pub(crate) mod output;
pub use self::{input::SessionInputs, output::SessionOutputs};

/// Type used to create a session using the _builder pattern_. Once created, you can use the different methods to
/// configure the session.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file`](crate::SessionBuilder::with_model_from_file)
/// method to "commit" the builder configuration into a [`Session`].
///
/// # Example
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ort::{Environment, LoggingLevel, GraphOptimizationLevel, SessionBuilder};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder()
/// 	.with_name("test")
/// 	.with_log_level(LoggingLevel::Verbose)
/// 	.build()?
/// 	.into_arc();
/// let mut session = SessionBuilder::new(&environment)?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.with_model_from_file("squeezenet.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
	env: Arc<Environment>,
	session_options_ptr: *mut ort_sys::OrtSessionOptions,

	allocator: AllocatorType,
	memory_type: MemType,
	custom_runtime_handles: Vec<*mut std::os::raw::c_void>,
	execution_providers: Vec<ExecutionProvider>
}

impl fmt::Debug for SessionBuilder {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		f.debug_struct("SessionBuilder")
			.field("env", &self.env.name())
			.field("allocator", &self.allocator)
			.field("memory_type", &self.memory_type)
			.finish()
	}
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		status_to_result(ortsys![unsafe CloneSessionOptions(self.session_options_ptr, &mut session_options_ptr as *mut _)])
			.expect("error cloning session options");
		Self {
			env: Arc::clone(&self.env),
			session_options_ptr,
			allocator: self.allocator,
			memory_type: self.memory_type,
			custom_runtime_handles: self.custom_runtime_handles.clone(),
			execution_providers: self.execution_providers.clone()
		}
	}
}

impl Drop for SessionBuilder {
	#[tracing::instrument]
	fn drop(&mut self) {
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
	pub fn new(env: &Arc<Environment>) -> OrtResult<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr) -> OrtError::CreateSessionOptions; nonNull(session_options_ptr)];

		Ok(Self {
			env: Arc::clone(env),
			session_options_ptr,
			allocator: AllocatorType::Device,
			memory_type: MemType::Default,
			custom_runtime_handles: Vec::new(),
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
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProvider]>) -> OrtResult<Self> {
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
	pub fn with_intra_threads(self, num_threads: i16) -> OrtResult<Self> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetIntraOpNumThreads(self.session_options_ptr, num_threads) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to disable per-session thread pool, instead using the environment's global thread pool.
	/// This must be used with an environment created with
	/// [`EnvironmentBuilder::with_global_thread_pool`](crate::environment::EnvironmentBuilder::with_global_thread_pool)
	/// enabled.
	pub fn with_disable_per_session_threads(self) -> OrtResult<Self> {
		ortsys![unsafe DisablePerSessionThreads(self.session_options_ptr) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution of the graph. If nodes can be run
	/// in parallel, this sets the maximum number of threads to use to run them in parallel.
	///
	/// This has no effect when the session execution mode is set to `Sequential`.
	///
	/// For configuring the number of threads used to parallelize the execution within nodes, see
	/// [`SessionBuilder::with_intra_threads()`].
	pub fn with_inter_threads(self, num_threads: i16) -> OrtResult<Self> {
		// We use a u16 in the builder to cover the 16-bits positive values of a i32.
		let num_threads = num_threads as i32;
		ortsys![unsafe SetInterOpNumThreads(self.session_options_ptr, num_threads) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Enable/disable the parallel execution mode for this session. By default, this is disabled.
	///
	/// Parallel execution can improve performance for models with many branches, at the cost of higher memory usage.
	/// You can configure the amount of threads used to parallelize the execution of the graph via
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_parallel_execution(self, parallel_execution: bool) -> OrtResult<Self> {
		let execution_mode = if parallel_execution {
			ort_sys::ExecutionMode::ORT_PARALLEL
		} else {
			ort_sys::ExecutionMode::ORT_SEQUENTIAL
		};
		ortsys![unsafe SetSessionExecutionMode(self.session_options_ptr, execution_mode) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(self, opt_level: GraphOptimizationLevel) -> OrtResult<Self> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.session_options_ptr, opt_level.into()) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Enables profiling. Profile information will be writen to `profiling_file` after profiling completes.
	/// See `Session::end_profiling`.
	#[cfg(feature = "profiling")]
	pub fn with_profiling<S: AsRef<str>>(self, profiling_file: S) -> OrtResult<Self> {
		#[cfg(windows)]
		let profiling_file = widestring::WideCString::from_str(profiling_file.as_ref())?;
		#[cfg(not(windows))]
		let profiling_file = CString::new(profiling_file.as_ref())?;
		ortsys![unsafe EnableProfiling(self.session_options_ptr, profiling_file.as_ptr()) -> OrtError::CreateSessionOptions];
		Ok(self)
	}

	/// Enables/disables memory pattern optimization. Disable it if the input size varies, i.e., dynamic batch
	pub fn with_memory_pattern(self, enable: bool) -> OrtResult<Self> {
		if enable {
			ortsys![unsafe EnableMemPattern(self.session_options_ptr) -> OrtError::CreateSessionOptions];
		} else {
			ortsys![unsafe DisableMemPattern(self.session_options_ptr) -> OrtError::CreateSessionOptions];
		}
		Ok(self)
	}

	/// Set the session's allocator. Defaults to [`AllocatorType::Device`].
	pub fn with_allocator(mut self, allocator: AllocatorType) -> OrtResult<Self> {
		self.allocator = allocator;
		Ok(self)
	}

	/// Set the session's memory type. Defaults to [`MemType::Default`].
	pub fn with_memory_type(mut self, memory_type: MemType) -> OrtResult<Self> {
		self.memory_type = memory_type;
		Ok(self)
	}

	/// Registers a custom operator library with the given library path in the session.
	pub fn with_custom_op_lib(mut self, lib_path: impl AsRef<str>) -> OrtResult<Self> {
		let path_cstr = CString::new(lib_path.as_ref())?;

		let mut handle: *mut ::std::os::raw::c_void = std::ptr::null_mut();

		let status = ortsys![unsafe RegisterCustomOpsLibrary(self.session_options_ptr, path_cstr.as_ptr(), &mut handle)];

		// per RegisterCustomOpsLibrary docs, release handle if there was an error and the handle
		// is non-null
		if let Err(e) = status_to_result(status).map_err(OrtError::CreateSessionOptions) {
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
	pub fn with_enable_ort_custom_ops(self) -> OrtResult<Self> {
		let status = ortsys![unsafe EnableOrtCustomOps(self.session_options_ptr)];
		status_to_result(status).map_err(OrtError::CreateSessionOptions)?;
		Ok(self)
	}

	/// Downloads a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models) and builds the session.
	#[cfg(feature = "fetch-models")]
	pub fn with_model_downloaded<M>(self, model: M) -> OrtResult<Session>
	where
		M: ModelUrl
	{
		self.with_model_downloaded_monomorphized(model.fetch_url())
	}

	#[cfg(feature = "fetch-models")]
	fn with_model_downloaded_monomorphized(self, model: &str) -> OrtResult<Session> {
		let download_dir = env::current_dir().map_err(OrtDownloadError::IoError)?;
		let downloaded_path = self.download_to(model, download_dir)?;
		self.with_model_from_file(downloaded_path)
	}

	#[cfg(feature = "fetch-models")]
	#[tracing::instrument]
	fn download_to<P>(&self, url: &str, download_dir: P) -> OrtResult<PathBuf>
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
				.map_err(OrtDownloadError::FetchError)?;

			assert!(resp.has("Content-Length"));
			let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
			tracing::info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_reader();

			let f = std::fs::File::create(&model_filepath).unwrap();
			let mut writer = std::io::BufWriter::new(f);

			let bytes_io_count = std::io::copy(&mut reader, &mut writer).map_err(OrtDownloadError::IoError)?;
			if bytes_io_count == len as u64 {
				Ok(model_filepath)
			} else {
				Err(OrtDownloadError::CopyError {
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
	pub fn with_model_from_file<P>(self, model_filepath_ref: P) -> OrtResult<Session>
	where
		P: AsRef<Path>
	{
		let model_filepath = model_filepath_ref.as_ref();
		if !model_filepath.exists() {
			return Err(OrtError::FileDoesNotExist {
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

		apply_execution_providers(
			self.session_options_ptr,
			self.execution_providers
				.iter()
				.chain(&self.env.execution_providers)
				.cloned()
				.collect::<Vec<_>>()
		);

		let env_ptr: *const ort_sys::OrtEnv = self.env.ptr();

		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession; nonNull(session_ptr)];

		let allocator = Allocator::default();

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				env: Arc::clone(&self.env),
				session_ptr,
				allocator
			}),
			inputs,
			outputs
		})
	}

	/// Loads an ONNX model from a file, replacing external data with data provided in initializers.
	///
	/// This will find initialized tensors with external data in the graph with the provided names and replace them with
	/// the provided tensors. The replacement will occur before any optimizations take place, and the data will be
	/// copied into the graph. Tensors replaced by this function must be using external data. (you cannot replace a
	/// non-external tensor)
	pub fn with_model_from_file_and_external_initializers<'v, 'i, P>(self, model_filepath_ref: P, initializers: &'i [(String, Value)]) -> OrtResult<Session>
	where
		'i: 'v,
		P: AsRef<Path>
	{
		let model_filepath = model_filepath_ref.as_ref();
		if !model_filepath.exists() {
			return Err(OrtError::FileDoesNotExist {
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

		apply_execution_providers(
			self.session_options_ptr,
			self.execution_providers
				.iter()
				.chain(&self.env.execution_providers)
				.cloned()
				.collect::<Vec<_>>()
		);

		let env_ptr: *const ort_sys::OrtEnv = self.env.ptr();

		let allocator = Allocator::default();

		let initializer_names: Vec<CString> = initializers
			.iter()
			.map(|(name, _)| CString::new(name.as_str()).unwrap())
			.map(|n| CString::new(n).unwrap())
			.collect();
		let initializer_names_ptr: Vec<*const c_char> = initializer_names.iter().map(|n| n.as_ptr() as *const c_char).collect();

		let initializers: Vec<*const ort_sys::OrtValue> = initializers.iter().map(|input_array_ort| input_array_ort.1.ptr() as *const _).collect();
		if !initializers.is_empty() {
			assert_eq!(initializer_names.len(), initializers.len());
			ortsys![unsafe AddExternalInitializers(self.session_options_ptr, initializer_names_ptr.as_ptr(), initializers.as_ptr(), initializers.len() as _) -> OrtError::CreateSession];
		}

		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession; nonNull(session_ptr)];

		std::mem::drop(initializer_names);
		std::mem::drop(initializers);

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				env: Arc::clone(&self.env),
				session_ptr,
				allocator
			}),
			inputs,
			outputs
		})
	}

	/// Load an ONNX graph from memory and commit the session
	/// For `.ort` models, we enable `session.use_ort_model_bytes_directly`.
	/// For more information, check [Load ORT format model from an in-memory byte array](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#load-ort-format-model-from-an-in-memory-byte-array).
	/// If you want to store the model file and the [`InMemorySession`] in same struct,
	/// please check crates for creating self-referential structs, such as [`ouroboros`](https://github.com/joshua-maros/ouroboros).
	pub fn with_model_from_memory_directly(self, model_bytes: &[u8]) -> OrtResult<InMemorySession<'_>> {
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
	pub fn with_model_from_memory(self, model_bytes: &[u8]) -> OrtResult<Session> {
		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();

		let env_ptr: *const ort_sys::OrtEnv = self.env.ptr();

		apply_execution_providers(
			self.session_options_ptr,
			self.execution_providers
				.iter()
				.chain(&self.env.execution_providers)
				.cloned()
				.collect::<Vec<_>>()
		);

		let model_data = model_bytes.as_ptr() as *const std::ffi::c_void;
		let model_data_length = model_bytes.len();
		ortsys![
			unsafe CreateSessionFromArray(env_ptr, model_data, model_data_length as _, self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession;
			nonNull(session_ptr)
		];

		let allocator = Allocator::default();

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator.ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		let session = Session {
			inner: Arc::new(SharedSessionInner {
				env: Arc::clone(&self.env),
				session_ptr,
				allocator
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
	// hold onto an environment arc to ensure the environment also stays alive
	#[allow(dead_code)]
	env: Arc<Environment>,
	pub(crate) session_ptr: *mut ort_sys::OrtSession,
	allocator: Allocator
}

unsafe impl Send for SharedSessionInner {}
unsafe impl Sync for SharedSessionInner {}

impl Drop for SharedSessionInner {
	#[tracing::instrument(skip_all)]
	fn drop(&mut self) {
		tracing::warn!("dropping SharedSessionInner");
		if !self.session_ptr.is_null() {
			tracing::warn!("dropping session ptr");
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
	pub input_type: DataType
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
	/// Name of the output layer
	pub name: String,
	/// Type of the output layer's elements
	pub output_type: DataType
}

impl Session {
	/// Returns this session's [`Allocator`].
	pub fn allocator(&self) -> &Allocator {
		&self.inner.allocator
	}

	/// Creates a new [`IoBinding`] for this session.
	pub fn create_binding(&self) -> OrtResult<IoBinding> {
		IoBinding::new(self)
	}

	/// Get an [`Arc`] reference to the underlying [`SharedSessionInner`], containing the C session and allocator.
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}

	/// Run the input data through the ONNX graph, performing inference.
	pub fn run<'s, 'm, 'v, 'i>(&'s self, input_values: impl Into<SessionInputs<'s>>) -> OrtResult<SessionOutputs>
	where
		's: 'm, // 's outlives 'm (session outlives memory info)
		'i: 'v
	{
		let input_values = input_values.into();
		if let SessionInputs::IoBinding(mut binding) = input_values {
			return binding.run();
		}

		match input_values {
			SessionInputs::ValueVec(input_values) => {
				let outputs = self.run_inner(&self.inputs.iter().map(|input| input.name.clone()).collect::<Vec<_>>(), &input_values)?;
				Ok(outputs)
			}
			SessionInputs::ValueMap(input_values) => {
				let (input_names, values): (Vec<&'static str>, Vec<Value>) = input_values.into_iter().unzip();
				self.run_inner(&input_names, &values)
			}
			_ => panic!()
		}
	}

	fn run_inner<'s, 'm, 'v, 'i, I>(&'s self, input_names: &[I], input_values: &[Value]) -> OrtResult<SessionOutputs>
	where
		I: AsRef<str>,
		's: 'm, // 's outlives 'm (session outlives memory info)
		'i: 'v
	{
		let input_names_ptr: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_ref()).unwrap())
			.map(|n| n.into_raw() as *const c_char)
			.collect();
		let output_names: Vec<String> = self.outputs.iter().map(|output| output.name.clone()).collect();
		let output_names_cstring: Vec<CString> = output_names.iter().cloned().map(|n| CString::new(n).unwrap()).collect();
		let output_names_ptr: Vec<*const c_char> = output_names_cstring.iter().map(|n| n.as_ptr() as *const c_char).collect();

		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.iter().map(|input_array_ort| input_array_ort.ptr() as *const _).collect();

		let run_options_ptr: *const ort_sys::OrtRunOptions = std::ptr::null();

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
			) -> OrtError::SessionRun
		];

		std::mem::drop(input_ort_values);

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe { Value::from_raw(tensor_ptr, Arc::clone(&self.inner)) })
			.collect();

		// Reconvert to CString so drop impl is called and memory is freed
		let cstrings: OrtResult<Vec<CString>> = input_names_ptr
			.into_iter()
			.map(|p| {
				assert_non_null_pointer(p, "c_char for CString")?;
				unsafe { Ok(CString::from_raw(p as *mut c_char)) }
			})
			.collect();
		cstrings?;

		Ok(SessionOutputs::new(output_names, outputs))
	}

	/// Gets the session model metadata. See [`Metadata`] for more info.
	pub fn metadata(&self) -> OrtResult<ModelMetadata> {
		let mut metadata_ptr: *mut ort_sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.inner.session_ptr, &mut metadata_ptr) -> OrtError::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(ModelMetadata::new(metadata_ptr, self.inner.allocator.ptr))
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiing file will be empty.
	#[cfg(feature = "profiling")]
	pub fn end_profiling(&self) -> OrtResult<String> {
		let mut profiling_name: *mut c_char = std::ptr::null_mut();

		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr, self.inner.allocator.ptr, &mut profiling_name)];
		assert_non_null_pointer(profiling_name, "ProfilingName")?;
		dangerous::raw_pointer_to_string(self.inner.allocator.ptr, profiling_name)
	}
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

#[cfg(unix)]
fn close_lib_handle(handle: *mut std::os::raw::c_void) {
	unsafe { libc::dlclose(handle) };
}

#[cfg(windows)]
fn close_lib_handle(handle: *mut std::os::raw::c_void) {
	unsafe { winapi::um::libloaderapi::FreeLibrary(handle as winapi::shared::minwindef::HINSTANCE) };
}

/// This module contains dangerous functions working on raw pointers.
/// Those functions are only to be used from inside the
/// `SessionBuilder::with_model_from_file()` method.
mod dangerous {
	use super::*;
	use crate::ortfree;

	unsafe fn extract_data_type_from_tensor_info(info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo) -> OrtResult<DataType> {
		let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		ortsys![GetTensorElementType(info_ptr, &mut type_sys) -> OrtError::GetTensorElementType];
		assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
		// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
		let mut num_dims = 0;
		ortsys![GetDimensionsCount(info_ptr, &mut num_dims) -> OrtError::GetDimensionsCount];

		let mut node_dims: Vec<i64> = vec![0; num_dims as _];
		ortsys![GetDimensions(info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> OrtError::GetDimensions];

		Ok(DataType::Tensor {
			ty: type_sys.into(),
			dimensions: node_dims
		})
	}

	unsafe fn extract_data_type_from_sequence_info(info_ptr: *const ort_sys::OrtSequenceTypeInfo) -> OrtResult<DataType> {
		let mut element_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
		ortsys![GetSequenceElementType(info_ptr, &mut element_type_info) -> OrtError::GetSequenceElementType];

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(element_type_info, &mut ty)];
		status_to_result(status).map_err(OrtError::GetOnnxTypeFromTypeInfo)?;

		match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(element_type_info, &mut info_ptr) -> OrtError::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				let ty = unsafe { extract_data_type_from_tensor_info(info_ptr)? };
				Ok(DataType::Sequence(Box::new(ty)))
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(element_type_info, &mut info_ptr) -> OrtError::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				let ty = unsafe { extract_data_type_from_map_info(info_ptr)? };
				Ok(DataType::Sequence(Box::new(ty)))
			}
			_ => unreachable!()
		}
	}

	unsafe fn extract_data_type_from_map_info(info_ptr: *const ort_sys::OrtMapTypeInfo) -> OrtResult<DataType> {
		let mut key_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		ortsys![GetMapKeyType(info_ptr, &mut key_type_sys) -> OrtError::GetMapKeyType];
		assert_ne!(key_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

		let mut value_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
		ortsys![GetMapValueType(info_ptr, &mut value_type_info) -> OrtError::GetMapValueType];
		let mut value_info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe CastTypeInfoToTensorInfo(value_type_info, &mut value_info_ptr) -> OrtError::CastTypeInfoToTensorInfo; nonNull(value_info_ptr)];
		let mut value_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		ortsys![GetTensorElementType(value_info_ptr, &mut value_type_sys) -> OrtError::GetTensorElementType];
		assert_ne!(value_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

		Ok(DataType::Map {
			key: key_type_sys.into(),
			value: value_type_sys.into()
		})
	}

	pub(super) fn extract_inputs_count(session_ptr: *mut ort_sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetInputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: *mut ort_sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetOutputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const ort_sys::OrtSession, *mut usize) -> *mut ort_sys::OrtStatus },
		session_ptr: *mut ort_sys::OrtSession
	) -> OrtResult<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr, &mut num_nodes) };
		status_to_result(status).map_err(OrtError::GetInOutCount)?;
		assert_null_pointer(status, "SessionStatus")?;
		(num_nodes != 0)
			.then_some(())
			.ok_or_else(|| OrtError::GetInOutCount(OrtApiError::Msg("No nodes in model".to_owned())))?;
		Ok(num_nodes)
	}

	fn extract_input_name(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: ort_sys::size_t) -> OrtResult<String> {
		let f = ort().SessionGetInputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	fn extract_output_name(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: ort_sys::size_t) -> OrtResult<String> {
		let f = ort().SessionGetOutputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	pub(crate) fn raw_pointer_to_string(allocator_ptr: *mut ort_sys::OrtAllocator, c_str: *mut c_char) -> OrtResult<String> {
		let name = char_p_to_string(c_str)?;
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
	) -> OrtResult<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
		status_to_result(status).map_err(OrtError::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		raw_pointer_to_string(allocator_ptr, name_bytes)
	}

	pub(super) fn extract_input(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: usize) -> OrtResult<Input> {
		let input_name = extract_input_name(session_ptr, allocator_ptr, i as _)?;
		let f = ort().SessionGetInputTypeInfo.unwrap();
		let input_type = extract_io(f, session_ptr, i as _)?;
		Ok(Input { name: input_name, input_type })
	}

	pub(super) fn extract_output(session_ptr: *mut ort_sys::OrtSession, allocator_ptr: *mut ort_sys::OrtAllocator, i: usize) -> OrtResult<Output> {
		let output_name = extract_output_name(session_ptr, allocator_ptr, i as _)?;
		let f = ort().SessionGetOutputTypeInfo.unwrap();
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
	) -> OrtResult<DataType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
		status_to_result(status).map_err(OrtError::GetTypeInfo)?;
		assert_non_null_pointer(typeinfo_ptr, "TypeInfo")?;

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty)];
		status_to_result(status).map_err(OrtError::GetOnnxTypeFromTypeInfo)?;
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr) -> OrtError::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr) -> OrtError::CastTypeInfoToSequenceTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_sequence_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr) -> OrtError::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr)? }
			}
			_ => unreachable!()
		};

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		Ok(io_type)
	}
}
