//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

#![allow(clippy::tabs_in_doc_comments)]

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
#[cfg(feature = "fetch-models")]
use std::{env, path::PathBuf, time::Duration};
use std::{
	ffi::CString,
	fmt::{self, Debug},
	marker::PhantomData,
	ops::Deref,
	os::raw::c_char,
	path::Path,
	sync::Arc
};

use tracing::error;

use super::{
	char_p_to_string,
	environment::Environment,
	error::{assert_non_null_pointer, assert_null_pointer, status_to_result, OrtApiError, OrtError, OrtResult},
	execution_providers::{apply_execution_providers, ExecutionProvider},
	extern_system_fn,
	metadata::Metadata,
	ort, ortsys, sys,
	tensor::TensorElementDataType,
	AllocatorType, GraphOptimizationLevel, MemType
};
#[cfg(feature = "fetch-models")]
use super::{download::ModelUrl, error::OrtDownloadError};
use crate::{io_binding::IoBinding, value::Value};

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
	session_options_ptr: *mut sys::OrtSessionOptions,

	allocator: AllocatorType,
	memory_type: MemType,
	custom_runtime_handles: Vec<*mut std::os::raw::c_void>,
	execution_providers: Vec<ExecutionProvider>
}

impl Debug for SessionBuilder {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
		f.debug_struct("SessionBuilder")
			.field("env", &self.env.name())
			.field("allocator", &self.allocator)
			.field("memory_type", &self.memory_type)
			.finish()
	}
}

impl Drop for SessionBuilder {
	#[tracing::instrument]
	fn drop(&mut self) {
		for &handle in self.custom_runtime_handles.iter() {
			close_lib_handle(handle);
		}

		if self.session_options_ptr.is_null() {
			error!("Session options pointer is null, not dropping");
		} else {
			ortsys![unsafe ReleaseSessionOptions(self.session_options_ptr)];
		}
	}
}

impl SessionBuilder {
	/// Creates a new session builder in the given environment.
	pub fn new(env: &Arc<Environment>) -> OrtResult<Self> {
		let mut session_options_ptr: *mut sys::OrtSessionOptions = std::ptr::null_mut();
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
	/// [`EnvBuilder::with_global_thread_pool`](crate::environment::EnvBuilder::with_global_thread_pool) enabled.
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
			sys::ExecutionMode::ORT_PARALLEL
		} else {
			sys::ExecutionMode::ORT_SEQUENTIAL
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
	pub fn with_custom_op_lib(mut self, lib_path: &str) -> OrtResult<Self> {
		let path_cstr = CString::new(lib_path)?;

		let mut handle: *mut ::std::os::raw::c_void = std::ptr::null_mut();

		let status = ortsys![unsafe RegisterCustomOpsLibrary(self.session_options_ptr, path_cstr.as_ptr(), &mut handle)];

		// per RegisterCustomOpsLibrary docs, release handle if there was an error and the handle
		// is non-null
		match status_to_result(status).map_err(OrtError::CreateSessionOptions) {
			Ok(_) => {}
			Err(e) => {
				if !handle.is_null() {
					// handle was written to, should release it
					close_lib_handle(handle);
				}

				return Err(e);
			}
		}

		self.custom_runtime_handles.push(handle);

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
		P: AsRef<Path> + std::fmt::Debug
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

		let env_ptr: *const sys::OrtEnv = self.env.ptr();

		let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession; nonNull(session_ptr)];

		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::CreateAllocator; nonNull(allocator_ptr)];

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			env: Arc::clone(&self.env),
			session_ptr: Arc::new(SessionPointerHolder { inner: session_ptr }),
			allocator_ptr,
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
	pub fn with_model_from_file_and_external_initializers<'v, 'i, P>(self, model_filepath_ref: P, initializers: &'i [(String, Value<'v>)]) -> OrtResult<Session>
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

		let env_ptr: *const sys::OrtEnv = self.env.ptr();

		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::CreateAllocator; nonNull(allocator_ptr)];

		let initializer_names: Vec<CString> = initializers
			.iter()
			.map(|(name, _)| CString::new(name.as_str()).unwrap())
			.map(|n| CString::new(n).unwrap())
			.collect();
		let initializer_names_ptr: Vec<*const c_char> = initializer_names.iter().map(|n| n.as_ptr() as *const c_char).collect();

		let initializers: Vec<*const sys::OrtValue> = initializers.iter().map(|input_array_ort| input_array_ort.1.ptr() as *const _).collect();
		if !initializers.is_empty() {
			assert_eq!(initializer_names.len(), initializers.len());
			ortsys![unsafe AddExternalInitializers(self.session_options_ptr, initializer_names_ptr.as_ptr(), initializers.as_ptr(), initializers.len() as _) -> OrtError::CreateSession];
		}

		let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr, &mut session_ptr) -> OrtError::CreateSession; nonNull(session_ptr)];

		std::mem::drop(initializer_names);
		std::mem::drop(initializers);

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		Ok(Session {
			env: Arc::clone(&self.env),
			session_ptr: Arc::new(SessionPointerHolder { inner: session_ptr }),
			allocator_ptr,
			inputs,
			outputs
		})
	}

	/// Load an ONNX graph from memory and commit the session.
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
		let mut session_ptr: *mut sys::OrtSession = std::ptr::null_mut();

		let env_ptr: *const sys::OrtEnv = self.env.ptr();

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

		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr) -> OrtError::CreateAllocator; nonNull(allocator_ptr)];

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, allocator_ptr, i))
			.collect::<OrtResult<Vec<Output>>>()?;

		let session = Session {
			env: Arc::clone(&self.env),
			session_ptr: Arc::new(SessionPointerHolder { inner: session_ptr }),
			allocator_ptr,
			inputs,
			outputs
		};
		Ok(session)
	}
}

#[derive(Debug)]
pub struct SessionPointerHolder {
	pub inner: *mut sys::OrtSession
}

unsafe impl Send for SessionPointerHolder {}
unsafe impl Sync for SessionPointerHolder {}

impl Drop for SessionPointerHolder {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseSession(self.inner)];
		self.inner = std::ptr::null_mut();
	}
}

/// Type storing the session information, built from an [`Environment`](crate::environment::Environment)
#[derive(Debug)]
pub struct Session {
	#[allow(dead_code)]
	env: Arc<Environment>,
	pub(crate) session_ptr: Arc<SessionPointerHolder>,
	allocator_ptr: *mut sys::OrtAllocator,
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
	pub input_type: TensorElementDataType,
	/// Shape of the input layer
	///
	/// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
	pub dimensions: Vec<Option<u32>>
}

/// Information about an ONNX's output as stored in loaded file
#[derive(Debug)]
pub struct Output {
	/// Name of the output layer
	pub name: String,
	/// Type of the output layer's elements
	pub output_type: TensorElementDataType,
	/// Shape of the output layer
	///
	/// C API uses a i64 for the dimensions. We use an unsigned of the same range of the positive values.
	pub dimensions: Vec<Option<u32>>
}

impl Input {
	/// Return an iterator over the shape elements of the input layer
	///
	/// Note: The member [`Input::dimensions`](struct.Input.html#structfield.dimensions)
	/// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
	/// iterator converts to `usize`.
	pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
		self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
	}
}

impl Output {
	/// Return an iterator over the shape elements of the output layer
	///
	/// Note: The member [`Output::dimensions`](struct.Output.html#structfield.dimensions)
	/// stores `u32` (since ONNX uses `i64` but which cannot be negative) so the
	/// iterator converts to `usize`.
	pub fn dimensions(&self) -> impl Iterator<Item = Option<usize>> + '_ {
		self.dimensions.iter().map(|d| d.map(|d2| d2 as usize))
	}
}

impl Drop for Session {
	#[tracing::instrument]
	fn drop(&mut self) {
		self.allocator_ptr = std::ptr::null_mut();
	}
}

impl Session {
	pub fn allocator(&self) -> *mut sys::OrtAllocator {
		self.allocator_ptr
	}

	pub fn bind(&self) -> OrtResult<IoBinding> {
		IoBinding::new(self)
	}

	/// Run the input data through the ONNX graph, performing inference.
	///
	/// Note that ONNX models can have multiple inputs; a `Vec<_>` is thus
	/// used for the input data here.
	pub fn run<'s, 'm, 'v, 'i>(&'s self, input_values: Vec<Value<'v>>) -> OrtResult<Vec<Value<'static>>>
	where
		's: 'm, // 's outlives 'm (session outlives memory info)
		'i: 'v
	{
		// Build arguments to Run()

		let input_names_ptr: Vec<*const c_char> = self
			.inputs
			.iter()
			.map(|input| input.name.clone())
			.map(|n| CString::new(n).unwrap())
			.map(|n| n.into_raw() as *const c_char)
			.collect();

		let output_names_cstring: Vec<CString> = self
			.outputs
			.iter()
			.map(|output| output.name.clone())
			.map(|n| CString::new(n).unwrap())
			.collect();
		let output_names_ptr: Vec<*const c_char> = output_names_cstring.iter().map(|n| n.as_ptr() as *const c_char).collect();

		let mut output_tensor_ptrs: Vec<*mut sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_values: Vec<*const sys::OrtValue> = input_values.iter().map(|input_array_ort| input_array_ort.ptr() as *const _).collect();

		let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();

		ortsys![
			unsafe Run(
				self.session_ptr.inner,
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
			.map(|tensor_ptr| Value::from_raw(tensor_ptr, Arc::clone(&self.session_ptr)))
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

		Ok(outputs)
	}

	pub fn run_with_binding<'s, 'a: 's>(&'a self, binding: &IoBinding<'s>) -> OrtResult<()> {
		let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
		ortsys![unsafe RunWithBinding(self.session_ptr.inner, run_options_ptr, binding.ptr) -> OrtError::SessionRun];
		Ok(())
	}

	/// Gets the session model metadata. See [`Metadata`] for more info.
	pub fn metadata(&self) -> OrtResult<Metadata> {
		let mut metadata_ptr: *mut sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.session_ptr.inner, &mut metadata_ptr) -> OrtError::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(Metadata::new(metadata_ptr, self.allocator_ptr))
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiing file will be empty.
	#[cfg(feature = "profiling")]
	pub fn end_profiling(&self) -> OrtResult<String> {
		let mut profiling_name: *mut c_char = std::ptr::null_mut();

		ortsys![unsafe SessionEndProfiling(self.session_ptr.inner, self.allocator_ptr, &mut profiling_name)];
		assert_non_null_pointer(profiling_name, "ProfilingName")?;
		dangerous::raw_pointer_to_string(self.allocator_ptr, profiling_name)
	}
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
unsafe impl Sync for Session {}

unsafe fn get_tensor_dimensions(tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo) -> OrtResult<Vec<i64>> {
	let mut num_dims = 0;
	ortsys![GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> OrtError::GetDimensionsCount];

	let mut node_dims: Vec<i64> = vec![0; num_dims as _];
	ortsys![GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> OrtError::GetDimensions];
	Ok(node_dims)
}

unsafe fn extract_data_type(tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo) -> OrtResult<TensorElementDataType> {
	let mut type_sys = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(tensor_info_ptr, &mut type_sys) -> OrtError::GetTensorElementType];
	assert_ne!(type_sys, sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	Ok(type_sys.into())
}

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
	use crate::{ortfree, sys::size_t, tensor::TensorElementDataType};

	pub(super) fn extract_inputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetInputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: *mut sys::OrtSession) -> OrtResult<usize> {
		let f = ort().SessionGetOutputCount.unwrap();
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const sys::OrtSession, *mut usize) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession
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

	fn extract_input_name(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: size_t) -> OrtResult<String> {
		let f = ort().SessionGetInputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	fn extract_output_name(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: size_t) -> OrtResult<String> {
		let f = ort().SessionGetOutputName.unwrap();
		extract_io_name(f, session_ptr, allocator_ptr, i)
	}

	pub(crate) fn raw_pointer_to_string(allocator_ptr: *mut sys::OrtAllocator, c_str: *mut c_char) -> OrtResult<String> {
		let name = char_p_to_string(c_str)?;
		ortfree!(unsafe allocator_ptr, c_str);
		Ok(name)
	}

	fn extract_io_name(
		f: extern_system_fn! { unsafe fn(
			*const sys::OrtSession,
			size_t,
			*mut sys::OrtAllocator,
			*mut *mut c_char,
		) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession,
		allocator_ptr: *mut sys::OrtAllocator,
		i: size_t
	) -> OrtResult<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, allocator_ptr, &mut name_bytes) };
		status_to_result(status).map_err(OrtError::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		raw_pointer_to_string(allocator_ptr, name_bytes)
	}

	pub(super) fn extract_input(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<Input> {
		let input_name = extract_input_name(session_ptr, allocator_ptr, i as _)?;
		let f = ort().SessionGetInputTypeInfo.unwrap();
		let (input_type, dimensions) = extract_io(f, session_ptr, i as _)?;
		Ok(Input {
			name: input_name,
			input_type,
			dimensions
		})
	}

	pub(super) fn extract_output(session_ptr: *mut sys::OrtSession, allocator_ptr: *mut sys::OrtAllocator, i: usize) -> OrtResult<Output> {
		let output_name = extract_output_name(session_ptr, allocator_ptr, i as _)?;
		let f = ort().SessionGetOutputTypeInfo.unwrap();
		let (output_type, dimensions) = extract_io(f, session_ptr, i as _)?;
		Ok(Output {
			name: output_name,
			output_type,
			dimensions
		})
	}

	fn extract_io(
		f: extern_system_fn! { unsafe fn(
			*const sys::OrtSession,
			size_t,
			*mut *mut sys::OrtTypeInfo,
		) -> *mut sys::OrtStatus },
		session_ptr: *mut sys::OrtSession,
		i: size_t
	) -> OrtResult<(TensorElementDataType, Vec<Option<u32>>)> {
		let mut typeinfo_ptr: *mut sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr, i, &mut typeinfo_ptr) };
		status_to_result(status).map_err(OrtError::GetTypeInfo)?;
		assert_non_null_pointer(typeinfo_ptr, "TypeInfo")?;

		let mut tensor_info_ptr: *const sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut tensor_info_ptr) -> OrtError::CastTypeInfoToTensorInfo; nonNull(tensor_info_ptr)];

		let io_type: TensorElementDataType = unsafe { extract_data_type(tensor_info_ptr)? };
		let node_dims = unsafe { get_tensor_dimensions(tensor_info_ptr)? };

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];

		Ok((io_type, node_dims.into_iter().map(|d| if d == -1 { None } else { Some(d as u32) }).collect()))
	}
}
