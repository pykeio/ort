//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

#[cfg(not(target_family = "windows"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
#[cfg(feature = "fetch-models")]
use std::path::PathBuf;
use std::{
	ffi::CString,
	marker::PhantomData,
	ops::Deref,
	os::raw::c_char,
	path::Path,
	ptr::{self, NonNull},
	rc::Rc,
	sync::{atomic::Ordering, Arc}
};

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
use crate::{environment::Environment, MemoryInfo};

pub(crate) mod input;
pub(crate) mod output;
pub use self::{input::SessionInputs, output::SessionOutputs};

/// Creates a session using the builder pattern.
///
/// Once configured, use the [`SessionBuilder::with_model_from_file`](crate::SessionBuilder::with_model_from_file)
/// method to 'commit' the builder configuration into a [`Session`].
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.with_model_from_file("tests/data/upsample.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
	pub(crate) session_options_ptr: NonNull<ort_sys::OrtSessionOptions>,
	memory_info: Option<Rc<MemoryInfo>>,
	#[cfg(feature = "custom-ops")]
	custom_runtime_handles: Vec<*mut std::os::raw::c_void>,
	execution_providers: Vec<ExecutionProviderDispatch>
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		status_to_result(ortsys![unsafe CloneSessionOptions(self.session_options_ptr.as_ptr(), ptr::addr_of_mut!(session_options_ptr))])
			.expect("error cloning session options");
		assert_non_null_pointer(session_options_ptr, "OrtSessionOptions").unwrap();
		Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: self.memory_info.clone(),
			#[cfg(feature = "custom-ops")]
			custom_runtime_handles: self.custom_runtime_handles.clone(),
			execution_providers: self.execution_providers.clone()
		}
	}
}

impl Drop for SessionBuilder {
	fn drop(&mut self) {
		#[cfg(feature = "custom-ops")]
		for &handle in &self.custom_runtime_handles {
			close_lib_handle(handle);
		}

		ortsys![unsafe ReleaseSessionOptions(self.session_options_ptr.as_ptr())];
	}
}

impl SessionBuilder {
	/// Creates a new session builder.
	///
	/// ```
	/// # use ort::{GraphOptimizationLevel, Session};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
	/// 	.with_intra_threads(1)?
	/// 	.with_model_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new() -> Result<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr) -> Error::CreateSessionOptions; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: None,
			#[cfg(feature = "custom-ops")]
			custom_runtime_handles: Vec::new(),
			execution_providers: Vec::new()
		})
	}

	/// Registers a list of execution providers for this session. Execution providers are registered in the order they
	/// are provided.
	///
	/// Execution providers will only work if the corresponding Cargo feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled will emit a warning.
	///
	/// ## Notes
	///
	/// - **Indiscriminate use of [`SessionBuilder::with_execution_providers`] in a library** (e.g. always enabling
	///   `CUDAExecutionProvider`) **is discouraged** unless you allow the user to configure the execution providers by
	///   providing a `Vec` of [`ExecutionProviderDispatch`]es.
	pub fn with_execution_providers(self, execution_providers: impl IntoIterator<Item = ExecutionProviderDispatch>) -> Result<Self> {
		apply_execution_providers(&self, execution_providers.into_iter());
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution within nodes. If ONNX Runtime was
	/// built with OpenMP (as is the case with Microsoft's prebuilt binaries), this will have no effect on the number of
	/// threads used. Instead, you can configure the number of threads OpenMP uses via the `OMP_NUM_THREADS` environment
	/// variable.
	///
	/// For configuring the number of threads used when the session execution mode is set to `Parallel`, see
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_intra_threads(self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetIntraOpNumThreads(self.session_options_ptr.as_ptr(), num_threads as _) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to disable per-session thread pool, instead using the environment's global thread pool.
	/// This must be used with an environment created with
	/// [`EnvironmentBuilder::with_global_thread_pool`](crate::environment::EnvironmentBuilder::with_global_thread_pool)
	/// enabled.
	pub fn with_disable_per_session_threads(self) -> Result<Self> {
		ortsys![unsafe DisablePerSessionThreads(self.session_options_ptr.as_ptr()) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution of the graph. If nodes can be run
	/// in parallel, this sets the maximum number of threads to use to run them in parallel.
	///
	/// This has no effect when the session execution mode is set to `Sequential`.
	///
	/// For configuring the number of threads used to parallelize the execution within nodes, see
	/// [`SessionBuilder::with_intra_threads()`].
	pub fn with_inter_threads(self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetInterOpNumThreads(self.session_options_ptr.as_ptr(), num_threads as _) -> Error::CreateSessionOptions];
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
		ortsys![unsafe SetSessionExecutionMode(self.session_options_ptr.as_ptr(), execution_mode) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(self, opt_level: GraphOptimizationLevel) -> Result<Self> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.session_options_ptr.as_ptr(), opt_level.into()) -> Error::CreateSessionOptions];
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
		ortsys![unsafe EnableProfiling(self.session_options_ptr.as_ptr(), profiling_file.as_ptr()) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Enables/disables memory pattern optimization. Disable it if the input size varies, i.e., dynamic batch
	pub fn with_memory_pattern(self, enable: bool) -> Result<Self> {
		if enable {
			ortsys![unsafe EnableMemPattern(self.session_options_ptr.as_ptr()) -> Error::CreateSessionOptions];
		} else {
			ortsys![unsafe DisableMemPattern(self.session_options_ptr.as_ptr()) -> Error::CreateSessionOptions];
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

		let status = ortsys![unsafe RegisterCustomOpsLibrary(self.session_options_ptr.as_ptr(), path_cstr.as_ptr(), &mut handle)];

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

	/// Enable custom operators. See onnxruntime-extensions: <https://github.com/microsoft/onnxruntime-extensions>
	#[cfg(feature = "custom-ops")]
	#[cfg_attr(docsrs, doc(cfg(feature = "custom-ops")))]
	pub fn with_enable_custom_ops(self) -> Result<Self> {
		let status = ortsys![unsafe EnableOrtCustomOps(self.session_options_ptr.as_ptr())];
		status_to_result(status).map_err(Error::CreateSessionOptions)?;
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
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{url:?}").as_str(), "Downloading model");

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
		apply_execution_providers(&self, self.execution_providers.iter().cloned());

		let env_ptr = env.env_ptr.load(Ordering::Relaxed);

		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();
		ortsys![unsafe CreateSession(env_ptr, model_path.as_ptr(), self.session_options_ptr.as_ptr(), &mut session_ptr) -> Error::CreateSession; nonNull(session_ptr)];

		let session_ptr = unsafe { NonNull::new_unchecked(session_ptr) };

		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
				ortsys![unsafe CreateAllocator(session_ptr.as_ptr(), info.ptr.as_ptr(), &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
				unsafe { Allocator::from_raw_unchecked(allocator_ptr) }
			}
			None => Allocator::default()
		};

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, &allocator, i))
			.collect::<Result<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, &allocator, i))
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
		ortsys![unsafe AddSessionConfigEntry(self.session_options_ptr.as_ptr(), str_to_char("session.use_ort_model_bytes_directly").as_ptr(), str_to_char("1").as_ptr())];
		ortsys![unsafe AddSessionConfigEntry(self.session_options_ptr.as_ptr(), str_to_char("session.use_ort_model_bytes_for_initializers").as_ptr(), str_to_char("1").as_ptr())];

		let session = self.with_model_from_memory(model_bytes)?;

		Ok(InMemorySession { session, phantom: PhantomData })
	}

	/// Load an ONNX graph from memory and commit the session.
	pub fn with_model_from_memory(self, model_bytes: &[u8]) -> Result<Session> {
		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();

		let env = get_environment()?;
		apply_execution_providers(&self, env.execution_providers.iter().cloned());

		let env_ptr = env.env_ptr.load(Ordering::Relaxed);

		let model_data = model_bytes.as_ptr().cast::<std::ffi::c_void>();
		let model_data_length = model_bytes.len();
		ortsys![
			unsafe CreateSessionFromArray(env_ptr, model_data, model_data_length as _, self.session_options_ptr.as_ptr(), &mut session_ptr) -> Error::CreateSession;
			nonNull(session_ptr)
		];

		let session_ptr = unsafe { NonNull::new_unchecked(session_ptr) };

		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
				ortsys![unsafe CreateAllocator(session_ptr.as_ptr(), info.ptr.as_ptr(), &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
				unsafe { Allocator::from_raw_unchecked(allocator_ptr) }
			}
			None => Allocator::default()
		};

		// Extract input and output properties
		let num_input_nodes = dangerous::extract_inputs_count(session_ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(session_ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(session_ptr, &allocator, i))
			.collect::<Result<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(session_ptr, &allocator, i))
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

/// Holds onto an [`ort_sys::OrtSession`] pointer and its associated allocator.
///
/// Internally, this is wrapped in an [`Arc`] and shared between a [`Session`] and any [`Value`]s created as a result
/// of [`Session::run`] to ensure that the [`Value`]s are kept alive until all references to the session are dropped.
#[derive(Debug)]
pub struct SharedSessionInner {
	pub(crate) session_ptr: NonNull<ort_sys::OrtSession>,
	allocator: Allocator,
	_environment: Arc<Environment>
}

impl SharedSessionInner {
	/// Returns the underlying [`ort_sys::OrtSession`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtSession {
		self.session_ptr.as_ptr()
	}
}

unsafe impl Send for SharedSessionInner {}
unsafe impl Sync for SharedSessionInner {}

impl Drop for SharedSessionInner {
	#[tracing::instrument]
	fn drop(&mut self) {
		tracing::debug!("dropping SharedSessionInner");
		ortsys![unsafe ReleaseSession(self.session_ptr.as_ptr())];
	}
}

/// An ONNX Runtime graph to be used for inference.
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![input]?)?;
/// # 	Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Session {
	pub(crate) inner: Arc<SharedSessionInner>,
	/// Information about the graph's inputs.
	pub inputs: Vec<Input>,
	/// Information about the graph's outputs.
	pub outputs: Vec<Output>
}

/// A [`Session`] where the graph data is stored in memory.
///
/// This type is automatically `Deref`'d into a `Session`, so you can use it like you would a regular `Session`. See
/// [`Session`] for usage details.
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

/// Information about a [`Session`] input.
#[derive(Debug)]
pub struct Input {
	/// Name of the input.
	pub name: String,
	/// Type of the input's elements.
	pub input_type: ValueType
}

/// Information about a [`Session`] output.
#[derive(Debug)]
pub struct Output {
	/// Name of the output.
	pub name: String,
	/// Type of the output's elements.
	pub output_type: ValueType
}

/// A structure which can be passed to [`Session::run_with_options`] to allow terminating/unterminating a session
/// inference run from a different thread.
#[derive(Debug)]
pub struct RunOptions {
	pub(crate) run_options_ptr: NonNull<ort_sys::OrtRunOptions>
}

// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ac2a08cac0a657604bd5899e0d1a13675
unsafe impl Send for RunOptions {}
unsafe impl Sync for RunOptions {}

impl RunOptions {
	/// Creates a new [`RunOptions`] struct.
	pub fn new() -> Result<Self> {
		let mut run_options_ptr: *mut ort_sys::OrtRunOptions = std::ptr::null_mut();
		ortsys![unsafe CreateRunOptions(&mut run_options_ptr) -> Error::CreateRunOptions; nonNull(run_options_ptr)];
		Ok(Self {
			run_options_ptr: unsafe { NonNull::new_unchecked(run_options_ptr) }
		})
	}

	/// Sets a tag to identify this run in logs.
	pub fn set_tag(&mut self, tag: impl AsRef<str>) -> Result<()> {
		let tag = CString::new(tag.as_ref())?;
		ortsys![unsafe RunOptionsSetRunTag(self.run_options_ptr.as_ptr(), tag.as_ptr()) -> Error::RunOptionsSetTag];
		Ok(())
	}

	/// Sets the termination flag for the runs associated with this [`RunOptions`].
	///
	/// This function returns immediately (it does not wait for the session run to terminate). The run will terminate as
	/// soon as it is able to.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn terminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsSetTerminate(self.run_options_ptr.as_ptr()) -> Error::RunOptionsSetTerminate];
		Ok(())
	}

	/// Resets the termination flag for the runs associated with [`RunOptions`].
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// 	// ...oops, didn't mean to do that
	/// 	let _ = run_options_.unterminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// assert!(res.is_ok());
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn unterminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsUnsetTerminate(self.run_options_ptr.as_ptr()) -> Error::RunOptionsUnsetTerminate];
		Ok(())
	}
}

impl Drop for RunOptions {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseRunOptions(self.run_options_ptr.as_ptr())];
	}
}

impl Session {
	/// Creates a new [`SessionBuilder`].
	pub fn builder() -> Result<SessionBuilder> {
		SessionBuilder::new()
	}

	/// Returns this session's [`Allocator`].
	#[must_use]
	pub fn allocator(&self) -> &Allocator {
		&self.inner.allocator
	}

	/// Creates a new [`IoBinding`] for this session.
	pub fn create_binding(&self) -> Result<IoBinding> {
		IoBinding::new(self)
	}

	/// Returns the underlying [`ort_sys::OrtSession`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtSession {
		self.inner.ptr()
	}

	/// Get a shared ([`Arc`]'d) reference to the underlying [`SharedSessionInner`], which holds the
	/// [`ort_sys::OrtSession`] pointer and the session allocator.
	#[must_use]
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}

	/// Run input data through the ONNX graph, performing inference.
	///
	/// See [`ort::inputs`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![input]?)?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run<'s, 'i, const N: usize>(&'s self, input_values: impl Into<SessionInputs<'i, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(&input_values.iter().map(|(k, _)| k.as_ref()).collect::<Vec<_>>(), input_values.iter().map(|(_, v)| v), None)
			}
		}
	}

	/// Run input data through the ONNX graph, performing inference, with a [`RunOptions`] struct. The most common usage
	/// of `RunOptions` is to allow the session run to be terminated from a different thread.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run_with_options<'s, 'i, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, N>>,
		run_options: Arc<RunOptions>
	) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), Some(run_options))
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), Some(run_options))
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(&input_values.iter().map(|(k, _)| k.as_ref()).collect::<Vec<_>>(), input_values.iter().map(|(_, v)| v), Some(run_options))
			}
		}
	}

	fn run_inner<'v>(
		&self,
		input_names: &[&str],
		input_values: impl Iterator<Item = &'v Value>,
		run_options: Option<Arc<RunOptions>>
	) -> Result<SessionOutputs<'_>> {
		let input_names_ptr: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();
		let output_names_ptr: Vec<*const c_char> = self
			.outputs
			.iter()
			.map(|output| CString::new(output.name.as_str()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();

		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr().cast_const()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};

		ortsys![
			unsafe Run(
				self.inner.session_ptr.as_ptr(),
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
			.map(|tensor_ptr| unsafe {
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), Some(Arc::clone(&self.inner)))
			})
			.collect();

		// Reconvert name ptrs to CString so drop impl is called and memory is freed
		drop(
			input_names_ptr
				.into_iter()
				.chain(output_names_ptr.into_iter())
				.map(|p| {
					assert_non_null_pointer(p, "c_char for CString")?;
					unsafe { Ok(CString::from_raw(p.cast_mut().cast())) }
				})
				.collect::<Result<Vec<_>>>()?
		);

		Ok(SessionOutputs::new(self.outputs.iter().map(|o| o.name.as_str()), outputs))
	}

	/// Gets the session model metadata. See [`ModelMetadata`] for more info.
	pub fn metadata(&self) -> Result<ModelMetadata<'_>> {
		let mut metadata_ptr: *mut ort_sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.inner.session_ptr.as_ptr(), &mut metadata_ptr) -> Error::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(ModelMetadata::new(unsafe { NonNull::new_unchecked(metadata_ptr) }, &self.inner.allocator))
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiling file will be empty.
	#[cfg(feature = "profiling")]
	#[cfg_attr(docsrs, doc(cfg(feature = "profiling")))]
	pub fn end_profiling(&self) -> Result<String> {
		let mut profiling_name: *mut c_char = std::ptr::null_mut();

		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr.as_ptr(), self.inner.allocator.ptr.as_ptr(), &mut profiling_name)];
		assert_non_null_pointer(profiling_name, "ProfilingName")?;
		dangerous::raw_pointer_to_string(&self.inner.allocator, profiling_name)
	}
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
// Allowing `Sync` segfaults with CUDA, DirectML, and seemingly any EP other than the CPU EP. I'm not certain if it's a
// temporary bug in ONNX Runtime or a wontfix. Maybe this impl should be removed just to be safe?
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
	use crate::value::{extract_data_type_from_map_info, extract_data_type_from_sequence_info, extract_data_type_from_tensor_info};

	pub(super) fn extract_inputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![unsafe SessionGetInputCount];
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![unsafe SessionGetOutputCount];
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const ort_sys::OrtSession, *mut ort_sys::size_t) -> *mut ort_sys::OrtStatus },
		session_ptr: NonNull<ort_sys::OrtSession>
	) -> Result<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr.as_ptr(), &mut num_nodes) };
		status_to_result(status).map_err(Error::GetInOutCount)?;
		assert_null_pointer(status, "SessionStatus")?;
		(num_nodes != 0)
			.then_some(())
			.ok_or_else(|| Error::GetInOutCount(ErrorInternal::Msg("No nodes in model".to_owned())))?;
		Ok(num_nodes as _)
	}

	fn extract_input_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetInputName];
		extract_io_name(f, session_ptr, allocator, i)
	}

	fn extract_output_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetOutputName];
		extract_io_name(f, session_ptr, allocator, i)
	}

	pub(crate) fn raw_pointer_to_string(allocator: &Allocator, c_str: *mut c_char) -> Result<String> {
		let name = match char_p_to_string(c_str) {
			Ok(name) => name,
			Err(e) => {
				unsafe { allocator.free(c_str) };
				return Err(e);
			}
		};
		unsafe { allocator.free(c_str) };
		Ok(name)
	}

	fn extract_io_name(
		f: extern_system_fn! { unsafe fn(
			*const ort_sys::OrtSession,
			ort_sys::size_t,
			*mut ort_sys::OrtAllocator,
			*mut *mut c_char,
		) -> *mut ort_sys::OrtStatus },
		session_ptr: NonNull<ort_sys::OrtSession>,
		allocator: &Allocator,
		i: ort_sys::size_t
	) -> Result<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, allocator.ptr.as_ptr(), &mut name_bytes) };
		status_to_result(status).map_err(Error::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		raw_pointer_to_string(allocator, name_bytes)
	}

	pub(super) fn extract_input(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Input> {
		let input_name = extract_input_name(session_ptr, allocator, i as _)?;
		let f = ortsys![unsafe SessionGetInputTypeInfo];
		let input_type = extract_io(f, session_ptr, i as _)?;
		Ok(Input { name: input_name, input_type })
	}

	pub(super) fn extract_output(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Output> {
		let output_name = extract_output_name(session_ptr, allocator, i as _)?;
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
		session_ptr: NonNull<ort_sys::OrtSession>,
		i: ort_sys::size_t
	) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, &mut typeinfo_ptr) };
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
