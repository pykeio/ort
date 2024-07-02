#[cfg(any(feature = "operator-libraries", not(windows)))]
use std::ffi::CString;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(feature = "fetch-models")]
use std::path::PathBuf;
use std::{
	any::Any,
	marker::PhantomData,
	ptr::{self, NonNull},
	rc::Rc,
	sync::{atomic::Ordering, Arc}
};

use super::{dangerous, InMemorySession, Input, Output, Session, SharedSessionInner};
#[cfg(feature = "fetch-models")]
use crate::error::FetchModelError;
use crate::{
	environment::get_environment,
	error::{assert_non_null_pointer, status_to_result, Error, Result},
	execution_providers::{apply_execution_providers, ExecutionProviderDispatch},
	memory::{Allocator, MemoryInfo},
	operator::OperatorDomain,
	ortsys
};

/// Creates a session using the builder pattern.
///
/// Once configured, use the [`SessionBuilder::commit_from_file`](crate::SessionBuilder::commit_from_file)
/// method to 'commit' the builder configuration into a [`Session`].
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.commit_from_file("tests/data/upsample.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
	pub(crate) session_options_ptr: NonNull<ort_sys::OrtSessionOptions>,
	memory_info: Option<Rc<MemoryInfo>>,
	#[cfg(feature = "operator-libraries")]
	custom_runtime_handles: Vec<Arc<LibHandle>>,
	operator_domains: Vec<Arc<OperatorDomain>>,
	execution_providers: Vec<ExecutionProviderDispatch>
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		status_to_result(ortsys![unsafe CloneSessionOptions(self.session_options_ptr.as_ptr(), ptr::addr_of_mut!(session_options_ptr))])
			.expect("error cloning session options");
		assert_non_null_pointer(session_options_ptr, "OrtSessionOptions").expect("Cloned session option pointer is null");
		Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: self.memory_info.clone(),
			#[cfg(feature = "operator-libraries")]
			custom_runtime_handles: self.custom_runtime_handles.clone(),
			operator_domains: self.operator_domains.clone(),
			execution_providers: self.execution_providers.clone()
		}
	}
}

impl Drop for SessionBuilder {
	fn drop(&mut self) {
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
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new() -> Result<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr) -> Error::CreateSessionOptions; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: None,
			#[cfg(feature = "operator-libraries")]
			custom_runtime_handles: Vec::new(),
			operator_domains: Vec::new(),
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
		apply_execution_providers(&self, execution_providers.into_iter())?;
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

	/// After performing optimization (configurable with [`SessionBuilder::with_optimization_level`]), serializes the
	/// newly optimized model to the given path (for 'offline' graph optimization).
	///
	/// Note that the file will only be created after the model is committed.
	#[cfg(not(target_arch = "wasm32"))]
	#[cfg_attr(docsrs, doc(cfg(not(target_arch = "wasm32"))))]
	pub fn with_optimized_model_path<S: AsRef<str>>(self, path: S) -> Result<Self> {
		#[cfg(windows)]
		let path = path.as_ref().encode_utf16().chain([0]).collect::<Vec<_>>();
		#[cfg(not(windows))]
		let path = CString::new(path.as_ref())?;
		ortsys![unsafe SetOptimizedModelFilePath(self.session_options_ptr.as_ptr(), path.as_ptr()) -> Error::CreateSessionOptions];
		Ok(self)
	}

	/// Enables profiling. Profile information will be writen to `profiling_file` after profiling completes.
	/// See [`Session::end_profiling`].
	#[cfg(not(target_arch = "wasm32"))]
	#[cfg_attr(docsrs, doc(cfg(not(target_arch = "wasm32"))))]
	pub fn with_profiling<S: AsRef<str>>(self, profiling_file: S) -> Result<Self> {
		#[cfg(windows)]
		let profiling_file = profiling_file.as_ref().encode_utf16().chain([0]).collect::<Vec<_>>();
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

	/// Registers a custom operator library at the given library path.
	#[cfg(all(feature = "operator-libraries", not(target_arch = "wasm32")))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "operator-libraries", not(target_arch = "wasm32")))))]
	pub fn with_operator_library(mut self, lib_path: impl AsRef<str>) -> Result<Self> {
		let path_cstr = CString::new(lib_path.as_ref())?;

		let mut handle: *mut ::std::os::raw::c_void = std::ptr::null_mut();

		let status = ortsys![unsafe RegisterCustomOpsLibrary(self.session_options_ptr.as_ptr(), path_cstr.as_ptr(), &mut handle)];

		let handle = LibHandle(handle);
		// per RegisterCustomOpsLibrary docs, release handle if there was an error and the handle
		// is non-null
		if let Err(e) = status_to_result(status).map_err(Error::CreateSessionOptions) {
			if !handle.is_null() {
				// handle was written to, should release it
				drop(handle);
			}

			return Err(e);
		}

		self.custom_runtime_handles.push(Arc::new(handle));

		Ok(self)
	}

	/// Enables [`onnxruntime-extensions`](https://github.com/microsoft/onnxruntime-extensions) custom operators.
	#[cfg(not(target_arch = "wasm32"))]
	#[cfg_attr(docsrs, doc(cfg(not(target_arch = "wasm32"))))]
	pub fn with_extensions(self) -> Result<Self> {
		let status = ortsys![unsafe EnableOrtCustomOps(self.session_options_ptr.as_ptr())];
		status_to_result(status).map_err(Error::CreateSessionOptions)?;
		Ok(self)
	}

	pub fn with_operators(mut self, domain: impl Into<Arc<OperatorDomain>>) -> Result<Self> {
		let domain = domain.into();
		ortsys![unsafe AddCustomOpDomain(self.session_options_ptr.as_ptr(), domain.ptr()) -> Error::AddCustomOperatorDomain];
		self.operator_domains.push(domain);
		Ok(self)
	}

	/// Downloads a pre-trained ONNX model from the given URL and builds the session.
	#[cfg(all(feature = "fetch-models", not(target_arch = "wasm32")))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "fetch-models", not(target_arch = "wasm32")))))]
	pub fn commit_from_url(self, model_url: impl AsRef<str>) -> Result<Session> {
		let mut download_dir = ort_sys::internal::dirs::cache_dir()
			.expect("could not determine cache directory")
			.join("models");
		if std::fs::create_dir_all(&download_dir).is_err() {
			download_dir = std::env::current_dir().expect("Failed to obtain current working directory");
		}

		let url = model_url.as_ref();
		let model_filename = PathBuf::from(url.split('/').last().expect("Missing filename in model URL"));
		let model_filepath = download_dir.join(model_filename);
		let downloaded_path = if model_filepath.exists() {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), "Model already exists, skipping download");
			model_filepath
		} else {
			tracing::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{url:?}").as_str(), "Downloading model");

			let resp = ureq::get(url).call().map_err(Box::new).map_err(FetchModelError::FetchError)?;

			let len = resp
				.header("Content-Length")
				.and_then(|s| s.parse::<usize>().ok())
				.expect("Missing Content-Length header");
			tracing::info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_reader();

			let f = std::fs::File::create(&model_filepath).expect("Failed to create model file");
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

		self.commit_from_file(downloaded_path)
	}

	/// Loads an ONNX model from a file and builds the session.
	#[cfg(not(target_arch = "wasm32"))]
	#[cfg_attr(docsrs, doc(cfg(not(target_arch = "wasm32"))))]
	pub fn commit_from_file<P>(mut self, model_filepath_ref: P) -> Result<Session>
	where
		P: AsRef<Path>
	{
		let model_filepath = model_filepath_ref.as_ref();
		if !model_filepath.exists() {
			return Err(Error::FileDoesNotExist {
				filename: model_filepath.to_path_buf()
			});
		}

		let model_path = crate::util::path_to_os_char(model_filepath);

		let env = get_environment()?;
		apply_execution_providers(&self, env.execution_providers.iter().cloned())?;

		if env.has_global_threadpool {
			ortsys![unsafe DisablePerSessionThreads(self.session_options_ptr.as_ptr()) -> Error::CreateSessionOptions];
		}

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

		let extras = self.operator_domains.drain(..).map(|d| Box::new(d) as Box<dyn Any>);
		#[cfg(feature = "operator-libraries")]
		let extras = extras.chain(self.custom_runtime_handles.drain(..).map(|d| Box::new(d) as Box<dyn Any>));
		let extras: Vec<Box<dyn Any>> = extras.collect();

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr,
				allocator,
				_extras: extras,
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
	pub fn commit_from_memory_directly(self, model_bytes: &[u8]) -> Result<InMemorySession<'_>> {
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

		let session = self.commit_from_memory(model_bytes)?;

		Ok(InMemorySession { session, phantom: PhantomData })
	}

	/// Load an ONNX graph from memory and commit the session.
	pub fn commit_from_memory(mut self, model_bytes: &[u8]) -> Result<Session> {
		let mut session_ptr: *mut ort_sys::OrtSession = std::ptr::null_mut();

		let env = get_environment()?;
		apply_execution_providers(&self, env.execution_providers.iter().cloned())?;

		if env.has_global_threadpool {
			ortsys![unsafe DisablePerSessionThreads(self.session_options_ptr.as_ptr()) -> Error::CreateSessionOptions];
		}

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

		let extras = self.operator_domains.drain(..).map(|d| Box::new(d) as Box<dyn Any>);
		#[cfg(feature = "operator-libraries")]
		let extras = extras.chain(self.custom_runtime_handles.drain(..).map(|d| Box::new(d) as Box<dyn Any>));
		let extras: Vec<Box<dyn Any>> = extras.collect();

		let session = Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr,
				allocator,
				_extras: extras,
				_environment: Arc::clone(env)
			}),
			inputs,
			outputs
		};
		Ok(session)
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

#[cfg(feature = "operator-libraries")]
struct LibHandle(*mut std::os::raw::c_void);

#[cfg(feature = "operator-libraries")]
impl LibHandle {
	pub(self) fn is_null(&self) -> bool {
		self.0.is_null()
	}
}

#[cfg(feature = "operator-libraries")]
impl Drop for LibHandle {
	fn drop(&mut self) {
		#[cfg(unix)]
		unsafe {
			libc::dlclose(self.0)
		};
		#[cfg(windows)]
		unsafe {
			winapi::um::libloaderapi::FreeLibrary(self.0 as winapi::shared::minwindef::HINSTANCE)
		};
	}
}
