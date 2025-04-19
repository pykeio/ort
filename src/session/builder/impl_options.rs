use alloc::sync::Arc;
use core::{
	any::Any,
	ffi::{c_int, c_void},
	ptr
};
#[cfg(feature = "std")]
use std::{borrow::Cow, path::Path};

use super::SessionBuilder;
#[cfg(feature = "std")]
use crate::util::path_to_os_char;
use crate::{
	AsPointer,
	environment::{self, ThreadManager},
	error::Result,
	execution_providers::{ExecutionProviderDispatch, apply_execution_providers},
	logging::{LogLevel, LoggerFunction},
	memory::MemoryInfo,
	operator::OperatorDomain,
	ortsys,
	util::with_cstr,
	value::DynValue
};

impl SessionBuilder {
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
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> Result<Self> {
		apply_execution_providers(&mut self, execution_providers.as_ref(), "session options")?;
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution within nodes. If ONNX Runtime was
	/// built with OpenMP (as is the case with Microsoft's prebuilt binaries), this will have no effect on the number of
	/// threads used. Instead, you can configure the number of threads OpenMP uses via the `OMP_NUM_THREADS` environment
	/// variable.
	///
	/// For configuring the number of threads used when the session execution mode is set to `Parallel`, see
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_intra_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetIntraOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	/// Configure the session to use a number of threads to parallelize the execution of the graph. If nodes can be run
	/// in parallel, this sets the maximum number of threads to use to run them in parallel.
	///
	/// This has no effect when the session execution mode is set to `Sequential`.
	///
	/// For configuring the number of threads used to parallelize the execution within nodes, see
	/// [`SessionBuilder::with_intra_threads()`].
	pub fn with_inter_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetInterOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	/// Enable/disable the parallel execution mode for this session. By default, this is disabled.
	///
	/// Parallel execution can improve performance for models with many branches, at the cost of higher memory usage.
	/// You can configure the amount of threads used to parallelize the execution of the graph via
	/// [`SessionBuilder::with_inter_threads()`].
	pub fn with_parallel_execution(mut self, parallel_execution: bool) -> Result<Self> {
		let execution_mode = if parallel_execution {
			ort_sys::ExecutionMode::ORT_PARALLEL
		} else {
			ort_sys::ExecutionMode::ORT_SEQUENTIAL
		};
		ortsys![unsafe SetSessionExecutionMode(self.ptr_mut(), execution_mode)?];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(mut self, opt_level: GraphOptimizationLevel) -> Result<Self> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.ptr_mut(), opt_level.into())?];
		Ok(self)
	}

	/// After performing optimization (configurable with [`SessionBuilder::with_optimization_level`]), serializes the
	/// newly optimized model to the given path (for 'offline' graph optimization).
	///
	/// Note that the file will only be created after the model is committed.
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_optimized_model_path<S: AsRef<Path>>(mut self, path: S) -> Result<Self> {
		let path = crate::util::path_to_os_char(path);
		ortsys![unsafe SetOptimizedModelFilePath(self.ptr_mut(), path.as_ptr())?];
		Ok(self)
	}

	/// Enables profiling. Profile information will be writen to `profiling_file` after profiling completes.
	/// See [`Session::end_profiling`].
	///
	/// [`Session::end_profiling`]: crate::session::Session::end_profiling
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_profiling<S: AsRef<Path>>(mut self, profiling_file: S) -> Result<Self> {
		let profiling_file = crate::util::path_to_os_char(profiling_file);
		ortsys![unsafe EnableProfiling(self.ptr_mut(), profiling_file.as_ptr())?];
		Ok(self)
	}

	/// Enables/disables memory pattern optimization. Disable it if the input size varies, i.e., dynamic batch
	pub fn with_memory_pattern(mut self, enable: bool) -> Result<Self> {
		if enable {
			ortsys![unsafe EnableMemPattern(self.ptr_mut())?];
		} else {
			ortsys![unsafe DisableMemPattern(self.ptr_mut())?];
		}
		Ok(self)
	}

	/// Set the session's allocator options from a [`MemoryInfo`].
	///
	/// If not provided, the session is created using ONNX Runtime's default device allocator.
	pub fn with_allocator(mut self, info: MemoryInfo) -> Result<Self> {
		self.memory_info = Some(Arc::new(info));
		Ok(self)
	}

	/// Registers a custom operator library at the given library path.
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_operator_library(mut self, lib_path: impl AsRef<Path>) -> Result<Self> {
		let path_cstr = path_to_os_char(lib_path);
		ortsys![unsafe RegisterCustomOpsLibrary_V2(self.ptr_mut(), path_cstr.as_ptr())?];
		Ok(self)
	}

	/// Enables [`onnxruntime-extensions`](https://github.com/microsoft/onnxruntime-extensions) custom operators.
	pub fn with_extensions(mut self) -> Result<Self> {
		ortsys![unsafe EnableOrtCustomOps(self.ptr_mut())?];
		Ok(self)
	}

	pub fn with_operators(mut self, domain: impl Into<Arc<OperatorDomain>>) -> Result<Self> {
		let domain: Arc<OperatorDomain> = domain.into();
		ortsys![unsafe AddCustomOpDomain(self.ptr_mut(), domain.ptr().cast_mut())?];
		self.operator_domains.push(domain);
		Ok(self)
	}

	/// Enables/disables deterministic computation.
	///
	/// The default (non-deterministic) kernels will typically use faster algorithms that may introduce slight variance.
	/// Enabling deterministic compute will output reproducible results, but may come at a performance penalty.
	pub fn with_deterministic_compute(mut self, enable: bool) -> Result<Self> {
		ortsys![unsafe SetDeterministicCompute(self.ptr_mut(), enable)?];
		Ok(self)
	}

	pub fn with_initializer(mut self, name: impl AsRef<str>, value: impl Into<Arc<DynValue>>) -> Result<Self> {
		let ptr = self.ptr_mut();
		let value: Arc<DynValue> = value.into();
		with_cstr(name.as_ref().as_bytes(), &|name| {
			ortsys![unsafe AddInitializer(ptr, name.as_ptr(), value.ptr())?];
			Ok(())
		})?;
		self.initializers.push(value);
		Ok(self)
	}

	pub fn with_external_initializer(mut self, name: impl AsRef<str>, value: impl Into<Arc<DynValue>>) -> Result<Self> {
		let ptr = self.ptr_mut();
		let value: Arc<DynValue> = value.into();
		with_cstr(name.as_ref().as_bytes(), &|name| {
			ortsys![unsafe AddExternalInitializers(ptr, &name.as_ptr(), &value.ptr(), 1)?];
			Ok(())
		})?;
		self.initializers.push(value);
		Ok(self)
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_external_initializer_file_in_memory(mut self, file_name: impl AsRef<Path>, buffer: Cow<'static, [u8]>) -> Result<Self> {
		let file_name = path_to_os_char(file_name);
		let sizes = [buffer.len()];
		ortsys![unsafe AddExternalInitializersFromMemory(self.ptr_mut(), &file_name.as_ptr(), &buffer.as_ptr().cast::<core::ffi::c_char>().cast_mut(), sizes.as_ptr(), 1)?];
		self.external_initializer_buffers.push(buffer);
		Ok(self)
	}

	pub fn with_log_id(mut self, id: impl AsRef<str>) -> Result<Self> {
		let ptr = self.ptr_mut();
		with_cstr(id.as_ref().as_bytes(), &|id| {
			ortsys![unsafe SetSessionLogId(ptr, id.as_ptr())?];
			Ok(())
		})?;
		Ok(self)
	}

	pub fn with_dimension_override(mut self, name: impl AsRef<str>, size: i64) -> Result<Self> {
		let ptr = self.ptr_mut();
		with_cstr(name.as_ref().as_bytes(), &|name| {
			ortsys![unsafe AddFreeDimensionOverrideByName(ptr, name.as_ptr(), size)?];
			Ok(())
		})?;
		Ok(self)
	}

	pub fn with_dimension_override_by_denotation(mut self, denotation: impl AsRef<str>, size: i64) -> Result<Self> {
		let ptr = self.ptr_mut();
		with_cstr(denotation.as_ref().as_bytes(), &|denotation| {
			ortsys![unsafe AddFreeDimensionOverride(ptr, denotation.as_ptr(), size)?];
			Ok(())
		})?;
		Ok(self)
	}

	pub fn with_prepacked_weights(mut self, weights: &PrepackedWeights) -> Result<Self> {
		self.prepacked_weights = Some(weights.clone());
		Ok(self)
	}

	/// Configures this environment to use its own thread pool instead of defaulting to the
	/// [`Environment`](crate::environment::Environment)'s global thread pool if one was defined.
	pub fn with_independent_thread_pool(mut self) -> Result<Self> {
		self.no_global_thread_pool = true;
		Ok(self)
	}

	pub fn with_no_environment_execution_providers(mut self) -> Result<Self> {
		self.no_env_eps = true;
		Ok(self)
	}

	pub fn with_thread_manager<T: ThreadManager + Any + 'static>(mut self, manager: T) -> Result<Self> {
		let manager = Arc::new(manager);
		ortsys![unsafe SessionOptionsSetCustomThreadCreationOptions(self.ptr_mut(), (&*manager as *const T) as *mut c_void)?];
		ortsys![unsafe SessionOptionsSetCustomCreateThreadFn(self.ptr_mut(), Some(environment::thread_create::<T>))?];
		ortsys![unsafe SessionOptionsSetCustomJoinThreadFn(self.ptr_mut(), Some(environment::thread_join::<T>))?];
		self.thread_manager = Some(manager as Arc<dyn Any>);
		Ok(self)
	}

	/// Configures this session to use a custom logger function.
	///
	/// This will be called whenever a message pertaining to this session is to be logged, overriding the default log
	/// handler ([`tracing`] if the `tracing` feature is enabled, otherwise ONNX Runtime's stdio logger).
	///
	/// ```
	/// # use ort::{session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let mut session = Session::builder()?
	/// 	.with_logger(Box::new(
	/// 		|level: ort::logging::LogLevel, category: &str, id: &str, code_location: &str, message: &str| {
	/// 			// ...
	/// 		}
	/// 	))?
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn with_logger(mut self, logger: LoggerFunction) -> Result<Self> {
		let logger = Arc::new(logger);
		ortsys![unsafe SetUserLoggingFunction(self.ptr_mut(), crate::logging::custom_logger, Arc::as_ptr(&logger) as *mut c_void)?];
		self.logger = Some(logger);
		Ok(self)
	}

	/// Sets the severity level for messages logged by this session.
	///
	/// Note that when [`tracing`] integration is enabled via the `tracing` feature, the global log level takes
	/// precedence, i.e. if the application was initialized with `ort`'s log level set to `warn` via the `RUST_LOG`
	/// environment variable or similar, setting a session's log severity level to `verbose` will still have it only
	/// log `warn` messages or higher.`
	pub fn with_log_level(mut self, level: LogLevel) -> Result<Self> {
		ortsys![unsafe SetSessionLogSeverityLevel(self.ptr_mut(), ort_sys::OrtLoggingLevel::from(level) as _)?];
		Ok(self)
	}

	/// Controls the level of verbosity for messages logged under [`LogLevel::Verbose`]; higher values = more verbose.
	pub fn with_log_verbosity(mut self, verbosity: c_int) -> Result<Self> {
		ortsys![unsafe SetSessionLogVerbosityLevel(self.ptr_mut(), verbosity)?];
		Ok(self)
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
	/// > Fusion for the CUDA execution provider. The impact on accuracy is negligible based on our evaluation; F1 score
	/// > for a BERT model on SQuAD v1.1 is almost the same (87.05 vs 87.03).
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

#[derive(Debug)]
struct PrepackedWeightsInner(*mut ort_sys::OrtPrepackedWeightsContainer);

impl Drop for PrepackedWeightsInner {
	fn drop(&mut self) {
		ortsys![unsafe ReleasePrepackedWeightsContainer(self.0)];
	}
}

#[derive(Debug, Clone)]
pub struct PrepackedWeights {
	inner: Arc<PrepackedWeightsInner>
}

impl PrepackedWeights {
	#[allow(clippy::new_without_default)]
	pub fn new() -> Self {
		let mut ptr: *mut ort_sys::OrtPrepackedWeightsContainer = ptr::null_mut();
		ortsys![unsafe CreatePrepackedWeightsContainer(&mut ptr).expect("")];
		Self {
			inner: Arc::new(PrepackedWeightsInner(ptr))
		}
	}
}

impl AsPointer for PrepackedWeights {
	type Sys = ort_sys::OrtPrepackedWeightsContainer;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.0
	}

	fn ptr_mut(&mut self) -> *mut Self::Sys {
		self.inner.0
	}
}
