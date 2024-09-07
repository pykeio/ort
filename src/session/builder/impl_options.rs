#[cfg(not(windows))]
use std::ffi::CString;
use std::{path::Path, rc::Rc, sync::Arc};

use super::SessionBuilder;
use crate::{
	error::Result,
	execution_providers::{apply_execution_providers, ExecutionProviderDispatch},
	ortsys,
	util::path_to_os_char,
	MemoryInfo, OperatorDomain
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
		ortsys![unsafe SetIntraOpNumThreads(self.session_options_ptr.as_ptr(), num_threads as _)?];
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
		ortsys![unsafe SetInterOpNumThreads(self.session_options_ptr.as_ptr(), num_threads as _)?];
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
		ortsys![unsafe SetSessionExecutionMode(self.session_options_ptr.as_ptr(), execution_mode)?];
		Ok(self)
	}

	/// Set the session's optimization level. See [`GraphOptimizationLevel`] for more information on the different
	/// optimization levels.
	pub fn with_optimization_level(self, opt_level: GraphOptimizationLevel) -> Result<Self> {
		ortsys![unsafe SetSessionGraphOptimizationLevel(self.session_options_ptr.as_ptr(), opt_level.into())?];
		Ok(self)
	}

	/// After performing optimization (configurable with [`SessionBuilder::with_optimization_level`]), serializes the
	/// newly optimized model to the given path (for 'offline' graph optimization).
	///
	/// Note that the file will only be created after the model is committed.
	pub fn with_optimized_model_path<S: AsRef<str>>(self, path: S) -> Result<Self> {
		#[cfg(windows)]
		let path = path.as_ref().encode_utf16().chain([0]).collect::<Vec<_>>();
		#[cfg(not(windows))]
		let path = CString::new(path.as_ref())?;
		ortsys![unsafe SetOptimizedModelFilePath(self.session_options_ptr.as_ptr(), path.as_ptr())?];
		Ok(self)
	}

	/// Enables profiling. Profile information will be writen to `profiling_file` after profiling completes.
	/// See [`Session::end_profiling`].
	pub fn with_profiling<S: AsRef<str>>(self, profiling_file: S) -> Result<Self> {
		#[cfg(windows)]
		let profiling_file = profiling_file.as_ref().encode_utf16().chain([0]).collect::<Vec<_>>();
		#[cfg(not(windows))]
		let profiling_file = CString::new(profiling_file.as_ref())?;
		ortsys![unsafe EnableProfiling(self.session_options_ptr.as_ptr(), profiling_file.as_ptr())?];
		Ok(self)
	}

	/// Enables/disables memory pattern optimization. Disable it if the input size varies, i.e., dynamic batch
	pub fn with_memory_pattern(self, enable: bool) -> Result<Self> {
		if enable {
			ortsys![unsafe EnableMemPattern(self.session_options_ptr.as_ptr())?];
		} else {
			ortsys![unsafe DisableMemPattern(self.session_options_ptr.as_ptr())?];
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
	pub fn with_operator_library(self, lib_path: impl AsRef<Path>) -> Result<Self> {
		let path_cstr = path_to_os_char(lib_path);
		ortsys![unsafe RegisterCustomOpsLibrary_V2(self.session_options_ptr.as_ptr(), path_cstr.as_ptr())?];
		Ok(self)
	}

	/// Enables [`onnxruntime-extensions`](https://github.com/microsoft/onnxruntime-extensions) custom operators.
	pub fn with_extensions(self) -> Result<Self> {
		ortsys![unsafe EnableOrtCustomOps(self.session_options_ptr.as_ptr())?];
		Ok(self)
	}

	pub fn with_operators(mut self, domain: impl Into<Arc<OperatorDomain>>) -> Result<Self> {
		let domain = domain.into();
		ortsys![unsafe AddCustomOpDomain(self.session_options_ptr.as_ptr(), domain.ptr())?];
		self.operator_domains.push(domain);
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
