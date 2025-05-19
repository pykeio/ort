use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLSpecializationStrategy {
	/// The strategy that should work well for most applications.
	Default,
	/// Prefer the prediction latency at the potential cost of specialization time, memory footprint, and the disk space
	/// usage of specialized artifacts.
	FastPrediction
}

impl CoreMLSpecializationStrategy {
	pub(crate) fn as_str(&self) -> &'static str {
		match self {
			Self::Default => "Default",
			Self::FastPrediction => "FastPrediction"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLComputeUnits {
	/// Enable CoreML EP for all compatible Apple devices.
	All,
	/// Enable CoreML EP for Apple devices with a compatible Neural Engine (ANE).
	CPUAndNeuralEngine,
	/// Enable CoreML EP for Apple devices with a compatible GPU.
	CPUAndGPU,
	/// Limit CoreML to running on CPU only.
	CPUOnly
}

impl CoreMLComputeUnits {
	pub(crate) fn as_str(&self) -> &'static str {
		match self {
			Self::All => "ALL",
			Self::CPUAndNeuralEngine => "CPUAndNeuralEngine",
			Self::CPUAndGPU => "CPUAndGPU",
			Self::CPUOnly => "CPUOnly"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLModelFormat {
	/// Requires Core ML 5 or later (iOS 15+ or macOS 12+).
	MLProgram,
	/// Default; requires Core ML 3 or later (iOS 13+ or macOS 10.15+).
	NeuralNetwork
}

impl CoreMLModelFormat {
	pub(crate) fn as_str(&self) -> &'static str {
		match self {
			Self::MLProgram => "MLProgram",
			Self::NeuralNetwork => "NeuralNetwork"
		}
	}
}

/// [CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) for hardware
/// acceleration on Apple devices.
#[derive(Debug, Default, Clone)]
pub struct CoreMLExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; CoreMLExecutionProvider);

impl CoreMLExecutionProvider {
	/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a `Loop`, `Scan` or `If`
	/// operator).
	///
	/// ```
	/// # use ort::{execution_providers::coreml::CoreMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_subgraphs(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_subgraphs(mut self, enable: bool) -> Self {
		self.options.set("EnableOnSubgraphs", if enable { "1" } else { "0" });
		self
	}

	/// Only allow the CoreML EP to take nodes with inputs that have static shapes. By default the CoreML EP will also
	/// allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::CoreMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_static_input_shapes(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_static_input_shapes(mut self, enable: bool) -> Self {
		self.options.set("RequireStaticInputShapes", if enable { "1" } else { "0" });
		self
	}

	/// Configures the format of the CoreML model created by the EP.
	///
	/// The default format, [NeuralNetwork](`CoreMLModelFormat::NeuralNetwork`), has better compatibility with older
	/// versions of macOS/iOS. The newer [MLProgram](`CoreMLModelFormat::MLProgram`) format supports more operators,
	/// and may be more performant.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::{CoreMLExecutionProvider, CoreMLModelFormat}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_model_format(CoreMLModelFormat::MLProgram).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_model_format(mut self, model_format: CoreMLModelFormat) -> Self {
		self.options.set("ModelFormat", model_format.as_str());
		self
	}

	/// Configures the specialization strategy.
	///
	/// CoreML segments the model's compute graph and specializes each segment for the target compute device. This
	/// process can affect the model loading time and the prediction latency. You can use this option to specialize a
	/// model for faster prediction, at the potential cost of session load time and memory footprint.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::{CoreMLExecutionProvider, CoreMLSpecializationStrategy}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_specialization_strategy(mut self, strategy: CoreMLSpecializationStrategy) -> Self {
		self.options.set("SpecializationStrategy", strategy.as_str());
		self
	}

	/// Configures what hardware can be used by CoreML for acceleration.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::{CoreMLExecutionProvider, CoreMLComputeUnits}, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default()
	/// 	.with_compute_units(CoreMLComputeUnits::CPUAndNeuralEngine)
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_compute_units(mut self, units: CoreMLComputeUnits) -> Self {
		self.options.set("MLComputeUnits", units.as_str());
		self
	}

	/// Configures whether to log the hardware each operator is dispatched to and the estimated execution time; useful
	/// for debugging unexpected performance with CoreML.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::CoreMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_profile_compute_plan(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_profile_compute_plan(mut self, enable: bool) -> Self {
		self.options.set("ProfileComputePlan", if enable { "1" } else { "0" });
		self
	}

	/// Configures whether to allow low-precision (fp16) accumulation on GPU.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::CoreMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_low_precision_accumulation_on_gpu(true).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_low_precision_accumulation_on_gpu(mut self, enable: bool) -> Self {
		self.options.set("AllowLowPrecisionAccumulationOnGPU", if enable { "1" } else { "0" });
		self
	}

	/// Configures a path to cache the compiled CoreML model.
	///
	/// If caching is not enabled (the default), the model will be compiled and saved to disk on each instantiation of a
	/// session. Setting this option allows the compiled model to be reused across session loads.
	///
	/// ```
	/// # use ort::{execution_providers::coreml::CoreMLExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = CoreMLExecutionProvider::default().with_model_cache_dir("/path/to/cache").build();
	/// # Ok(())
	/// # }
	/// ```
	///
	/// ## Updating the cache
	/// The cached model will only be recompiled if the ONNX model's metadata or the structure of the graph changes. To
	/// ensure a model updates when i.e. only weights change, you can add the hash of the model file as a custom
	/// metadata option:
	/// ```python
	/// import onnx
	/// import hashlib
	///
	/// # You can use any other hash algorithms to ensure the model and its hash-value is a one-one mapping.
	/// def hash_file(file_path, algorithm='sha256', chunk_size=8192):
	/// 	hash_func = hashlib.new(algorithm)
	/// 	with open(file_path, 'rb') as file:
	/// 		while chunk := file.read(chunk_size):
	/// 		hash_func.update(chunk)
	/// 	return hash_func.hexdigest()
	///
	/// CACHE_KEY_NAME = "CACHE_KEY"
	/// model_path = "/a/b/c/model.onnx"
	/// m = onnx.load(model_path)
	///
	/// cache_key = m.metadata_props.add()
	/// cache_key.key = CACHE_KEY_NAME
	/// cache_key.value = str(hash_file(model_path))
	///
	/// onnx.save_model(m, model_path)
	/// ```
	#[must_use]
	pub fn with_model_cache_dir(mut self, path: impl ToString) -> Self {
		self.options.set("ModelCacheDirectory", path.to_string());
		self
	}
}

impl ExecutionProvider for CoreMLExecutionProvider {
	fn name(&self) -> &'static str {
		"CoreMLExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_vendor = "apple")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "coreml"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"CoreML".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];

			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
