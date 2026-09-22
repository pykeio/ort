use alloc::string::ToString;
use core::fmt;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecializationStrategy {
	/// The strategy that should work well for most applications.
	Default,
	/// Prefer the prediction latency at the potential cost of specialization time, memory footprint, and the disk space
	/// usage of specialized artifacts.
	FastPrediction
}

impl fmt::Display for SpecializationStrategy {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::Default => "Default",
			Self::FastPrediction => "FastPrediction"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnits {
	/// Enable CoreML EP for all compatible Apple devices.
	All,
	/// Enable CoreML EP for Apple devices with a compatible Neural Engine (ANE).
	CPUAndNeuralEngine,
	/// Enable CoreML EP for Apple devices with a compatible GPU.
	CPUAndGPU,
	/// Limit CoreML to running on CPU only.
	CPUOnly
}

impl fmt::Display for ComputeUnits {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::All => "ALL",
			Self::CPUAndNeuralEngine => "CPUAndNeuralEngine",
			Self::CPUAndGPU => "CPUAndGPU",
			Self::CPUOnly => "CPUOnly"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
	/// Requires Core ML 5 or later (iOS 15+ or macOS 12+).
	MLProgram,
	/// Default; requires Core ML 3 or later (iOS 13+ or macOS 10.15+).
	NeuralNetwork
}

impl fmt::Display for ModelFormat {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::MLProgram => "MLProgram",
			Self::NeuralNetwork => "NeuralNetwork"
		})
	}
}

/// [CoreML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) for hardware
/// acceleration on Apple devices.
#[derive(Debug, Default, Clone)]
pub struct CoreML(ExecutionProviderOptions);

super::impl_ep!(arbitrary; CoreML);

impl CoreML {
	super::define_options! {
		/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a `Loop`, `Scan` or `If`
		/// operator).
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_subgraphs(true).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_subgraphs(mut self, enable: bool) -> Self = "EnableOnSubgraphs";

		/// Only allow the CoreML EP to take nodes with inputs that have static shapes. By default the CoreML EP will also
		/// allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_static_input_shapes(true).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_static_input_shapes(mut self, enable: bool) -> Self = "RequireStaticInputShapes";

		/// Configures the format of the CoreML model created by the EP.
		///
		/// The default format, [NeuralNetwork](`ModelFormat::NeuralNetwork`), has better compatibility with older
		/// versions of macOS/iOS. The newer [MLProgram](`ModelFormat::MLProgram`) format supports more operators,
		/// and may be more performant.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_model_format(ep::coreml::ModelFormat::MLProgram).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_model_format(mut self, model_format: ModelFormat) -> Self = "ModelFormat";

		/// Configures the specialization strategy.
		///
		/// CoreML segments the model's compute graph and specializes each segment for the target compute device. This
		/// process can affect the model loading time and the prediction latency. You can use this option to specialize a
		/// model for faster prediction, at the potential cost of session load time and memory footprint.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default()
		/// 	.with_specialization_strategy(ep::coreml::SpecializationStrategy::FastPrediction)
		/// 	.build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_specialization_strategy(mut self, strategy: SpecializationStrategy) -> Self = "SpecializationStrategy";

		/// Configures what hardware can be used by CoreML for acceleration.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default()
		/// 	.with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
		/// 	.build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_compute_units(mut self, units: ComputeUnits) -> Self = "MLComputeUnits";

		/// Configures whether to log the hardware each operator is dispatched to and the estimated execution time; useful
		/// for debugging unexpected performance with CoreML.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_profile_compute_plan(true).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_profile_compute_plan(mut self, enable: bool) -> Self = "ProfileComputePlan";

		/// Configures whether to allow low-precision (fp16) accumulation on GPU.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_low_precision_accumulation_on_gpu(true).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_low_precision_accumulation_on_gpu(mut self, enable: bool) -> Self = "AllowLowPrecisionAccumulationOnGPU";

		/// Configures a path to cache the compiled CoreML model.
		///
		/// If caching is not enabled (the default), the model will be compiled and saved to disk on each instantiation of a
		/// session. Setting this option allows the compiled model to be reused across session loads.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::CoreML::default().with_model_cache_dir("/path/to/cache").build();
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
		/// # You can use any other hash algorithms to ensure the model and its hash-value is a one-to-one mapping.
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
		pub fn with_model_cache_dir(mut self, path: impl ToString) -> Self = "ModelCacheDirectory";
	}
}

impl SimpleExecutionProvider for CoreML {
	const CANONICAL_NAME: &'static str = "CoreMLExecutionProvider";
	const SHORT_NAME: &'static str = "CoreML";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
