use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLExecutionProviderSpecializationStrategy {
	Default,
	FastPrediction
}

impl CoreMLExecutionProviderSpecializationStrategy {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::Default => "Default",
			Self::FastPrediction => "FastPrediction"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLExecutionProviderComputeUnits {
	All,
	CPUAndNeuralEngine,
	CPUAndGPU,
	CPUOnly
}

impl CoreMLExecutionProviderComputeUnits {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::All => "ALL",
			Self::CPUAndNeuralEngine => "CPUAndNeuralEngine",
			Self::CPUAndGPU => "CPUAndGPU",
			Self::CPUOnly => "CPUOnly"
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMLExecutionProviderModelFormat {
	/// Requires Core ML 5 or later (iOS 15+ or macOS 12+).
	MLProgram,
	/// Default; requires Core ML 3 or later (iOS 13+ or macOS 10.15+).
	NeuralNetwork
}

impl CoreMLExecutionProviderModelFormat {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::MLProgram => "MLProgram",
			Self::NeuralNetwork => "NeuralNetwork"
		}
	}
}

#[derive(Debug, Default, Clone)]
pub struct CoreMLExecutionProvider {
	options: ExecutionProviderOptions
}

impl CoreMLExecutionProvider {
	/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a Loop, Scan or If operator).
	#[must_use]
	pub fn with_subgraphs(mut self, enable: bool) -> Self {
		self.options.set("EnableOnSubgraphs", if enable { "1" } else { "0" });
		self
	}

	/// Only allow the CoreML EP to take nodes with inputs that have static shapes. By default the CoreML EP will also
	/// allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.
	#[must_use]
	pub fn with_static_input_shapes(mut self, enable: bool) -> Self {
		self.options.set("RequireStaticInputShapes", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_model_format(mut self, model_format: CoreMLExecutionProviderModelFormat) -> Self {
		self.options.set("ModelFormat", model_format.as_str());
		self
	}

	#[must_use]
	pub fn with_specialization_strategy(mut self, strategy: CoreMLExecutionProviderSpecializationStrategy) -> Self {
		self.options.set("SpecializationStrategy", strategy.as_str());
		self
	}

	#[must_use]
	pub fn with_compute_units(mut self, units: CoreMLExecutionProviderComputeUnits) -> Self {
		self.options.set("MLComputeUnits", units.as_str());
		self
	}

	/// This logs the hardware each operator is dispatched to and the estimated execution time.
	/// Intended for developer usage but provide useful diagnostic information if performance is not as expected.
	#[must_use]
	pub fn with_profile_compute_plan(mut self, enable: bool) -> Self {
		self.options.set("ProfileComputePlan", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_low_precision_accumulation(mut self, enable: bool) -> Self {
		self.options.set("AllowLowPrecisionAccumulationOnGPU", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for CoreMLExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<CoreMLExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: CoreMLExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for CoreMLExecutionProvider {
	fn as_str(&self) -> &'static str {
		"CoreMLExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_os = "macos", target_os = "ios"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "coreml"))]
		{
			use crate::AsPointer;

			let ffi_options = self.options.to_ffi();
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"CoreML".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
