use alloc::format;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "coreml"))]
extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut ort_sys::OrtSessionOptions, flags: u32) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct CoreMLExecutionProvider {
	use_cpu_only: bool,
	enable_on_subgraph: bool,
	only_enable_device_with_ane: bool,
	only_static_input_shapes: bool,
	mlprogram: bool,
	use_cpu_and_gpu: bool
}

impl CoreMLExecutionProvider {
	/// Limit CoreML to running on CPU only. This may decrease the performance but will provide reference output value
	/// without precision loss, which is useful for validation.
	#[must_use]
	pub fn with_cpu_only(mut self, enable: bool) -> Self {
		self.use_cpu_only = enable;
		self
	}

	/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a Loop, Scan or If operator).
	#[must_use]
	pub fn with_subgraphs(mut self, enable: bool) -> Self {
		self.enable_on_subgraph = enable;
		self
	}

	/// By default the CoreML EP will be enabled for all compatible Apple devices. Setting this option will only enable
	/// CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE). Note, enabling this option does not
	/// guarantee the entire model to be executed using ANE only.
	#[must_use]
	pub fn with_ane_only(mut self, enable: bool) -> Self {
		self.only_enable_device_with_ane = enable;
		self
	}

	/// Only allow the CoreML EP to take nodes with inputs that have static shapes. By default the CoreML EP will also
	/// allow inputs with dynamic shapes, however performance may be negatively impacted by inputs with dynamic shapes.
	#[must_use]
	pub fn with_static_input_shapes(mut self, enable: bool) -> Self {
		self.only_static_input_shapes = enable;
		self
	}

	/// Create an MLProgram format model. Requires Core ML 5 or later (iOS 15+ or macOS 12+). The default is for a
	/// NeuralNetwork model to be created as that requires Core ML 3 or later (iOS 13+ or macOS 10.15+).
	#[must_use]
	pub fn with_mlprogram(mut self, enable: bool) -> Self {
		self.mlprogram = enable;
		self
	}

	#[must_use]
	pub fn with_cpu_and_gpu(mut self, enable: bool) -> Self {
		self.use_cpu_and_gpu = enable;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
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

			super::get_ep_register!(OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut ort_sys::OrtSessionOptions, flags: u32) -> ort_sys::OrtStatusPtr);
			let mut flags = 0;
			if self.use_cpu_only {
				flags |= 0x001;
			}
			if self.enable_on_subgraph {
				flags |= 0x002;
			}
			if self.only_enable_device_with_ane {
				flags |= 0x004;
			}
			if self.only_static_input_shapes {
				flags |= 0x008;
			}
			if self.mlprogram {
				flags |= 0x010;
			}
			if self.use_cpu_and_gpu {
				flags |= 0x020;
			}
			return unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_CoreML(session_builder.ptr_mut(), flags)) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
