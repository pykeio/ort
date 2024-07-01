use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[cfg(all(not(feature = "load-dynamic"), feature = "coreml"))]
extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CoreML(options: *mut ort_sys::OrtSessionOptions, flags: u32) -> ort_sys::OrtStatusPtr;
}

#[derive(Debug, Default, Clone)]
pub struct CoreMLExecutionProvider {
	use_cpu_only: bool,
	enable_on_subgraph: bool,
	only_enable_device_with_ane: bool
}

impl CoreMLExecutionProvider {
	/// Limit CoreML to running on CPU only. This may decrease the performance but will provide reference output value
	/// without precision loss, which is useful for validation.
	#[must_use]
	pub fn with_cpu_only(mut self) -> Self {
		self.use_cpu_only = true;
		self
	}

	/// Enable CoreML EP to run on a subgraph in the body of a control flow operator (i.e. a Loop, Scan or If operator).
	#[must_use]
	pub fn with_subgraphs(mut self) -> Self {
		self.enable_on_subgraph = true;
		self
	}

	/// By default the CoreML EP will be enabled for all compatible Apple devices. Setting this option will only enable
	/// CoreML EP for Apple devices with a compatible Apple Neural Engine (ANE). Note, enabling this option does not
	/// guarantee the entire model to be executed using ANE only.
	#[must_use]
	pub fn with_ane_only(mut self) -> Self {
		self.only_enable_device_with_ane = true;
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "coreml"))]
		{
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
			return crate::error::status_to_result(unsafe {
				OrtSessionOptionsAppendExecutionProvider_CoreML(session_builder.session_options_ptr.as_ptr(), flags)
			})
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
