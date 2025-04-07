use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OpenVINOModelPriority {
	Low,
	Medium,
	High,
	Default
}

impl OpenVINOModelPriority {
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::Low => "LOW",
			Self::Medium => "MEDIUM",
			Self::High => "HIGH",
			Self::Default => "DEFAULT"
		}
	}
}

#[derive(Default, Debug, Clone)]
pub struct OpenVINOExecutionProvider {
	options: ExecutionProviderOptions
}

unsafe impl Send for OpenVINOExecutionProvider {}
unsafe impl Sync for OpenVINOExecutionProvider {}

impl OpenVINOExecutionProvider {
	/// Overrides the accelerator hardware type and precision with these values at runtime. If this option is not
	/// explicitly set, default hardware and precision specified during build time is used.
	#[must_use]
	pub fn with_device_type(mut self, device_type: impl AsRef<str>) -> Self {
		self.options.set("device_type", device_type.as_ref());
		self
	}

	/// Overrides the accelerator default value of number of threads with this value at runtime. If this option is not
	/// explicitly set, default value of 8 is used during build time.
	#[must_use]
	pub fn with_num_threads(mut self, num_threads: usize) -> Self {
		self.options.set("num_of_threads", num_threads.to_string());
		self
	}

	/// Explicitly specify the path to save and load the blobs, enabling model caching.
	#[must_use]
	pub fn with_cache_dir(mut self, dir: impl AsRef<str>) -> Self {
		self.options.set("cache_dir", dir.as_ref());
		self
	}

	/// This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU).
	#[must_use]
	pub fn with_opencl_throttling(mut self, enable: bool) -> Self {
		self.options.set("enable_opencl_throttling", if enable { "true" } else { "false" });
		self
	}

	#[must_use]
	pub fn with_qdq_optimizer(mut self, enable: bool) -> Self {
		self.options.set("enable_qdq_optimizer", if enable { "true" } else { "false" });
		self
	}

	/// This option if enabled works for dynamic shaped models whose shape will be set dynamically based on the infer
	/// input image/data shape at run time in CPU. This gives best result for running multiple inferences with varied
	/// shaped images/data.
	#[must_use]
	pub fn with_dynamic_shapes(mut self, enable: bool) -> Self {
		self.options.set("disable_dynamic_shapes", if enable { "false" } else { "true" });
		self
	}

	#[must_use]
	pub fn with_num_streams(mut self, num_streams: u8) -> Self {
		self.options.set("num_streams", num_streams.to_string());
		self
	}

	#[must_use]
	pub fn with_precision(mut self, precision: impl AsRef<str>) -> Self {
		self.options.set("precision", precision.as_ref());
		self
	}

	#[must_use]
	pub fn with_model_priority(mut self, priority: OpenVINOModelPriority) -> Self {
		self.options.set("model_priority", priority.as_str());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for OpenVINOExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<OpenVINOExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: OpenVINOExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for OpenVINOExecutionProvider {
	fn as_str(&self) -> &'static str {
		"OpenVINOExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", any(target_os = "windows", target_os = "linux")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "openvino"))]
		{
			use alloc::ffi::CString;
			use core::ffi::c_char;

			use crate::AsPointer;

			// Like TensorRT, the OpenVINO EP is also pretty picky about needing an environment by this point.
			let _ = crate::environment::get_environment();

			let ffi_options = self.options.to_ffi();
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_OpenVINO_V2(
				session_builder.ptr_mut(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
