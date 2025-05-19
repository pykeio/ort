use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [OpenVINO execution provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) for
/// Intel CPUs/GPUs/NPUs.
#[derive(Default, Debug, Clone)]
pub struct OpenVINOExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; OpenVINOExecutionProvider);

impl OpenVINOExecutionProvider {
	/// Overrides the accelerator hardware type and precision.
	///
	/// `device_type` should be in the format `CPU`, `NPU`, `GPU`, `GPU.0`, `GPU.1`, etc. Heterogenous combinations are
	/// also supported in the format `HETERO:NPU,GPU`.
	///
	/// ```
	/// # use ort::{execution_providers::openvino::OpenVINOExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = OpenVINOExecutionProvider::default().with_device_type("GPU.0").build();
	/// # Ok(())
	/// # }
	/// ```
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
}

impl ExecutionProvider for OpenVINOExecutionProvider {
	fn name(&self) -> &'static str {
		"OpenVINOExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", any(target_os = "windows", target_os = "linux")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "openvino"))]
		{
			use alloc::ffi::CString;
			use core::ffi::c_char;

			use crate::{AsPointer, environment::get_environment, ortsys};

			// Like TensorRT, the OpenVINO EP is also pretty picky about needing an environment by this point.
			let _ = get_environment();

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider_OpenVINO_V2(
				session_builder.ptr_mut(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
