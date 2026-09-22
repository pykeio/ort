use alloc::string::ToString;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

/// [OpenVINO execution provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) for
/// Intel CPUs/GPUs/NPUs.
#[derive(Default, Debug, Clone)]
pub struct OpenVINO(ExecutionProviderOptions);

super::impl_ep!(arbitrary; OpenVINO);

impl OpenVINO {
	super::define_options! {
		/// Overrides the accelerator hardware type and precision.
		///
		/// `device_type` should be in the format `CPU`, `NPU`, `GPU`, `GPU.0`, `GPU.1`, etc. Heterogenous combinations are
		/// also supported in the format `HETERO:NPU,GPU`.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::OpenVINO::default().with_device_type("GPU.0").build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_device_type(mut self, device_type: impl ToString) -> Self = "device_type";

		/// Overrides the accelerator default value of number of threads with this value at runtime. If this option is not
		/// explicitly set, default value of 8 is used during build time.
		pub fn with_num_threads(mut self, num_threads: usize) -> Self = "num_of_threads";

		/// Explicitly specify the path to save and load the blobs, enabling model caching.
		pub fn with_cache_dir(mut self, dir: impl ToString) -> Self = "cache_dir";

		pub fn with_num_streams(mut self, num_streams: u8) -> Self = "num_streams";

		pub fn with_precision(mut self, precision: impl ToString) -> Self = "precision";
	}

	// OpenVINO is the only EP that doesn't accept 1/0 for boolean options, so we have to define these separately

	/// This option enables OpenCL queue throttling for GPU devices (reduces CPU utilization when using GPU).
	#[must_use]
	pub fn with_opencl_throttling(mut self, enable: bool) -> Self {
		self.0.set("enable_opencl_throttling", if enable { "true" } else { "false" });
		self
	}

	#[must_use]
	pub fn with_qdq_optimizer(mut self, enable: bool) -> Self {
		self.0.set("enable_qdq_optimizer", if enable { "true" } else { "false" });
		self
	}

	/// This option if enabled works for dynamic shaped models whose shape will be set dynamically based on the infer
	/// input image/data shape at run time in CPU. This gives best result for running multiple inferences with varied
	/// shaped images/data.
	#[must_use]
	pub fn with_dynamic_shapes(mut self, enable: bool) -> Self {
		self.0.set("disable_dynamic_shapes", if enable { "false" } else { "true" });
		self
	}
}

impl SimpleExecutionProvider for OpenVINO {
	const CANONICAL_NAME: &'static str = "OpenVINOExecutionProvider";
	const SHORT_NAME: &'static str = "OpenVINO";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
