use alloc::string::ToString;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Default, Clone)]
pub struct NVRTX(ExecutionProviderOptions);

super::impl_ep!(arbitrary; NVRTX);

impl NVRTX {
	super::define_options! {
		pub fn with_device_id(mut self, device_id: u32) -> Self = "device_id";

		pub fn with_cuda_graph(mut self, enable: bool) -> Self = "enable_cuda_graph";

		pub fn with_max_workspace(mut self, limit: usize) -> Self = "nv_max_workspace_size";

		pub fn with_max_shared_mem(mut self, limit: usize) -> Self = "nv_max_shared_mem_size";

		pub fn with_profile_min_shapes(mut self, shapes: impl ToString) -> Self = "nv_profile_min_shapes";

		pub fn with_profile_max_shapes(mut self, shapes: impl ToString) -> Self = "nv_profile_max_shapes";

		pub fn with_profile_opt_shapes(mut self, shapes: impl ToString) -> Self = "nv_profile_opt_shapes";

		pub fn with_multi_profile(mut self, enable: bool) -> Self = "nv_multi_profile_enable";

		pub fn with_runtime_cache_path(mut self, path: impl ToString) -> Self = "nv_runtime_cache_path";
	}

	/// Use a custom CUDA device stream rather than the default one.
	///
	/// # Safety
	/// The provided `stream` must outlive the environment/session configured to use this execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.0.set("has_user_compute_stream", "1");
		self.0.set("user_compute_stream", (stream as usize).to_string());
		self
	}
}

impl SimpleExecutionProvider for NVRTX {
	const CANONICAL_NAME: &'static str = "NvTensorRTRTXExecutionProvider";
	const SHORT_NAME: &'static str = "NvTensorRtRtx";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
