use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct NVRTX {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; NVRTX);

impl NVRTX {
	#[must_use]
	pub fn with_device_id(mut self, device_id: u32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// Use a custom CUDA device stream rather than the default one.
	///
	/// # Safety
	/// The provided `stream` must outlive the environment/session configured to use this execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.options.set("has_user_compute_stream", "1");
		self.options.set("user_compute_stream", (stream as usize).to_string());
		self
	}

	#[must_use]
	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_cuda_graph", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_max_workspace(mut self, limit: usize) -> Self {
		self.options.set("nv_max_workspace_size", limit.to_string());
		self
	}

	#[must_use]
	pub fn with_max_shared_mem(mut self, limit: usize) -> Self {
		self.options.set("nv_max_shared_mem_size", limit.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_min_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("nv_profile_min_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_max_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("nv_profile_max_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_opt_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("nv_profile_opt_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_multi_profile(mut self, enable: bool) -> Self {
		self.options.set("nv_multi_profile_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_runtime_cache_path(mut self, path: impl ToString) -> Self {
		self.options.set("nv_runtime_cache_path", path.to_string());
		self
	}
}

impl ExecutionProvider for NVRTX {
	fn name(&self) -> &'static str {
		"NvTensorRTRTXExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_os = "windows", target_arch = "x86_64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "nvrtx"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"NvTensorRtRtx".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
