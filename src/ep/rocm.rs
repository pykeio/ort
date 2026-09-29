use alloc::string::ToString;
use core::ffi::c_void;

use super::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderOptions};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct ROCm(ExecutionProviderOptions);

super::impl_ep!(arbitrary; ROCm);

impl ROCm {
	super::define_options! {
		pub fn with_device_id(mut self, device_id: i32) -> Self = "device_id";

		pub fn with_exhaustive_conv_search(mut self, enable: bool) -> Self = "miopen_conv_exhaustive_search";

		pub fn with_conv_use_max_workspace(mut self, enable: bool) -> Self = "miopen_conv_use_max_workspace";

		pub fn with_mem_limit(mut self, limit: usize) -> Self = "gpu_mem_limit";

		pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self = "arena_extend_strategy";

		pub fn with_copy_in_default_stream(mut self, enable: bool) -> Self = "do_copy_in_default_stream";

		pub fn with_hip_graph(mut self, enable: bool) -> Self = "enable_hip_graph";

		pub fn with_tunable_op(mut self, enable: bool) -> Self = "tunable_op_enable";

		pub fn with_tuning(mut self, enable: bool) -> Self = "tunable_op_tuning_enable";

		pub fn with_max_tuning_duration(mut self, ms: i32) -> Self = "tunable_op_max_tuning_duration_ms";
	}

	#[must_use]
	pub fn with_compute_stream(mut self, ptr: *mut c_void) -> Self {
		self.0.set("has_user_compute_stream", "1");
		self.0.set("user_compute_stream", (ptr as usize).to_string());
		self
	}
}

impl ExecutionProvider for ROCm {
	const NAME: &'static str = "ROCMExecutionProvider";

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::register_with_options_struct!(
			self.0,
			session_builder,
			ort_sys::OrtROCMProviderOptions,
			CreateROCMProviderOptions,
			UpdateROCMProviderOptions,
			ReleaseROCMProviderOptions,
			SessionOptionsAppendExecutionProvider_ROCM
		)
	}
}
