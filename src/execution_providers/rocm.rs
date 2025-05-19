use alloc::string::ToString;
use core::ffi::c_void;

use super::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct ROCmExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; ROCmExecutionProvider);

impl ROCmExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	#[must_use]
	pub fn with_exhaustive_conv_search(mut self, enable: bool) -> Self {
		self.options.set("miopen_conv_exhaustive_search", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_conv_use_max_workspace(mut self, enable: bool) -> Self {
		self.options.set("miopen_conv_use_max_workspace", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_mem_limit(mut self, limit: usize) -> Self {
		self.options.set("gpu_mem_limit", limit.to_string());
		self
	}

	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.options.set(
			"arena_extend_strategy",
			match strategy {
				ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
				ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
			}
		);
		self
	}

	#[must_use]
	pub fn with_copy_in_default_stream(mut self, enable: bool) -> Self {
		self.options.set("do_copy_in_default_stream", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_compute_stream(mut self, ptr: *mut c_void) -> Self {
		self.options.set("has_user_compute_stream", "1");
		self.options.set("user_compute_stream", (ptr as usize).to_string());
		self
	}

	#[must_use]
	pub fn with_hip_graph(mut self, enable: bool) -> Self {
		self.options.set("enable_hip_graph", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_tunable_op(mut self, enable: bool) -> Self {
		self.options.set("tunable_op_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_tuning(mut self, enable: bool) -> Self {
		self.options.set("tunable_op_tuning_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_max_tuning_duration(mut self, ms: i32) -> Self {
		self.options.set("tunable_op_max_tuning_duration_ms", ms.to_string());
		self
	}
}

impl ExecutionProvider for ROCmExecutionProvider {
	fn name(&self) -> &'static str {
		"ROCMExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", target_os = "linux"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "rocm"))]
		{
			use core::ptr;

			use crate::{AsPointer, ortsys, util};

			let mut rocm_options: *mut ort_sys::OrtROCMProviderOptions = core::ptr::null_mut();
			ortsys![unsafe CreateROCMProviderOptions(&mut rocm_options)?];
			let _guard = util::run_on_drop(|| {
				ortsys![unsafe ReleaseROCMProviderOptions(rocm_options)];
			});

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe UpdateROCMProviderOptions(
				rocm_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			ortsys![unsafe SessionOptionsAppendExecutionProvider_ROCM(session_builder.ptr_mut(), rocm_options)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
