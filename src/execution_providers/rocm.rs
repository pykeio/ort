use alloc::{format, string::ToString};
use core::ffi::c_void;

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct ROCmExecutionProvider {
	options: ExecutionProviderOptions
}

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
		self.options.set("arena_extend_strategy", match strategy {
			ArenaExtendStrategy::NextPowerOfTwo => "kNextPowerOfTwo",
			ArenaExtendStrategy::SameAsRequested => "kSameAsRequested"
		});
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

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for ROCmExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<ROCmExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: ROCmExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for ROCmExecutionProvider {
	fn as_str(&self) -> &'static str {
		"ROCmExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", target_os = "linux"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "rocm"))]
		{
			use crate::AsPointer;

			let mut rocm_options: *mut ort_sys::OrtROCMProviderOptions = core::ptr::null_mut();
			crate::ortsys![unsafe CreateROCMProviderOptions(&mut rocm_options)?];
			let ffi_options = self.options.to_ffi();

			let res = crate::ortsys![unsafe UpdateROCMProviderOptions(
				rocm_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)];
			if let Err(e) = unsafe { crate::error::status_to_result(res) } {
				crate::ortsys![unsafe ReleaseROCMProviderOptions(rocm_options)];
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_ROCM(session_builder.ptr_mut(), rocm_options)];
			crate::ortsys![unsafe ReleaseROCMProviderOptions(rocm_options)];
			return unsafe { crate::error::status_to_result(status) };

			// use core::ptr;

			// use crate::AsPointer;

			// let rocm_options = ort_sys::OrtROCMProviderOptions {
			// 	device_id: self.device_id,
			// 	miopen_conv_exhaustive_search: self.miopen_conv_exhaustive_search.into(),
			// 	gpu_mem_limit: self.gpu_mem_limit,
			// 	arena_extend_strategy: match self.arena_extend_strategy {
			// 		ArenaExtendStrategy::NextPowerOfTwo => 0,
			// 		ArenaExtendStrategy::SameAsRequested => 1
			// 	},
			// 	do_copy_in_default_stream: self.do_copy_in_default_stream.into(),
			// 	has_user_compute_stream: self.user_compute_stream.is_some().into(),
			// 	user_compute_stream: self.user_compute_stream.unwrap_or_else(ptr::null_mut),
			// 	default_memory_arena_cfg: self.default_memory_arena_cfg.unwrap_or_else(ptr::null_mut),
			// 	enable_hip_graph: self.enable_hip_graph.into(),
			// 	tunable_op_enable: self.tunable_op_enable.into(),
			// 	tunable_op_tuning_enable: self.tunable_op_tuning_enable.into(),
			// 	tunable_op_max_tuning_duration_ms: self.tunable_op_max_tuning_duration_ms
			// };
			// crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_ROCM(session_builder.ptr_mut(),
			// ptr::addr_of!(rocm_options))?]; return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
