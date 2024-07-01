use std::os::raw::c_void;

use crate::{
	error::{Error, Result},
	execution_providers::{ArenaExtendStrategy, ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Clone)]
pub struct ROCmExecutionProvider {
	device_id: i32,
	miopen_conv_exhaustive_search: bool,
	gpu_mem_limit: ort_sys::size_t,
	arena_extend_strategy: ArenaExtendStrategy,
	do_copy_in_default_stream: bool,
	user_compute_stream: Option<*mut c_void>,
	default_memory_arena_cfg: Option<*mut ort_sys::OrtArenaCfg>,
	enable_hip_graph: bool,
	tunable_op_enable: bool,
	tunable_op_tuning_enable: bool,
	tunable_op_max_tuning_duration_ms: i32
}

unsafe impl Send for ROCmExecutionProvider {}
unsafe impl Sync for ROCmExecutionProvider {}

impl Default for ROCmExecutionProvider {
	fn default() -> Self {
		Self {
			device_id: 0,
			miopen_conv_exhaustive_search: false,
			gpu_mem_limit: ort_sys::size_t::MAX,
			arena_extend_strategy: ArenaExtendStrategy::NextPowerOfTwo,
			do_copy_in_default_stream: true,
			user_compute_stream: None,
			default_memory_arena_cfg: None,
			enable_hip_graph: false,
			tunable_op_enable: false,
			tunable_op_tuning_enable: false,
			tunable_op_max_tuning_duration_ms: 0
		}
	}
}

impl ROCmExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = device_id;
		self
	}

	#[must_use]
	pub fn with_exhaustive_conv_search(mut self) -> Self {
		self.miopen_conv_exhaustive_search = true;
		self
	}

	#[must_use]
	pub fn with_mem_limit(mut self, limit: usize) -> Self {
		self.gpu_mem_limit = limit as _;
		self
	}

	#[must_use]
	pub fn with_arena_extend_strategy(mut self, strategy: ArenaExtendStrategy) -> Self {
		self.arena_extend_strategy = strategy;
		self
	}

	#[must_use]
	pub fn with_copy_in_default_stream(mut self, enable: bool) -> Self {
		self.do_copy_in_default_stream = enable;
		self
	}

	#[must_use]
	pub fn with_compute_stream(mut self, ptr: *mut c_void) -> Self {
		self.user_compute_stream = Some(ptr);
		self
	}

	#[must_use]
	pub fn with_default_memory_arena_cfg(mut self, cfg: *mut ort_sys::OrtArenaCfg) -> Self {
		self.default_memory_arena_cfg = Some(cfg);
		self
	}

	#[must_use]
	pub fn with_hip_graph(mut self, enable: bool) -> Self {
		self.enable_hip_graph = enable;
		self
	}

	#[must_use]
	pub fn with_tunable_op(mut self, enable: bool) -> Self {
		self.tunable_op_enable = enable;
		self
	}

	#[must_use]
	pub fn with_tuning(mut self, enable: bool) -> Self {
		self.tunable_op_tuning_enable = enable;
		self
	}

	#[must_use]
	pub fn with_max_tuning_duration(mut self, ms: i32) -> Self {
		self.tunable_op_max_tuning_duration_ms = ms;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "rocm"))]
		{
			let rocm_options = ort_sys::OrtROCMProviderOptions {
				device_id: self.device_id,
				miopen_conv_exhaustive_search: self.miopen_conv_exhaustive_search.into(),
				gpu_mem_limit: self.gpu_mem_limit as _,
				arena_extend_strategy: match self.arena_extend_strategy {
					ArenaExtendStrategy::NextPowerOfTwo => 0,
					ArenaExtendStrategy::SameAsRequested => 1
				},
				do_copy_in_default_stream: self.do_copy_in_default_stream.into(),
				has_user_compute_stream: self.user_compute_stream.is_some().into(),
				user_compute_stream: self.user_compute_stream.unwrap_or_else(std::ptr::null_mut),
				default_memory_arena_cfg: self.default_memory_arena_cfg.unwrap_or_else(std::ptr::null_mut),
				enable_hip_graph: self.enable_hip_graph.into(),
				tunable_op_enable: self.tunable_op_enable.into(),
				tunable_op_tuning_enable: self.tunable_op_tuning_enable.into(),
				tunable_op_max_tuning_duration_ms: self.tunable_op_max_tuning_duration_ms
			};
			return crate::error::status_to_result(
				crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_ROCM(session_builder.session_options_ptr.as_ptr(), std::ptr::addr_of!(rocm_options))]
			)
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
