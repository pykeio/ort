use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct TensorRTExecutionProvider {
	device_id: Option<i32>,
	max_workspace_size: Option<usize>,
	max_partition_iterations: Option<u32>,
	min_subgraph_size: Option<usize>,
	fp16_enable: Option<bool>,
	int8_enable: Option<bool>,
	int8_calibration_table_name: Option<String>,
	int8_use_native_calibration_table: Option<bool>,
	dla_enable: Option<bool>,
	dla_core: Option<u32>,
	engine_cache_enable: Option<bool>,
	engine_cache_path: Option<String>,
	dump_subgraphs: Option<bool>,
	force_sequential_engine_build: Option<bool>,
	enable_context_memory_sharing: Option<bool>,
	layer_norm_fp32_fallback: Option<bool>,
	timing_cache_enable: Option<bool>,
	force_timing_cache: Option<bool>,
	detailed_build_log: Option<bool>,
	enable_build_heuristics: Option<bool>,
	enable_sparsity: Option<bool>,
	builder_optimization_level: Option<u8>,
	auxiliary_streams: Option<i8>,
	tactic_sources: Option<String>,
	extra_plugin_lib_paths: Option<String>,
	profile_min_shapes: Option<String>,
	profile_max_shapes: Option<String>,
	profile_opt_shapes: Option<String>
}

impl TensorRTExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = Some(device_id);
		self
	}

	#[must_use]
	pub fn with_max_workspace_size(mut self, max_size: usize) -> Self {
		self.max_workspace_size = Some(max_size);
		self
	}

	#[must_use]
	pub fn with_min_subgraph_size(mut self, min_size: usize) -> Self {
		self.min_subgraph_size = Some(min_size);
		self
	}

	#[must_use]
	pub fn with_max_partition_iterations(mut self, iterations: u32) -> Self {
		self.max_partition_iterations = Some(iterations);
		self
	}

	#[must_use]
	pub fn with_fp16(mut self, enable: bool) -> Self {
		self.fp16_enable = Some(enable);
		self
	}

	#[must_use]
	pub fn with_int8(mut self, enable: bool) -> Self {
		self.int8_enable = Some(enable);
		self
	}

	#[must_use]
	pub fn with_dla(mut self, enable: bool) -> Self {
		self.dla_enable = Some(enable);
		self
	}

	#[must_use]
	pub fn with_dla_core(mut self, core: u32) -> Self {
		self.dla_core = Some(core);
		self
	}

	#[must_use]
	pub fn with_int8_calibration_table_name(mut self, name: impl ToString) -> Self {
		self.int8_calibration_table_name = Some(name.to_string());
		self
	}

	#[must_use]
	pub fn with_int8_use_native_calibration_table(mut self, enable: bool) -> Self {
		self.int8_use_native_calibration_table = Some(enable);
		self
	}

	#[must_use]
	pub fn with_engine_cache(mut self, enable: bool) -> Self {
		self.engine_cache_enable = Some(enable);
		self
	}

	#[must_use]
	pub fn with_engine_cache_path(mut self, path: impl ToString) -> Self {
		self.engine_cache_path = Some(path.to_string());
		self
	}

	#[must_use]
	pub fn with_dump_subgraphs(mut self, enable: bool) -> Self {
		self.dump_subgraphs = Some(enable);
		self
	}

	#[must_use]
	pub fn with_force_sequential_engine_build(mut self, enable: bool) -> Self {
		self.force_sequential_engine_build = Some(enable);
		self
	}

	#[must_use]
	pub fn with_context_memory_sharing(mut self, enable: bool) -> Self {
		self.enable_context_memory_sharing = Some(enable);
		self
	}

	#[must_use]
	pub fn with_layer_norm_fp32_fallback(mut self, enable: bool) -> Self {
		self.layer_norm_fp32_fallback = Some(enable);
		self
	}

	#[must_use]
	pub fn with_timing_cache(mut self, enable: bool) -> Self {
		self.timing_cache_enable = Some(enable);
		self
	}

	#[must_use]
	pub fn with_force_timing_cache(mut self, enable: bool) -> Self {
		self.force_timing_cache = Some(enable);
		self
	}

	#[must_use]
	pub fn with_detailed_build_log(mut self, enable: bool) -> Self {
		self.detailed_build_log = Some(enable);
		self
	}

	#[must_use]
	pub fn with_build_heuristics(mut self, enable: bool) -> Self {
		self.enable_build_heuristics = Some(enable);
		self
	}

	#[must_use]
	pub fn with_sparsity(mut self, enable: bool) -> Self {
		self.enable_sparsity = Some(enable);
		self
	}

	#[must_use]
	pub fn with_builder_optimization_level(mut self, level: u8) -> Self {
		self.builder_optimization_level = Some(level);
		self
	}

	#[must_use]
	pub fn with_auxiliary_streams(mut self, streams: i8) -> Self {
		self.auxiliary_streams = Some(streams);
		self
	}

	#[must_use]
	pub fn with_tactic_sources(mut self, sources: impl ToString) -> Self {
		self.tactic_sources = Some(sources.to_string());
		self
	}

	#[must_use]
	pub fn with_extra_plugin_lib_paths(mut self, paths: impl ToString) -> Self {
		self.extra_plugin_lib_paths = Some(paths.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_min_shapes(mut self, shapes: impl ToString) -> Self {
		self.profile_min_shapes = Some(shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_max_shapes(mut self, shapes: impl ToString) -> Self {
		self.profile_max_shapes = Some(shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_opt_shapes(mut self, shapes: impl ToString) -> Self {
		self.profile_opt_shapes = Some(shapes.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<TensorRTExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: TensorRTExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for TensorRTExecutionProvider {
	fn as_str(&self) -> &'static str {
		"TensorrtExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "tensorrt"))]
		{
			// The TensorRT execution provider specifically is pretty picky about requiring an environment to be initialized by the
			// time we register it. This isn't always the case in `ort`, so if we get to this point, let's make sure we have an
			// environment initialized.
			let _ = crate::get_environment();

			let mut trt_options: *mut ort_sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
			crate::error::status_to_result(crate::ortsys![unsafe CreateTensorRTProviderOptions(&mut trt_options)]).map_err(Error::ExecutionProvider)?;
			let (key_ptrs, value_ptrs, len, keys, values) = super::map_keys! {
				device_id = self.device_id,
				trt_max_workspace_size = self.max_workspace_size,
				trt_max_partition_iterations = self.max_partition_iterations,
				trt_min_subgraph_size = self.min_subgraph_size,
				trt_fp16_enable = self.fp16_enable.map(<bool as Into<i32>>::into),
				trt_int8_enable = self.int8_enable.map(<bool as Into<i32>>::into),
				trt_int8_use_native_calibration_table = self.int8_use_native_calibration_table.map(<bool as Into<i32>>::into),
				trt_int8_calibration_table_name = self.int8_calibration_table_name.clone(),
				trt_dla_enable = self.dla_enable.map(<bool as Into<i32>>::into),
				trt_dla_core = self.dla_core,
				trt_engine_cache_enable = self.engine_cache_enable.map(<bool as Into<i32>>::into),
				trt_engine_cache_path = self.engine_cache_path.clone(),
				trt_dump_subgraphs = self.dump_subgraphs.map(<bool as Into<i32>>::into),
				trt_force_sequential_engine_build = self.force_sequential_engine_build.map(<bool as Into<i32>>::into),
				trt_context_memory_sharing_enable = self.enable_context_memory_sharing.map(<bool as Into<i32>>::into),
				trt_layer_norm_fp32_fallback = self.layer_norm_fp32_fallback.map(<bool as Into<i32>>::into),
				trt_timing_cache_enable = self.timing_cache_enable.map(<bool as Into<i32>>::into),
				// https://github.com/microsoft/onnxruntime/pull/17956
				trt_force_timing_cache = self.force_timing_cache.map(<bool as Into<i32>>::into),
				trt_detailed_build_log = self.detailed_build_log.map(<bool as Into<i32>>::into),
				trt_build_heuristics_enable = self.enable_build_heuristics.map(<bool as Into<i32>>::into),
				trt_sparsity_enable = self.enable_sparsity.map(<bool as Into<i32>>::into),
				trt_builder_optimization_level = self.builder_optimization_level,
				trt_auxiliary_streams = self.auxiliary_streams,
				trt_tactic_sources = self.tactic_sources.clone(),
				trt_extra_plugin_lib_paths = self.extra_plugin_lib_paths.clone(),
				trt_profile_min_shapes = self.profile_min_shapes.clone(),
				trt_profile_max_shapes = self.profile_max_shapes.clone(),
				trt_profile_opt_shapes = self.profile_opt_shapes.clone()
			};
			if let Err(e) = crate::error::status_to_result(
				crate::ortsys![unsafe UpdateTensorRTProviderOptions(trt_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), len as _)]
			)
			.map_err(Error::ExecutionProvider)
			{
				crate::ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
				std::mem::drop((keys, values));
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(session_builder.session_options_ptr.as_ptr(), trt_options)];
			crate::ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
			std::mem::drop((keys, values));
			return crate::error::status_to_result(status).map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
