use alloc::string::ToString;
use core::ptr;

use super::{ExecutionProvider, ExecutionProviderOptions};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder, util};

#[derive(Debug, Default, Clone)]
pub struct TensorRT(ExecutionProviderOptions);

super::impl_ep!(arbitrary; TensorRT);

impl TensorRT {
	super::define_options! {
		pub fn with_device_id(mut self, device_id: i32) -> Self = "device_id";

		pub fn with_max_workspace_size(mut self, max_size: usize) -> Self = "trt_max_workspace_size";

		pub fn with_min_subgraph_size(mut self, min_size: usize) -> Self = "trt_min_subgraph_size";

		pub fn with_max_partition_iterations(mut self, iterations: u32) -> Self = "trt_max_partition_iterations";

		pub fn with_fp16(mut self, enable: bool) -> Self = "trt_fp16_enable";

		pub fn with_bf16(mut self, enable: bool) -> Self = "trt_bf16_enable";

		pub fn with_int8(mut self, enable: bool) -> Self = "trt_int8_enable";

		pub fn with_dla(mut self, enable: bool) -> Self = "trt_dla_enable";

		pub fn with_dla_core(mut self, core: u32) -> Self = "trt_dla_core";

		pub fn with_int8_calibration_table_name(mut self, name: impl ToString) -> Self = "trt_int8_calibration_table_name";

		pub fn with_int8_use_native_calibration_table(mut self, enable: bool) -> Self = "trt_int8_use_native_calibration_table";

		pub fn with_engine_cache(mut self, enable: bool) -> Self = "trt_engine_cache_enable";

		pub fn with_engine_cache_path(mut self, path: impl ToString) -> Self = "trt_engine_cache_path";

		pub fn with_dump_subgraphs(mut self, enable: bool) -> Self = "trt_dump_subgraphs";

		pub fn with_engine_cache_prefix(mut self, prefix: impl ToString) -> Self = "trt_engine_cache_prefix";

		pub fn with_weight_stripped_engine(mut self, enable: bool) -> Self = "trt_weight_stripped_engine_enable";

		pub fn with_onnx_model_folder_path(mut self, path: impl ToString) -> Self = "trt_onnx_model_folder_path";

		pub fn with_engine_decryption(mut self, enable: bool) -> Self = "trt_engine_decryption_enable";

		pub fn with_engine_decryption_lib_path(mut self, lib_path: impl ToString) -> Self = "trt_engine_decryption_lib_path";

		pub fn with_force_sequential_engine_build(mut self, enable: bool) -> Self = "trt_force_sequential_engine_build";

		pub fn with_context_memory_sharing(mut self, enable: bool) -> Self = "trt_context_memory_sharing_enable";

		pub fn with_layer_norm_fp32_fallback(mut self, enable: bool) -> Self = "trt_layer_norm_fp32_fallback";

		pub fn with_timing_cache(mut self, enable: bool) -> Self = "trt_timing_cache_enable";

		pub fn with_timing_cache_path(mut self, path: impl ToString) -> Self = "trt_timing_cache_path";

		pub fn with_force_timing_cache(mut self, enable: bool) -> Self = "trt_force_timing_cache";

		pub fn with_detailed_build_log(mut self, enable: bool) -> Self = "trt_detailed_build_log";

		pub fn with_build_heuristics(mut self, enable: bool) -> Self = "trt_build_heuristics_enable";

		pub fn with_sparsity(mut self, enable: bool) -> Self = "trt_sparsity_enable";

		pub fn with_builder_optimization_level(mut self, level: u8) -> Self = "trt_builder_optimization_level";

		pub fn with_auxiliary_streams(mut self, streams: i8) -> Self = "trt_auxiliary_streams";

		pub fn with_tactic_sources(mut self, sources: impl ToString) -> Self = "trt_tactic_sources";

		pub fn with_extra_plugin_lib_paths(mut self, paths: impl ToString) -> Self = "trt_extra_plugin_lib_paths";

		pub fn with_profile_min_shapes(mut self, shapes: impl ToString) -> Self = "trt_profile_min_shapes";

		pub fn with_profile_max_shapes(mut self, shapes: impl ToString) -> Self = "trt_profile_max_shapes";

		pub fn with_profile_opt_shapes(mut self, shapes: impl ToString) -> Self = "trt_profile_opt_shapes";

		pub fn with_cuda_graph(mut self, enable: bool) -> Self = "trt_cuda_graph_enable";

		pub fn with_dump_ep_context_model(mut self, enable: bool) -> Self = "trt_dump_ep_context_model";

		pub fn with_ep_context_file_path(mut self, path: impl ToString) -> Self = "trt_ep_context_file_path";

		pub fn with_ep_context_embed_mode(mut self, mode: u8) -> Self = "trt_ep_context_embed_mode";

		pub fn with_engine_hw_compatible(mut self, enable: bool) -> Self = "trt_engine_hw_compatible";
	}

	/// # Safety
	/// The provided `stream` must outlive the environment/session created with the execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.0.set("has_user_compute_stream", "1");
		self.0.set("user_compute_stream", (stream as usize).to_string());
		self
	}
}

impl ExecutionProvider for TensorRT {
	fn name(&self) -> &'static str {
		"TensorrtExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		let mut trt_options: *mut ort_sys::OrtTensorRTProviderOptionsV2 = ptr::null_mut();
		ortsys![unsafe CreateTensorRTProviderOptions(&mut trt_options)?];
		let _guard = util::run_on_drop(|| {
			ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
		});

		let ffi_options = self.0.to_ffi();
		ortsys![unsafe UpdateTensorRTProviderOptions(
			trt_options,
			ffi_options.key_ptrs(),
			ffi_options.value_ptrs(),
			ffi_options.len()
		)?];

		ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(session_builder.ptr_mut(), trt_options)?];

		Ok(())
	}
}
