use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct TensorRTExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; TensorRTExecutionProvider);

impl TensorRTExecutionProvider {
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.options.set("device_id", device_id.to_string());
		self
	}

	/// # Safety
	/// The provided `stream` must outlive the environment/session created with the execution provider.
	#[must_use]
	pub unsafe fn with_compute_stream(mut self, stream: *mut ()) -> Self {
		self.options.set("user_compute_stream", (stream as usize).to_string());
		self
	}

	#[must_use]
	pub fn with_max_workspace_size(mut self, max_size: usize) -> Self {
		self.options.set("trt_max_workspace_size", max_size.to_string());
		self
	}

	#[must_use]
	pub fn with_min_subgraph_size(mut self, min_size: usize) -> Self {
		self.options.set("trt_min_subgraph_size", min_size.to_string());
		self
	}

	#[must_use]
	pub fn with_max_partition_iterations(mut self, iterations: u32) -> Self {
		self.options.set("trt_max_partition_iterations", iterations.to_string());
		self
	}

	#[must_use]
	pub fn with_fp16(mut self, enable: bool) -> Self {
		self.options.set("trt_fp16_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_int8(mut self, enable: bool) -> Self {
		self.options.set("trt_int8_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_dla(mut self, enable: bool) -> Self {
		self.options.set("trt_dla_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_dla_core(mut self, core: u32) -> Self {
		self.options.set("trt_dla_core", core.to_string());
		self
	}

	#[must_use]
	pub fn with_int8_calibration_table_name(mut self, name: impl ToString) -> Self {
		self.options.set("trt_int8_calibration_table_name", name.to_string());
		self
	}

	#[must_use]
	pub fn with_int8_use_native_calibration_table(mut self, enable: bool) -> Self {
		self.options.set("trt_int8_use_native_calibration_table", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_engine_cache(mut self, enable: bool) -> Self {
		self.options.set("trt_engine_cache_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_engine_cache_path(mut self, path: impl ToString) -> Self {
		self.options.set("trt_engine_cache_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_dump_subgraphs(mut self, enable: bool) -> Self {
		self.options.set("trt_dump_subgraphs", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_engine_cache_prefix(mut self, prefix: impl ToString) -> Self {
		self.options.set("trt_engine_cache_prefix", prefix.to_string());
		self
	}

	#[must_use]
	pub fn with_weight_stripped_engine(mut self, enable: bool) -> Self {
		self.options.set("trt_weight_stripped_engine_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_onnx_model_folder_path(mut self, path: impl ToString) -> Self {
		self.options.set("trt_onnx_model_folder_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_engine_decryption(mut self, enable: bool) -> Self {
		self.options.set("trt_engine_decryption_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_engine_decryption_lib_path(mut self, lib_path: impl ToString) -> Self {
		self.options.set("trt_engine_decryption_lib_path", lib_path.to_string());
		self
	}

	#[must_use]
	pub fn with_force_sequential_engine_build(mut self, enable: bool) -> Self {
		self.options.set("trt_force_sequential_engine_build", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_context_memory_sharing(mut self, enable: bool) -> Self {
		self.options.set("trt_context_memory_sharing_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_layer_norm_fp32_fallback(mut self, enable: bool) -> Self {
		self.options.set("trt_layer_norm_fp32_fallback", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_timing_cache(mut self, enable: bool) -> Self {
		self.options.set("trt_timing_cache_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_timing_cache_path(mut self, path: impl ToString) -> Self {
		self.options.set("trt_timing_cache_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_force_timing_cache(mut self, enable: bool) -> Self {
		self.options.set("trt_force_timing_cache", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_detailed_build_log(mut self, enable: bool) -> Self {
		self.options.set("trt_detailed_build_log", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_build_heuristics(mut self, enable: bool) -> Self {
		self.options.set("trt_build_heuristics_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_sparsity(mut self, enable: bool) -> Self {
		self.options.set("trt_sparsity_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_builder_optimization_level(mut self, level: u8) -> Self {
		self.options.set("trt_builder_optimization_level", level.to_string());
		self
	}

	#[must_use]
	pub fn with_auxiliary_streams(mut self, streams: i8) -> Self {
		self.options.set("trt_auxiliary_streams", streams.to_string());
		self
	}

	#[must_use]
	pub fn with_tactic_sources(mut self, sources: impl ToString) -> Self {
		self.options.set("trt_tactic_sources", sources.to_string());
		self
	}

	#[must_use]
	pub fn with_extra_plugin_lib_paths(mut self, paths: impl ToString) -> Self {
		self.options.set("trt_extra_plugin_lib_paths", paths.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_min_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("trt_profile_min_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_max_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("trt_profile_max_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_profile_opt_shapes(mut self, shapes: impl ToString) -> Self {
		self.options.set("trt_profile_opt_shapes", shapes.to_string());
		self
	}

	#[must_use]
	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options.set("trt_cuda_graph_enable", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_dump_ep_context_model(mut self, enable: bool) -> Self {
		self.options.set("trt_dump_ep_context_model", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn with_ep_context_file_path(mut self, path: impl ToString) -> Self {
		self.options.set("trt_ep_context_file_path", path.to_string());
		self
	}

	#[must_use]
	pub fn with_ep_context_embed_mode(mut self, mode: u8) -> Self {
		self.options.set("trt_ep_context_embed_mode", mode.to_string());
		self
	}

	#[must_use]
	pub fn with_engine_hw_compatible(mut self, enable: bool) -> Self {
		self.options.set("trt_engine_hw_compatible", if enable { "1" } else { "0" });
		self
	}
}

impl ExecutionProvider for TensorRTExecutionProvider {
	fn name(&self) -> &'static str {
		"TensorrtExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "x86_64")), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "tensorrt"))]
		{
			use core::ptr;

			use crate::{AsPointer, environment::get_environment, ortsys, util};

			// The TensorRT execution provider specifically is pretty picky about requiring an environment to be initialized by the
			// time we register it. This isn't always the case in `ort`, so if we get to this point, let's make sure we have an
			// environment initialized.
			let _ = get_environment();

			let mut trt_options: *mut ort_sys::OrtTensorRTProviderOptionsV2 = ptr::null_mut();
			ortsys![unsafe CreateTensorRTProviderOptions(&mut trt_options)?];
			let _guard = util::run_on_drop(|| {
				ortsys![unsafe ReleaseTensorRTProviderOptions(trt_options)];
			});

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe UpdateTensorRTProviderOptions(
				trt_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(session_builder.ptr_mut(), trt_options)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
