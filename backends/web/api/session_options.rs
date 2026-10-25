use core::ffi::CStr;

use ort_sys::{stub::Error, *};

use crate::{
	binding,
	session::SessionOptions,
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateSessionOptions(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { options.write((Box::leak(Box::new(SessionOptions::new())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionOptionsAppendExecutionProvider(
	options: *mut OrtSessionOptions,
	provider_name: *const c_char,
	provider_options_keys: *const *const c_char,
	provider_options_values: *const *const c_char,
	num_keys: usize
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let execution_providers = options.js.execution_providers.get_or_insert_default();

	let mut provider_options = Vec::with_capacity(num_keys);
	for i in 0..num_keys {
		let (Ok(key), Ok(value)) = (unsafe { str_from_ptr(*provider_options_keys.add(i)) }, unsafe { str_from_ptr(*provider_options_values.add(i)) }) else {
			return Error::new_sys(OrtErrorCode::ORT_FAIL, "EP options contain invalid UTF-8");
		};
		provider_options.push((key, value));
	}
	let get = |key: &str| provider_options.iter().find(|(k, _)| *k == key).map(|(_, v)| *v);

	let provider_name = unsafe { CStr::from_ptr(provider_name) };
	match provider_name.to_string_lossy().as_ref() {
		"CPUExecutionProvider" => {}
		"WASM" | "WASMExecutionProvider" => {
			execution_providers.push(binding::ExecutionProvider::WASM);
		}
		"WebGL" | "WebGLExecutionProvider" => {
			execution_providers.push(binding::ExecutionProvider::WebGL);
		}
		"WebGPU" | "WebGpuExecutionProvider" => {
			execution_providers.push(binding::ExecutionProvider::WebGPU {
				preferred_layout: match get("preferredLayout") {
					Some("NHWC") => Some(binding::WebGPUPreferredLayout::NHWC),
					Some("NCHW") => Some(binding::WebGPUPreferredLayout::NCHW),
					_ => None
				}
			});
		}
		"WebNN" | "WebNNExecutionProvider" => {
			execution_providers.push(binding::ExecutionProvider::WebNN {
				power_preference: match get("powerPreference") {
					Some("default") => Some(binding::WebNNPowerPreference::Default),
					Some("high-performance") => Some(binding::WebNNPowerPreference::HighPerformance),
					Some("low-power") => Some(binding::WebNNPowerPreference::LowPower),
					_ => None
				},
				device_type: match get("deviceType") {
					Some("cpu") => Some(binding::WebNNDeviceType::CPU),
					Some("npu") => Some(binding::WebNNDeviceType::NPU),
					Some("gpu") => Some(binding::WebNNDeviceType::GPU),
					_ => None
				},
				num_threads: get("numThreads").and_then(|c| c.parse().ok())
			});
		}
		x => return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("EP '{x}' not supported"))
	}

	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	unsafe { out_options.write((Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSessionGraphOptimizationLevel(
	options: *mut OrtSessionOptions,
	graph_optimization_level: GraphOptimizationLevel
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.graph_optimization_level = Some(match graph_optimization_level {
		GraphOptimizationLevel::ORT_DISABLE_ALL => binding::GraphOptimizationLevel::Disabled,
		GraphOptimizationLevel::ORT_ENABLE_BASIC => binding::GraphOptimizationLevel::Basic,
		GraphOptimizationLevel::ORT_ENABLE_EXTENDED => binding::GraphOptimizationLevel::Extended,
		GraphOptimizationLevel::ORT_ENABLE_LAYOUT => binding::GraphOptimizationLevel::Layout,
		GraphOptimizationLevel::ORT_ENABLE_ALL => binding::GraphOptimizationLevel::All
	});
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetIntraOpNumThreads(options: *mut OrtSessionOptions, intra_op_num_threads: c_int) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.intra_op_num_threads = u32::try_from(intra_op_num_threads).ok().filter(|n| *n > 0);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetInterOpNumThreads(options: *mut OrtSessionOptions, inter_op_num_threads: c_int) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.inter_op_num_threads = u32::try_from(inter_op_num_threads).ok().filter(|n| *n > 0);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSessionExecutionMode(options: *mut OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.execution_mode = Some(match execution_mode {
		ExecutionMode::ORT_SEQUENTIAL => binding::ExecutionMode::Sequential,
		ExecutionMode::ORT_PARALLEL => binding::ExecutionMode::Parallel
	});
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn EnableMemPattern(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_mem_pattern = Some(true);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn DisableMemPattern(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_mem_pattern = Some(false);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn EnableCpuMemArena(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_cpu_mem_arena = Some(true);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn DisableCpuMemArena(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_cpu_mem_arena = Some(false);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn EnableProfiling(options: *mut OrtSessionOptions, _profile_file_prefix: *const os_char) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_profiling = Some(true);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn DisableProfiling(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<SessionOptions>() }.js.enable_profiling = Some(false);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSessionLogId(options: *mut OrtSessionOptions, logid: *const c_char) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.log_id = Some(trys!(unsafe { str_from_ptr(logid) }).to_string());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSessionLogSeverityLevel(options: *mut OrtSessionOptions, session_log_severity_level: c_int) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.log_severity_level = u8::try_from(session_log_severity_level).ok();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSessionLogVerbosityLevel(options: *mut OrtSessionOptions, session_log_verbosity_level: c_int) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.js.log_verbosity_level = u16::try_from(session_log_verbosity_level).ok();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddFreeDimensionOverrideByName(options: *mut OrtSessionOptions, dim_name: *const c_char, dim_value: i64) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let name = trys!(unsafe { str_from_ptr(dim_name) });
	let Ok(value) = i32::try_from(dim_value) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "dimension override too large");
	};
	options.js.free_dimension_override.get_or_insert_default().insert(name.to_string(), value);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddSessionConfigEntry(
	options: *mut OrtSessionOptions,
	config_key: *const c_char,
	config_value: *const c_char
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let key = trys!(unsafe { str_from_ptr(config_key) });
	let value = trys!(unsafe { str_from_ptr(config_value) });
	options.js.extra.get_or_insert_default().insert(key.to_string(), value.to_string());
	options.config.insert(key.to_string(), unsafe { CStr::from_ptr(config_value) }.to_owned());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {
	drop(unsafe { Box::from_raw(input.cast::<SessionOptions>()) });
}
