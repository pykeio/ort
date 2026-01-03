#![allow(non_snake_case)]

use alloc::{
	boxed::Box,
	ffi::CString,
	format,
	string::{String, ToString},
	vec::Vec
};
use core::{
	ffi::{self, CStr},
	future::Future,
	pin::Pin
};
use std::collections::HashMap;

use ort_sys::{stub::Error, *};

use crate::{
	binding,
	env::{Environment, TelemetryEvent},
	memory::{Allocator, MemoryInfo},
	session::{RunOptions, Session, SessionOptions},
	tensor::{SyncDirection, Tensor, TensorData, TypeInfo, create_buffer, onnx_to_dtype},
	util::value_to_string
};

unsafe extern "system" fn CreateEnv(_log_severity_level: OrtLoggingLevel, _logid: *const ffi::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateEnvWithCustomLogger(
	_logging_function: OrtLoggingFunction,
	_logger_param: *mut ffi::c_void,
	_log_severity_level: OrtLoggingLevel,
	_logid: *const ffi::c_char,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn EnableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	let env = unsafe { Environment::cast_from_sys_mut(env.cast_mut()) };
	env.with_telemetry = true;
	OrtStatusPtr::default()
}

unsafe extern "system" fn DisableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	let env = unsafe { Environment::cast_from_sys_mut(env.cast_mut()) };
	env.with_telemetry = false;
	OrtStatusPtr::default()
}

unsafe fn CreateSession(
	env: *const OrtEnv,
	model_path: &str,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let fut = Box::pin(async move {
		match Session::from_url(model_path, options).await {
			Ok(session) => {
				let ptr = (Box::leak(Box::new(session))) as *mut Session;
				unsafe { out.write(ptr.cast()) };

				{
					let env = unsafe { Environment::cast_from_sys(env) };
					env.send_telemetry_event(TelemetryEvent::SessionInit);
				}

				OrtStatusPtr::default()
			}
			Err(e) => e.into_sys()
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { core::mem::transmute(fut) }
}

unsafe fn CreateSessionFromArray(
	env: *const OrtEnv,
	model_data: &[u8],
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let fut = Box::pin(async move {
		match Session::from_bytes(model_data, options).await {
			Ok(session) => {
				let ptr = (Box::leak(Box::new(session))) as *mut Session;
				unsafe { out.write(ptr.cast()) };

				{
					let env = unsafe { Environment::cast_from_sys(env) };
					env.send_telemetry_event(TelemetryEvent::SessionInit);
				}

				OrtStatusPtr::default()
			}
			Err(e) => e.into_sys()
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { core::mem::transmute(fut) }
}

unsafe extern "system" fn Run(
	_session: *mut OrtSession,
	_run_options: *const OrtRunOptions,
	_input_names: *const *const ::core::ffi::c_char,
	_inputs: *const *const OrtValue,
	_input_len: usize,
	_output_names: *const *const ::core::ffi::c_char,
	_output_names_len: usize,
	_output_ptrs: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_FAIL, "Synchronous `Session::run` is not supported in ort-web; use `run_async()`.")
}

unsafe fn RunAsync(
	session: *mut OrtSession,
	_run_options: *const OrtRunOptions,
	input_names: &[&str],
	inputs: &[*const OrtValue],
	output_names: &[&str],
	output_ptrs: &mut [*mut OrtValue]
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let session = unsafe { &*session.cast::<Session>() };

	let fut = Box::pin(async move {
		let inputs = input_names
			.iter()
			.zip(inputs)
			.map(|(&name, &input)| (name, unsafe { &*input.cast::<Tensor>() }))
			.collect::<Vec<(&str, &Tensor)>>();

		match session.js.run(inputs.into_iter()).await {
			Ok(outputs) => {
				let output_names: Vec<String> = output_names.iter().map(|&name| name.to_string()).collect();
				let output_view = unsafe { core::slice::from_raw_parts_mut(output_ptrs.as_mut_ptr().cast::<*mut Tensor>(), output_ptrs.len()) };

				for (name, mut tensor) in outputs {
					if let Some(index) = output_names
						.iter()
						.zip(output_view.iter_mut())
						.find_map(|(o_name, output)| if name == *o_name { Some(output) } else { None })
					{
						if !session.disable_sync {
							if let Err(e) = tensor.sync(SyncDirection::Rust).await {
								return Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to synchronize output '{name}': {e}"));
							}
						}

						*index = Box::leak(Box::new(tensor));
					}
				}

				OrtStatusPtr::default()
			}
			Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to run session: {}", value_to_string(&e)))
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { core::mem::transmute(fut) }
}

unsafe extern "system" fn CreateSessionOptions(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { options.write((Box::leak(Box::new(SessionOptions::new())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider(
	options: *mut OrtSessionOptions,
	provider_name: *const ::core::ffi::c_char,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let execution_providers = options.js.execution_providers.get_or_insert_default();

	let Ok(options) = unsafe { core::slice::from_raw_parts(provider_options_keys, num_keys) }
		.iter()
		.zip(unsafe { core::slice::from_raw_parts(provider_options_values, num_keys) }.iter())
		.map(|(k, v)| Ok((unsafe { CStr::from_ptr(*k) }.to_str()?, unsafe { CStr::from_ptr(*v) }.to_str()?)))
		.collect::<Result<HashMap<&str, &str>, core::str::Utf8Error>>()
	else {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "EP options contains invalid UTF-8");
	};

	let provider_name = unsafe { CStr::from_ptr(provider_name) };
	match provider_name.to_string_lossy().as_ref() {
		"WASM" => {
			execution_providers.push(binding::ExecutionProvider::WASM);
		}
		"WebGL" => {
			execution_providers.push(binding::ExecutionProvider::WebGL);
		}
		"WebGPU" => {
			execution_providers.push(binding::ExecutionProvider::WebGPU {
				preferred_layout: match options.get("ep.webgpuexecutionprovider.preferredLayout") {
					Some(&"NHWC") => Some(binding::WebGPUPreferredLayout::NHWC),
					Some(&"NCHW") => Some(binding::WebGPUPreferredLayout::NCHW),
					_ => None
				}
			});
		}
		"WebNN" => {
			execution_providers.push(binding::ExecutionProvider::WebNN {
				power_preference: match options.get("powerPreference") {
					Some(&"default") => Some(binding::WebNNPowerPreference::Default),
					Some(&"high-performance") => Some(binding::WebNNPowerPreference::HighPerformance),
					Some(&"low-power") => Some(binding::WebNNPowerPreference::LowPower),
					_ => None
				},
				device_type: match options.get("deviceType") {
					Some(&"cpu") => Some(binding::WebNNDeviceType::CPU),
					Some(&"npu") => Some(binding::WebNNDeviceType::NPU),
					Some(&"gpu") => Some(binding::WebNNDeviceType::GPU),
					_ => None
				},
				num_threads: options.get("numThreads").and_then(|c| c.parse().ok())
			});
		}
		x => return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("Provider '{x}' not supported"))
	}

	OrtStatusPtr::default()
}

unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	unsafe { out_options.write((Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.js.input_len()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.js.output_len()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOverridableInitializerCount(_session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	unsafe { out.write(0) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let metadata = session.js.input_metadata().remove(index);
	if !metadata.is_tensor {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "non-tensor types are not currently supported");
	}

	unsafe { type_info.write(TypeInfo::new_sys_from_value_metadata(&metadata)) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let metadata = session.js.output_metadata().remove(index);
	if !metadata.is_tensor {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "non-tensor types are not currently supported");
	}

	unsafe { type_info.write(TypeInfo::new_sys_from_value_metadata(&metadata)) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ffi::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let name = CString::new(&*session.js.input_names().remove(index)).unwrap();
	unsafe { value.write(name.into_raw()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ffi::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let name = CString::new(&*session.js.output_names().remove(index)).unwrap();
	unsafe { value.write(name.into_raw()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateRunOptions(out: *mut *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { out.write((Box::leak(Box::new(RunOptions::new())) as *mut RunOptions).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateTensorAsOrtValue(
	_allocator: *mut OrtAllocator,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let shape = unsafe { core::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.map(|c| *c as i32)
		.collect::<Vec<_>>();
	let Some(dtype) = onnx_to_dtype(type_) else {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "unsupported dtype");
	};

	match binding::Tensor::new_from_buffer(dtype, create_buffer(dtype, &shape), &shape) {
		Ok(tensor) => {
			unsafe { out.write((Box::leak(Box::new(Tensor::from_tensor(tensor))) as *mut Tensor).cast()) };
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to create tensor: {}", value_to_string(&e)))
	}
}

unsafe extern "system" fn CreateTensorWithDataAsOrtValue(
	_info: *const OrtMemoryInfo,
	p_data: *mut ffi::c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let shape = unsafe { core::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.map(|c| *c as i32)
		.collect::<Vec<_>>();
	let Some(dtype) = onnx_to_dtype(type_) else {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "unsupported dtype");
	};

	match unsafe { Tensor::from_ptr(dtype, p_data, p_data_len, &shape) } {
		Ok(tensor) => {
			unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to create tensor: {}", value_to_string(&e)))
	}
}

unsafe extern "system" fn IsTensor(_value: *const OrtValue, out: *mut ffi::c_int) -> OrtStatusPtr {
	unsafe { out.write(1) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorMutableData(value: *mut OrtValue, out: *mut *mut ffi::c_void) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast::<Tensor>() };
	match &mut tensor.data {
		TensorData::RustView { ptr, .. } => {
			unsafe { out.write(*ptr) };
			OrtStatusPtr::default()
		}
		TensorData::External { buffer } => {
			if let Some(buffer) = buffer {
				unsafe { out.write(buffer.as_mut_ptr().cast()) };
				OrtStatusPtr::default()
			} else {
				Error::new_sys(OrtErrorCode::ORT_FAIL, "External data is not synchronized; you should call `TensorExt::sync`.")
			}
		}
	}
}

unsafe extern "system" fn CastTypeInfoToTensorInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	unsafe { out.write(type_info.cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetOnnxTypeFromTypeInfo(_type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr {
	unsafe { out.write(ONNXType::ONNX_TYPE_TENSOR) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateTensorTypeAndShapeInfo(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	unsafe { out.write(TypeInfo::new_sys(binding::DataType::Float32, Vec::new()).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn SetTensorElementType(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	match onnx_to_dtype(type_) {
		Some(_) => {
			info.dtype = type_;
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_FAIL, "Unsupported tensor data type")
	}
}

unsafe extern "system" fn SetDimensions(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: usize) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	info.shape = unsafe { core::slice::from_raw_parts(dim_values.cast(), dim_count) }.to_vec();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorElementType(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	unsafe { out.write(info.dtype) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetDimensionsCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	unsafe { out.write(info.shape.len()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetDimensions(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	for (i, dim) in info.shape.iter().enumerate().take(dim_values_length) {
		unsafe { dim_values.add(i).write(*dim as _) };
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetSymbolicDimensions(
	_info: *const OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const ffi::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	for i in 0..dim_params_length {
		unsafe { dim_params.add(i).write(c"".as_ptr()) };
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorShapeElementCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	let mut size = 1usize;
	for dim in &info.shape {
		size *= *dim as usize;
	}
	unsafe { out.write(size) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorTypeAndShape(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(TypeInfo::new_sys_from_tensor(tensor).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTypeInfo(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(TypeInfo::new_sys_from_tensor(tensor)) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetValueType(_value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	unsafe { out.write(ONNXType::ONNX_TYPE_TENSOR) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateMemoryInfo(
	name: *const ffi::c_char,
	_type: OrtAllocatorType,
	_id: ffi::c_int,
	_mem_type: OrtMemType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	let device_name = unsafe { CStr::from_ptr(name) };
	match MemoryInfo::from_location(&*device_name.to_string_lossy()) {
		Some(inf) => {
			unsafe { *out = (Box::leak(Box::new(inf)) as *mut MemoryInfo).cast() };
			OrtStatusPtr::default()
		}
		None => Error::new_sys(
			OrtErrorCode::ORT_FAIL,
			"Unsupported MemoryInfo type - only CPU tensors can be created this way. Tensors must be created from existing non-CPU buffers using `ort_web::TensorExt::from_*`."
		)
	}
}

unsafe extern "system" fn CreateCpuMemoryInfo(_type: OrtAllocatorType, _mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
	unsafe { *out = (Box::leak(Box::new(MemoryInfo { location: binding::DataLocation::Cpu })) as *mut MemoryInfo).cast() };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CompareMemoryInfo(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut ffi::c_int) -> OrtStatusPtr {
	let info1 = unsafe { &*info1.cast::<MemoryInfo>() };
	let info2 = unsafe { &*info2.cast::<MemoryInfo>() };
	unsafe { out.write(if info1 == info2 { 0 } else { -1 }) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetName(ptr: *const OrtMemoryInfo, out: *mut *const ffi::c_char) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.location_exposed().unwrap_or(c"").as_ptr().cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetId(_ptr: *const OrtMemoryInfo, out: *mut ffi::c_int) -> OrtStatusPtr {
	unsafe { out.write(0) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetMemType(_ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr {
	unsafe { out.write(OrtMemType::OrtMemTypeDefault) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetType(_ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr {
	unsafe { out.write(OrtAllocatorType::OrtDeviceAllocator) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetAllocatorWithDefaultOptions(out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	unsafe { out.write((&crate::memory::DEFAULT_CPU_ALLOCATOR as *const Allocator).cast_mut().cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseEnv(input: *mut OrtEnv) {
	drop(unsafe { Environment::consume_sys(input) });
}

unsafe extern "system" fn ReleaseStatus(input: *mut OrtStatus) {
	drop(unsafe { Error::consume_sys(input) });
}

unsafe extern "system" fn ReleaseMemoryInfo(input: *mut OrtMemoryInfo) {
	drop(unsafe { Box::<MemoryInfo>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseSession(input: *mut OrtSession) {
	drop(unsafe { Box::<Session>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseValue(input: *mut OrtValue) {
	drop(unsafe { Box::<Tensor>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseRunOptions(input: *mut OrtRunOptions) {
	drop(unsafe { Box::<RunOptions>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseTypeInfo(input: *mut OrtTypeInfo) {
	drop(unsafe { TypeInfo::consume_sys(input) });
}

unsafe extern "system" fn ReleaseTensorTypeAndShapeInfo(input: *mut OrtTensorTypeAndShapeInfo) {
	drop(unsafe { TypeInfo::consume_sys(input.cast()) });
}

unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {
	drop(unsafe { Box::from_raw(input.cast::<SessionOptions>()) });
}

unsafe extern "system" fn CreateAllocator(_session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	let mem_info = unsafe { &*mem_info.cast::<MemoryInfo>() };
	if mem_info.location != binding::DataLocation::Cpu {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "Only CPU allocators are supported.");
	}

	unsafe { out.write((Box::leak(Box::new(Allocator::new())) as *mut Allocator).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {
	drop(unsafe { Box::from_raw(input.cast::<Allocator>()) });
}

unsafe extern "system" fn GetTensorMemoryInfo(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { mem_info.write((&tensor.memory_info as *const MemoryInfo).cast()) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetDeviceType(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType) {
	let memory_info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe {
		out.write(match memory_info.location {
			binding::DataLocation::Cpu | binding::DataLocation::CpuPinned => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
			_ => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU
		})
	};
}

unsafe extern "system" fn GetBuildInfoString() -> *const ffi::c_char {
	concat!("ORT Build Info: backend=ort-web, version=", env!("CARGO_PKG_VERSION"), ", with <3\0")
		.as_ptr()
		.cast()
}

pub const fn api() -> OrtApi {
	OrtApi {
		CreateEnv,
		CreateEnvWithCustomLogger,
		EnableTelemetryEvents,
		DisableTelemetryEvents,
		CreateSession,
		CreateSessionFromArray,
		Run,
		RunAsync,
		CreateSessionOptions,
		CloneSessionOptions,
		SessionGetInputCount,
		SessionGetOutputCount,
		SessionGetOverridableInitializerCount,
		SessionGetInputTypeInfo,
		SessionGetOutputTypeInfo,
		SessionGetInputName,
		SessionGetOutputName,
		CreateTensorAsOrtValue,
		CreateTensorWithDataAsOrtValue,
		IsTensor,
		GetTensorMutableData,
		CastTypeInfoToTensorInfo,
		GetOnnxTypeFromTypeInfo,
		CreateTensorTypeAndShapeInfo,
		SetTensorElementType,
		SetDimensions,
		GetTensorElementType,
		GetDimensionsCount,
		GetDimensions,
		GetSymbolicDimensions,
		GetTensorShapeElementCount,
		GetTensorTypeAndShape,
		GetTypeInfo,
		GetValueType,
		CreateMemoryInfo,
		CreateCpuMemoryInfo,
		CompareMemoryInfo,
		MemoryInfoGetName,
		MemoryInfoGetId,
		MemoryInfoGetMemType,
		MemoryInfoGetType,
		GetAllocatorWithDefaultOptions,
		ReleaseEnv,
		ReleaseStatus,
		ReleaseMemoryInfo,
		ReleaseSession,
		ReleaseValue,
		ReleaseTypeInfo,
		ReleaseTensorTypeAndShapeInfo,
		ReleaseSessionOptions,
		CreateAllocator,
		ReleaseAllocator,
		GetTensorMemoryInfo,
		MemoryInfoGetDeviceType,
		GetBuildInfoString,
		CreateRunOptions,
		ReleaseRunOptions,
		SessionOptionsAppendExecutionProvider,
		..ort_sys::stub::api()
	}
}
