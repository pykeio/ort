#![allow(non_snake_case)]

use std::{
	ffi::{CStr, CString, OsString},
	fs, ptr
};

use ort_sys::*;
use tract_onnx::prelude::*;

use crate::{
	Environment, Error, convert_datum_type_to_sys, convert_sys_to_datum_type,
	memory::Allocator,
	session::{Session, SessionOptions},
	tensor::TypeInfo
};

unsafe extern "system" fn CreateStatus(code: OrtErrorCode, msg: *const ::std::os::raw::c_char) -> OrtStatusPtr {
	let msg = CString::from_raw(msg.cast_mut());
	Error::new_sys(code, msg.to_string_lossy())
}

unsafe extern "system" fn GetErrorCode(status: *const OrtStatus) -> OrtErrorCode {
	Error::cast_from_sys(status).code
}

unsafe extern "system" fn GetErrorMessage(status: *const OrtStatus) -> *const ::std::os::raw::c_char {
	Error::cast_from_sys(status).message_ptr()
}

unsafe extern "system" fn CreateEnv(_log_severity_level: OrtLoggingLevel, _logid: *const ::std::os::raw::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	*out = Environment::new_sys();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateEnvWithCustomLogger(
	_logging_function: OrtLoggingFunction,
	_logger_param: *mut ::std::os::raw::c_void,
	_log_severity_level: OrtLoggingLevel,
	_logid: *const ::std::os::raw::c_char,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	*out = Environment::new_sys();
	OrtStatusPtr::default()
}

unsafe extern "system" fn EnableTelemetryEvents(_env: *const OrtEnv) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

unsafe extern "system" fn DisableTelemetryEvents(_env: *const OrtEnv) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateSession(
	env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	let env = unsafe { &*env.cast::<Environment>() };
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let len = (0..).take_while(|&i| *model_path.offset(i) != 0).count();
	let path = std::slice::from_raw_parts(model_path, len);
	#[cfg(target_os = "windows")]
	let path = {
		use std::os::windows::ffi::OsStringExt;
		OsString::from_wide(path)
	};
	#[cfg(not(target_os = "windows"))]
	let path = OsString::from_encoded_bytes_unchecked(path.iter().map(|c| *c as u8).collect::<Vec<_>>());

	let buf = match fs::read(path) {
		Ok(buf) => buf,
		Err(e) => return Error::new_sys(OrtErrorCode::ORT_NO_SUCHFILE, format!("Failed to read model file: {e}"))
	};

	match Session::from_buffer(env, options, &buf) {
		Ok(session) => {
			*out = (Box::leak(Box::new(session)) as *mut Session).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to parse model: {e}"))
	}
}

unsafe extern "system" fn CreateSessionFromArray(
	env: *const OrtEnv,
	model_data: *const ::std::os::raw::c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	let env = unsafe { &*env.cast::<Environment>() };
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let buf = std::slice::from_raw_parts(model_data.cast::<u8>(), model_data_length);

	match Session::from_buffer(env, options, buf) {
		Ok(session) => {
			*out = (Box::leak(Box::new(session)) as *mut Session).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to parse model: {e}"))
	}
}

unsafe extern "system" fn Run(
	session: *mut OrtSession,
	_run_options: *const OrtRunOptions,
	input_names: *const *const ::std::os::raw::c_char,
	inputs: *const *const OrtValue,
	input_len: usize,
	output_names: *const *const ::std::os::raw::c_char,
	output_names_len: usize,
	output_ptrs: *mut *mut OrtValue
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };

	let inputs: Vec<(String, Tensor)> = std::slice::from_raw_parts(input_names, input_len)
		.iter()
		.zip(std::slice::from_raw_parts(inputs, input_len))
		.map(|(&name, &input)| {
			let name = unsafe { CStr::from_ptr(name) };
			let input = unsafe { &*input.cast::<Tensor>() };
			(name.to_string_lossy().to_string(), input.clone())
		})
		.collect();

	match session.run(inputs) {
		Ok(outputs) => {
			let output_names: Vec<String> = std::slice::from_raw_parts(output_names, output_names_len)
				.iter()
				.map(|&name| unsafe { CStr::from_ptr(name) }.to_string_lossy().to_string())
				.collect();
			let output_view = std::slice::from_raw_parts_mut(output_ptrs.cast::<*mut Tensor>(), output_names_len);

			for (name, tensor) in outputs {
				if let Some(index) = output_names
					.iter()
					.zip(output_view.iter_mut())
					.find_map(|(o_name, output)| if name == *o_name { Some(output) } else { None })
				{
					*index = Box::leak(Box::new(tensor));
				}
			}

			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to run session: {e}"))
	}
}

unsafe extern "system" fn CreateSessionOptions(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	*options = (Box::leak(Box::new(SessionOptions::default())) as *mut SessionOptions).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	*out_options = (Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SetSessionGraphOptimizationLevel(options: *mut OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	options.perform_optimizations = graph_optimization_level != GraphOptimizationLevel::ORT_DISABLE_ALL;
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	*out = session.inputs.len();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	*out = session.outputs.len();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOverridableInitializerCount(_session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	*out = 0;
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let fact = match session.original_graph.input_fact(index) {
		Ok(fact) => fact,
		Err(e) => return Error::new_sys(OrtErrorCode::ORT_FAIL, e.to_string())
	};
	*type_info = TypeInfo::new_sys(
		fact.datum_type,
		fact.shape
			.to_tvec()
			.into_iter()
			.map(|dim| match dim {
				TDim::Val(size) => size,
				_ => -1
			})
			.collect()
	);
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let fact = match session.original_graph.output_fact(index) {
		Ok(fact) => fact,
		Err(e) => return Error::new_sys(OrtErrorCode::ORT_FAIL, e.to_string())
	};
	*type_info = TypeInfo::new_sys(
		fact.datum_type,
		fact.shape
			.to_tvec()
			.into_iter()
			.map(|dim| match dim {
				TDim::Val(size) => size,
				_ => -1
			})
			.collect()
	);
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let name = match session.inputs.get(index) {
		Some(value) => CString::new(value.name.as_str()).unwrap(),
		None => return Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Invalid input #{}", index + 1))
	};
	*value = name.into_raw();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let name = match session.outputs.get(index) {
		Some(value) => CString::new(value.name.as_str()).unwrap(),
		None => return Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Invalid output #{}", index + 1))
	};
	*value = name.into_raw();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateTensorAsOrtValue(
	_allocator: *mut OrtAllocator,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.copied()
		.map(|c| c as usize)
		.collect();
	let dtype = match convert_sys_to_datum_type(type_) {
		Ok(dtype) => dtype,
		Err(e) => return e.into_sys()
	};
	match Tensor::zero_dt(dtype, &shape) {
		Ok(tensor) => {
			*out = (Box::leak(Box::new(tensor)) as *mut Tensor).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_EP_FAIL, format!("Failed to create tensor: {e}"))
	}
}

unsafe extern "system" fn CreateTensorWithDataAsOrtValue(
	_info: *const OrtMemoryInfo,
	p_data: *mut ::std::os::raw::c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let data_slice = unsafe { std::slice::from_raw_parts(p_data.cast::<u8>(), p_data_len) };
	let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.copied()
		.map(|c| c as usize)
		.collect();
	let dtype = match convert_sys_to_datum_type(type_) {
		Ok(dtype) => dtype,
		Err(e) => return e.into_sys()
	};
	match Tensor::from_raw_dt(dtype, &shape, data_slice) {
		Ok(tensor) => {
			*out = (Box::leak(Box::new(tensor)) as *mut Tensor).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_EP_FAIL, format!("Failed to create tensor: {e}"))
	}
}

unsafe extern "system" fn IsTensor(_value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	*out = 1;
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorMutableData(value: *mut OrtValue, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast::<Tensor>() };
	*out = tensor.as_bytes_mut().as_mut_ptr().cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CastTypeInfoToTensorInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	*out = type_info.cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetOnnxTypeFromTypeInfo(_type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr {
	*out = ONNXType::ONNX_TYPE_TENSOR;
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateTensorTypeAndShapeInfo(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	*out = TypeInfo::new_sys(DatumType::F32, Vec::new()).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SetTensorElementType(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	match convert_sys_to_datum_type(type_) {
		Ok(dtype) => {
			info.dtype = dtype;
			OrtStatusPtr::default()
		}
		Err(e) => e.into_sys()
	}
}

unsafe extern "system" fn SetDimensions(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: usize) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	info.shape = unsafe { std::slice::from_raw_parts(dim_values.cast(), dim_count) }.to_vec();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorElementType(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	*out = convert_datum_type_to_sys(info.dtype);
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetDimensionsCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	*out = info.shape.len();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetDimensions(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	for (i, dim) in info.shape.iter().enumerate().take(dim_values_length) {
		*dim_values.add(i) = *dim as _;
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetSymbolicDimensions(
	_info: *const OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const ::std::os::raw::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	for i in 0..dim_params_length {
		*dim_params.add(i) = c"".as_ptr();
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorShapeElementCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	let mut size = 1usize;
	for dim in &info.shape {
		size *= *dim as usize;
	}
	*out = size;
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTensorTypeAndShape(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	*out = TypeInfo::new_sys(tensor.datum_type(), tensor.shape().iter().map(|c| *c as i64).collect()).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTypeInfo(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	*out = TypeInfo::new_sys(tensor.datum_type(), tensor.shape().iter().map(|c| *c as i64).collect());
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetValueType(_value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	*out = ONNXType::ONNX_TYPE_TENSOR;
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateMemoryInfo(
	name: *const ::std::os::raw::c_char,
	_type_: OrtAllocatorType,
	_id: ::std::os::raw::c_int,
	_mem_type: OrtMemType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	let device_name = unsafe { CStr::from_ptr(name) };
	let device_name = device_name.to_string_lossy();
	if device_name != "Cpu" {
		return Error::new(OrtErrorCode::ORT_ENGINE_ERROR, format!("tract does not support the '{device_name}' device")).into_sys();
	}
	unsafe { *out = ptr::dangling_mut() };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateCpuMemoryInfo(_type_: OrtAllocatorType, _mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
	unsafe { *out = ptr::dangling_mut() };
	OrtStatusPtr::default()
}

unsafe extern "system" fn CompareMemoryInfo(_info1: *const OrtMemoryInfo, _info2: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	*out = 0;
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetName(_ptr: *const OrtMemoryInfo, out: *mut *const ::std::os::raw::c_char) -> OrtStatusPtr {
	*out = b"Cpu\0".as_ptr().cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetId(_ptr: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	*out = 0;
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetMemType(_ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr {
	*out = OrtMemType::OrtMemTypeDefault;
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetType(_ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr {
	*out = OrtAllocatorType::OrtDeviceAllocator;
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorAlloc(ort_allocator: *mut OrtAllocator, size: usize, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr {
	*out = unsafe { &*ort_allocator }.Alloc.unwrap()(ort_allocator, size);
	if unsafe { *out }.is_null() {
		return Error::new_sys(OrtErrorCode::ORT_RUNTIME_EXCEPTION, "Allocation failed");
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorFree(ort_allocator: *mut OrtAllocator, p: *mut ::std::os::raw::c_void) -> OrtStatusPtr {
	unsafe { &*ort_allocator }.Free.unwrap()(ort_allocator, p);
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorGetInfo(ort_allocator: *const OrtAllocator, out: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	*out = unsafe { &*ort_allocator }.Info.unwrap()(ort_allocator);
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetAllocatorWithDefaultOptions(out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	*out = (&crate::memory::DEFAULT_CPU_ALLOCATOR as *const Allocator).cast_mut().cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseEnv(input: *mut OrtEnv) {
	drop(Environment::consume_sys(input));
}

unsafe extern "system" fn ReleaseStatus(input: *mut OrtStatus) {
	drop(Error::consume_sys(input));
}

unsafe extern "system" fn ReleaseSession(input: *mut OrtSession) {
	drop(unsafe { Box::<Session>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseValue(input: *mut OrtValue) {
	drop(unsafe { Box::<Tensor>::from_raw(input.cast()) });
}

unsafe extern "system" fn ReleaseTypeInfo(input: *mut OrtTypeInfo) {
	drop(TypeInfo::consume_sys(input));
}

unsafe extern "system" fn ReleaseTensorTypeAndShapeInfo(input: *mut OrtTensorTypeAndShapeInfo) {
	drop(TypeInfo::consume_sys(input.cast()));
}

unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {
	drop(Box::from_raw(input.cast::<SessionOptions>()));
}

unsafe extern "system" fn CreateAllocator(_session: *const OrtSession, _mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	*out = (Box::leak(Box::new(Allocator::new())) as *mut Allocator).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {
	drop(Box::from_raw(input.cast::<Allocator>()));
}

unsafe extern "system" fn GetTensorMemoryInfo(_value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	*mem_info = ptr::dangling();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetBuildInfoString() -> *const ::std::os::raw::c_char {
	concat!("ORT Build Info: backend=ort-tract, version=", env!("CARGO_PKG_VERSION"), "\0")
		.as_ptr()
		.cast()
}

pub const fn api() -> OrtApi {
	OrtApi {
		CreateStatus,
		GetErrorCode,
		GetErrorMessage,
		CreateEnv,
		CreateEnvWithCustomLogger,
		EnableTelemetryEvents,
		DisableTelemetryEvents,
		CreateSession,
		CreateSessionFromArray,
		Run,
		CreateSessionOptions,
		CloneSessionOptions,
		SetSessionGraphOptimizationLevel,
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
		AllocatorAlloc,
		AllocatorFree,
		AllocatorGetInfo,
		GetAllocatorWithDefaultOptions,
		ReleaseEnv,
		ReleaseStatus,
		ReleaseSession,
		ReleaseValue,
		ReleaseTypeInfo,
		ReleaseTensorTypeAndShapeInfo,
		ReleaseSessionOptions,
		CreateAllocator,
		ReleaseAllocator,
		GetTensorMemoryInfo,
		GetBuildInfoString,
		..ort_sys::stub::api()
	}
}
