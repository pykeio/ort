#![allow(non_snake_case)]

use std::{
	collections::HashMap,
	ffi::{CStr, CString, OsString},
	fs, mem
};

use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
use ort_sys::*;

use crate::{
	Environment, Error, convert_dtype_to_sys, convert_sys_to_dtype,
	memory::{Allocator, MemoryInfo},
	session::{Session, SessionOptions},
	tensor::TypeInfo
};

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
	_env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
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

	match Session::from_buffer(options, &buf) {
		Ok(session) => {
			*out = (Box::leak(Box::new(session)) as *mut Session).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Failed to parse model: {e}"))
	}
}

unsafe extern "system" fn CreateSessionFromArray(
	_env: *const OrtEnv,
	model_data: *const ::std::os::raw::c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let buf = std::slice::from_raw_parts(model_data.cast::<u8>(), model_data_length);

	match Session::from_buffer(options, buf) {
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

	let inputs: HashMap<String, Tensor> = std::slice::from_raw_parts(input_names, input_len)
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
	*options = (Box::leak(Box::new(SessionOptions)) as *mut SessionOptions).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	*out_options = (Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			*out = graph.input.len();
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			*out = graph.output.len();
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn SessionGetOverridableInitializerCount(_session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	*out = 0;
	OrtStatusPtr::default()
}

unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			let type_proto = graph.input[index].r#type.as_ref().unwrap().value.as_ref().unwrap();
			match type_proto {
				candle_onnx::onnx::type_proto::Value::TensorType(tensor_proto) => {
					let dtype = mem::transmute::<i32, candle_onnx::onnx::tensor_proto::DataType>(tensor_proto.elem_type);
					let dtype = match candle_onnx::dtype(dtype) {
						Some(dtype) => dtype,
						None => return Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Unsupported data type {dtype:?} for input #{}", index + 1))
					};

					let mut shape_out = vec![];
					if let Some(shape) = tensor_proto.shape.as_ref() {
						for dim in &shape.dim {
							shape_out.push(match &dim.value {
								Some(v) => match v {
									candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(v) => *v,
									candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimParam(_) => -1i64
								},
								None => -1i64
							});
						}
					}

					*type_info = TypeInfo::new_sys(dtype, shape_out);
					OrtStatusPtr::default()
				}
				_ => Error::new_sys(OrtErrorCode::ORT_FAIL, "Invalid type; only tensors are supported")
			}
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			let type_proto = graph.output[index].r#type.as_ref().unwrap().value.as_ref().unwrap();
			match type_proto {
				candle_onnx::onnx::type_proto::Value::TensorType(tensor_proto) => {
					let dtype = mem::transmute::<i32, candle_onnx::onnx::tensor_proto::DataType>(tensor_proto.elem_type);
					let dtype = match candle_onnx::dtype(dtype) {
						Some(dtype) => dtype,
						None => return Error::new_sys(OrtErrorCode::ORT_FAIL, format!("Unsupported data type {dtype:?} for output #{}", index + 1))
					};

					let mut shape_out = vec![];
					if let Some(shape) = tensor_proto.shape.as_ref() {
						for dim in &shape.dim {
							shape_out.push(match &dim.value {
								Some(v) => match v {
									candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimValue(v) => *v,
									candle_onnx::onnx::tensor_shape_proto::dimension::Value::DimParam(_) => -1i64
								},
								None => -1i64
							});
						}
					}

					*type_info = TypeInfo::new_sys(dtype, shape_out);
					OrtStatusPtr::default()
				}
				_ => Error::new_sys(OrtErrorCode::ORT_FAIL, "Invalid type; only tensors are supported")
			}
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			let name = CString::new(&*graph.input[index].name).unwrap();
			*value = name.into_raw();
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	_allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	match session.model.graph.as_ref() {
		Some(graph) => {
			let name = CString::new(&*graph.output[index].name).unwrap();
			*value = name.into_raw();
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_NO_MODEL, "Graph is missing")
	}
}

unsafe extern "system" fn CreateTensorAsOrtValue(
	allocator: *mut OrtAllocator,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let allocator = unsafe { &*allocator.cast::<Allocator>() };
	let mem_info = allocator.memory_info;
	let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.copied()
		.map(|c| c as usize)
		.collect();
	let dtype = match convert_sys_to_dtype(type_) {
		Ok(dtype) => dtype,
		Err(e) => return e.into_sys()
	};
	match Tensor::zeros(shape, dtype, mem_info.device()) {
		Ok(tensor) => {
			*out = (Box::leak(Box::new(tensor)) as *mut Tensor).cast();
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_EP_FAIL, format!("Failed to create tensor: {e}"))
	}
}

unsafe extern "system" fn CreateTensorWithDataAsOrtValue(
	info: *const OrtMemoryInfo,
	p_data: *mut ::std::os::raw::c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let mem_info = unsafe { &*info.cast::<MemoryInfo>() };
	let data_slice = unsafe { std::slice::from_raw_parts(p_data.cast::<u8>(), p_data_len) };
	let shape: Vec<usize> = unsafe { std::slice::from_raw_parts(shape, shape_len) }
		.iter()
		.copied()
		.map(|c| c as usize)
		.collect();
	let dtype = match convert_sys_to_dtype(type_) {
		Ok(dtype) => dtype,
		Err(e) => return e.into_sys()
	};
	match Tensor::from_raw_buffer(data_slice, dtype, &shape, mem_info.device()) {
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
	let tensor = unsafe { &*value.cast::<Tensor>() };
	let (storage, _layout) = tensor.storage_and_layout();
	match &*storage {
		Storage::Cpu(storage) => {
			*out = match storage {
				CpuStorage::U8(v) => v.as_ptr() as *mut _,
				CpuStorage::U32(v) => v.as_ptr() as *mut _,
				CpuStorage::I64(v) => v.as_ptr() as *mut _,
				CpuStorage::F64(v) => v.as_ptr() as *mut _,
				CpuStorage::F32(v) => v.as_ptr() as *mut _,
				CpuStorage::F16(v) => v.as_ptr() as *mut _,
				CpuStorage::BF16(v) => v.as_ptr() as *mut _,
				_ => return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented dtype")
			};
			OrtStatusPtr::default()
		}
		_ => Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
	}
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
	*out = TypeInfo::new_sys(DType::F32, Vec::new()).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SetTensorElementType(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	match convert_sys_to_dtype(type_) {
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
	match convert_dtype_to_sys(info.dtype) {
		Ok(ty) => {
			*out = ty;
			OrtStatusPtr::default()
		}
		Err(e) => e.into_sys()
	}
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
	*out = TypeInfo::new_sys(tensor.dtype(), tensor.shape().dims().iter().map(|c| *c as i64).collect()).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetTypeInfo(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	*out = TypeInfo::new_sys(tensor.dtype(), tensor.shape().dims().iter().map(|c| *c as i64).collect());
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetValueType(_value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	*out = ONNXType::ONNX_TYPE_TENSOR;
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateMemoryInfo(
	name: *const ::std::os::raw::c_char,
	_type: OrtAllocatorType,
	id: ::std::os::raw::c_int,
	mem_type: OrtMemType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	let device_name = unsafe { CStr::from_ptr(name) };
	match MemoryInfo::new(device_name.to_string_lossy(), id as _, mem_type) {
		Ok(inf) => {
			unsafe { *out = (Box::leak(Box::new(inf)) as *mut MemoryInfo).cast() };
			OrtStatusPtr::default()
		}
		Err(e) => e.into_sys()
	}
}

unsafe extern "system" fn CreateCpuMemoryInfo(_type: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
	match MemoryInfo::new("Cpu", 0, mem_type) {
		Ok(inf) => {
			unsafe { *out = (Box::leak(Box::new(inf)) as *mut MemoryInfo).cast() };
			OrtStatusPtr::default()
		}
		Err(e) => e.into_sys()
	}
}

unsafe extern "system" fn CompareMemoryInfo(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	let info1 = unsafe { &*info1.cast::<MemoryInfo>() };
	let info2 = unsafe { &*info2.cast::<MemoryInfo>() };
	*out = if info1 == info2 { 0 } else { -1 };
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetName(ptr: *const OrtMemoryInfo, out: *mut *const ::std::os::raw::c_char) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	*out = info.device_name_sys().as_ptr().cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetId(ptr: *const OrtMemoryInfo, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	*out = info.device_id() as _;
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetMemType(ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	*out = info.memory_type() as _;
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

unsafe extern "system" fn ReleaseMemoryInfo(input: *mut OrtMemoryInfo) {
	drop(unsafe { Box::<MemoryInfo>::from_raw(input.cast()) });
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

unsafe extern "system" fn CreateAllocator(_session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	let mem_info = unsafe { &*mem_info.cast::<MemoryInfo>() };
	*out = (Box::leak(Box::new(Allocator::new(mem_info))) as *mut Allocator).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {
	drop(Box::from_raw(input.cast::<Allocator>()));
}

unsafe extern "system" fn GetTensorMemoryInfo(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	// `MemoryInfo` is #[repr(transparent)], so &MemoryInfo is &Device.
	*mem_info = (tensor.device() as *const Device).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn MemoryInfoGetDeviceType(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType) {
	let memory_info = unsafe { &*ptr.cast::<MemoryInfo>() };
	*out = memory_info.device_type();
}

unsafe extern "system" fn GetBuildInfoString() -> *const ::std::os::raw::c_char {
	concat!("ORT Build Info: backend=ort-candle, version=", env!("CARGO_PKG_VERSION"), "\0")
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
		AllocatorAlloc,
		AllocatorFree,
		AllocatorGetInfo,
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
		..ort_sys::stub::api()
	}
}
