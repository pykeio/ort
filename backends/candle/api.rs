#![allow(non_snake_case, unused)]

use std::{
	collections::HashMap,
	ffi::{CStr, CString, OsStr, OsString},
	fs, mem, ptr
};

use candle_core::{CpuStorage, DType, Device, Shape, Storage, Tensor};
use ort_sys::{
	ExecutionMode, GraphOptimizationLevel, ONNXTensorElementDataType, ONNXType, OrtAllocator, OrtAllocatorType, OrtApi, OrtArenaCfg, OrtCANNProviderOptions,
	OrtCUDAProviderOptions, OrtCUDAProviderOptionsV2, OrtCustomCreateThreadFn, OrtCustomJoinThreadFn, OrtCustomOp, OrtCustomOpDomain, OrtDnnlProviderOptions,
	OrtEnv, OrtErrorCode, OrtIoBinding, OrtKernelContext, OrtKernelInfo, OrtLanguageProjection, OrtLogger, OrtLoggingFunction, OrtLoggingLevel, OrtLoraAdapter,
	OrtMIGraphXProviderOptions, OrtMapTypeInfo, OrtMemType, OrtMemoryInfo, OrtMemoryInfoDeviceType, OrtModelMetadata, OrtOp, OrtOpAttr, OrtOpAttrType,
	OrtOpenVINOProviderOptions, OrtOptionalTypeInfo, OrtPrepackedWeightsContainer, OrtROCMProviderOptions, OrtRunOptions, OrtSequenceTypeInfo, OrtSession,
	OrtSessionOptions, OrtShapeInferContext, OrtSparseFormat, OrtSparseIndicesFormat, OrtStatus, OrtStatusPtr, OrtTensorRTProviderOptions,
	OrtTensorRTProviderOptionsV2, OrtTensorTypeAndShapeInfo, OrtThreadingOptions, OrtTrainingApi, OrtTypeInfo, OrtValue, RunAsyncCallbackFn, ortchar
};

use crate::{
	Environment, convert_dtype_to_sys, convert_sys_to_dtype,
	error::Error,
	memory::{Allocator, MemoryInfo},
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

unsafe extern "system" fn CreateEnv(log_severity_level: OrtLoggingLevel, logid: *const ::std::os::raw::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	*out = Environment::new_sys();
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateEnvWithCustomLogger(
	logging_function: OrtLoggingFunction,
	logger_param: *mut ::std::os::raw::c_void,
	log_severity_level: OrtLoggingLevel,
	logid: *const ::std::os::raw::c_char,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	*out = Environment::new_sys();
	OrtStatusPtr::default()
}

unsafe extern "system" fn EnableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

unsafe extern "system" fn DisableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateSession(
	env: *const OrtEnv,
	model_path: *const ortchar,
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
	env: *const OrtEnv,
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
	run_options: *const OrtRunOptions,
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

unsafe extern "system" fn SetOptimizedModelFilePath(options: *mut OrtSessionOptions, optimized_model_filepath: *const ortchar) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	*out_options = (Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn SetSessionExecutionMode(options: *mut OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn EnableProfiling(options: *mut OrtSessionOptions, profile_file_prefix: *const ortchar) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn DisableProfiling(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn EnableMemPattern(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn DisableMemPattern(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn EnableCpuMemArena(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn DisableCpuMemArena(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionLogId(options: *mut OrtSessionOptions, logid: *const ::std::os::raw::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionLogVerbosityLevel(options: *mut OrtSessionOptions, session_log_verbosity_level: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionLogSeverityLevel(options: *mut OrtSessionOptions, session_log_severity_level: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionGraphOptimizationLevel(options: *mut OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetIntraOpNumThreads(options: *mut OrtSessionOptions, intra_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetInterOpNumThreads(options: *mut OrtSessionOptions, inter_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateCustomOpDomain(domain: *const ::std::os::raw::c_char, out: *mut *mut OrtCustomOpDomain) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CustomOpDomain_Add(custom_op_domain: *mut OrtCustomOpDomain, op: *const OrtCustomOp) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddCustomOpDomain(options: *mut OrtSessionOptions, custom_op_domain: *mut OrtCustomOpDomain) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterCustomOpsLibrary(
	options: *mut OrtSessionOptions,
	library_path: *const ::std::os::raw::c_char,
	library_handle: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
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

unsafe extern "system" fn SessionGetOverridableInitializerCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
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

unsafe extern "system" fn SessionGetOverridableInitializerTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
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
	allocator: *mut OrtAllocator,
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

unsafe extern "system" fn SessionGetOverridableInitializerName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateRunOptions(out: *mut *mut OrtRunOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunLogVerbosityLevel(options: *mut OrtRunOptions, log_verbosity_level: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunLogSeverityLevel(options: *mut OrtRunOptions, log_severity_level: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunTag(options: *mut OrtRunOptions, run_tag: *const ::std::os::raw::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunLogVerbosityLevel(options: *const OrtRunOptions, log_verbosity_level: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunLogSeverityLevel(options: *const OrtRunOptions, log_severity_level: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunTag(options: *const OrtRunOptions, run_tag: *mut *const ::std::os::raw::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsUnsetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
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

unsafe extern "system" fn IsTensor(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
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
				CpuStorage::BF16(v) => v.as_ptr() as *mut _
			};
			OrtStatusPtr::default()
		}
		_ => Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
	}
}

unsafe extern "system" fn FillStringTensor(value: *mut OrtValue, s: *const *const ::std::os::raw::c_char, s_len: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorDataLength(value: *const OrtValue, len: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorContent(
	value: *const OrtValue,
	s: *mut ::std::os::raw::c_void,
	s_len: usize,
	offsets: *mut usize,
	offsets_len: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CastTypeInfoToTensorInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	*out = type_info.cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetOnnxTypeFromTypeInfo(type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr {
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
	*out = convert_dtype_to_sys(info.dtype);
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
	info: *const OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const ::std::os::raw::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	for i in 0..dim_params_length {
		*dim_params.add(i) = ptr::null();
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

unsafe extern "system" fn GetValueType(value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	*out = ONNXType::ONNX_TYPE_TENSOR;
	OrtStatusPtr::default()
}

unsafe extern "system" fn CreateMemoryInfo(
	name: *const ::std::os::raw::c_char,
	type_: OrtAllocatorType,
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

unsafe extern "system" fn CreateCpuMemoryInfo(type_: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
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

unsafe extern "system" fn MemoryInfoGetType(ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr {
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

unsafe extern "system" fn AddFreeDimensionOverride(
	options: *mut OrtSessionOptions,
	dim_denotation: *const ::std::os::raw::c_char,
	dim_value: i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetValue(
	value: *const OrtValue,
	index: ::std::os::raw::c_int,
	allocator: *mut OrtAllocator,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetValueCount(value: *const OrtValue, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateValue(in_: *const *const OrtValue, num_values: usize, value_type: ONNXType, out: *mut *mut OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateOpaqueValue(
	domain_name: *const ::std::os::raw::c_char,
	type_name: *const ::std::os::raw::c_char,
	data_container: *const ::std::os::raw::c_void,
	data_container_size: usize,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetOpaqueValue(
	domain_name: *const ::std::os::raw::c_char,
	type_name: *const ::std::os::raw::c_char,
	in_: *const OrtValue,
	data_container: *mut ::std::os::raw::c_void,
	data_container_size: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_float(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut f32) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_int64(info: *const OrtKernelInfo, name: *const ::std::os::raw::c_char, out: *mut i64) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_string(
	info: *const OrtKernelInfo,
	name: *const ::std::os::raw::c_char,
	out: *mut ::std::os::raw::c_char,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetInputCount(context: *const OrtKernelContext, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetOutputCount(context: *const OrtKernelContext, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetInput(context: *const OrtKernelContext, index: usize, out: *mut *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetOutput(
	context: *mut OrtKernelContext,
	index: usize,
	dim_values: *const i64,
	dim_count: usize,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
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

unsafe extern "system" fn ReleaseRunOptions(input: *mut OrtRunOptions) {}

unsafe extern "system" fn ReleaseTypeInfo(input: *mut OrtTypeInfo) {
	drop(TypeInfo::consume_sys(input));
}

unsafe extern "system" fn ReleaseTensorTypeAndShapeInfo(input: *mut OrtTensorTypeAndShapeInfo) {
	drop(TypeInfo::consume_sys(input.cast()));
}

unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {
	drop(Box::from_raw(input.cast::<SessionOptions>()));
}

unsafe extern "system" fn ReleaseCustomOpDomain(input: *mut OrtCustomOpDomain) {}

unsafe extern "system" fn GetDenotationFromTypeInfo(
	type_info: *const OrtTypeInfo,
	denotation: *mut *const ::std::os::raw::c_char,
	len: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CastTypeInfoToMapTypeInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtMapTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CastTypeInfoToSequenceTypeInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtSequenceTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetMapKeyType(map_type_info: *const OrtMapTypeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetMapValueType(map_type_info: *const OrtMapTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSequenceElementType(sequence_type_info: *const OrtSequenceTypeInfo, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseMapTypeInfo(input: *mut OrtMapTypeInfo) {}

unsafe extern "system" fn ReleaseSequenceTypeInfo(input: *mut OrtSequenceTypeInfo) {}

unsafe extern "system" fn SessionEndProfiling(session: *mut OrtSession, allocator: *mut OrtAllocator, out: *mut *mut ::std::os::raw::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetModelMetadata(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetProducerName(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetGraphName(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetDomain(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetDescription(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataLookupCustomMetadataMap(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	key: *const ::std::os::raw::c_char,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetVersion(model_metadata: *const OrtModelMetadata, value: *mut i64) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseModelMetadata(input: *mut OrtModelMetadata) {}

unsafe extern "system" fn CreateEnvWithGlobalThreadPools(
	log_severity_level: OrtLoggingLevel,
	logid: *const ::std::os::raw::c_char,
	tp_options: *const OrtThreadingOptions,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn DisablePerSessionThreads(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateThreadingOptions(out: *mut *mut OrtThreadingOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseThreadingOptions(input: *mut OrtThreadingOptions) {}

unsafe extern "system" fn ModelMetadataGetCustomMetadataMapKeys(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	keys: *mut *mut *mut ::std::os::raw::c_char,
	num_keys: *mut i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddFreeDimensionOverrideByName(
	options: *mut OrtSessionOptions,
	dim_name: *const ::std::os::raw::c_char,
	dim_value: i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetAvailableProviders(out_ptr: *mut *mut *mut ::std::os::raw::c_char, provider_length: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseAvailableProviders(ptr: *mut *mut ::std::os::raw::c_char, providers_length: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorElementLength(value: *const OrtValue, index: usize, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorElement(value: *const OrtValue, s_len: usize, index: usize, s: *mut ::std::os::raw::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillStringTensorElement(value: *mut OrtValue, s: *const ::std::os::raw::c_char, index: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddSessionConfigEntry(
	options: *mut OrtSessionOptions,
	config_key: *const ::std::os::raw::c_char,
	config_value: *const ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateAllocator(session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	let mem_info = unsafe { &*mem_info.cast::<MemoryInfo>() };
	*out = (Box::leak(Box::new(Allocator::new(mem_info))) as *mut Allocator).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {
	drop(Box::from_raw(input.cast::<Allocator>()));
}

unsafe extern "system" fn RunWithBinding(session: *mut OrtSession, run_options: *const OrtRunOptions, binding_ptr: *const OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateIoBinding(session: *mut OrtSession, out: *mut *mut OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseIoBinding(input: *mut OrtIoBinding) {}

unsafe extern "system" fn BindInput(binding_ptr: *mut OrtIoBinding, name: *const ::std::os::raw::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn BindOutput(binding_ptr: *mut OrtIoBinding, name: *const ::std::os::raw::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn BindOutputToDevice(
	binding_ptr: *mut OrtIoBinding,
	name: *const ::std::os::raw::c_char,
	mem_info_ptr: *const OrtMemoryInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetBoundOutputNames(
	binding_ptr: *const OrtIoBinding,
	allocator: *mut OrtAllocator,
	buffer: *mut *mut ::std::os::raw::c_char,
	lengths: *mut *mut usize,
	count: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetBoundOutputValues(
	binding_ptr: *const OrtIoBinding,
	allocator: *mut OrtAllocator,
	output: *mut *mut *mut OrtValue,
	output_count: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ClearBoundInputs(binding_ptr: *mut OrtIoBinding) {}

unsafe extern "system" fn ClearBoundOutputs(binding_ptr: *mut OrtIoBinding) {}

unsafe extern "system" fn TensorAt(
	value: *mut OrtValue,
	location_values: *const i64,
	location_values_count: usize,
	out: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateAndRegisterAllocator(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo, arena_cfg: *const OrtArenaCfg) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetLanguageProjection(ort_env: *const OrtEnv, projection: OrtLanguageProjection) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetProfilingStartTimeNs(session: *const OrtSession, out: *mut u64) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalIntraOpNumThreads(tp_options: *mut OrtThreadingOptions, intra_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalInterOpNumThreads(tp_options: *mut OrtThreadingOptions, inter_op_num_threads: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalSpinControl(tp_options: *mut OrtThreadingOptions, allow_spinning: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddInitializer(options: *mut OrtSessionOptions, name: *const ::std::os::raw::c_char, val: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateEnvWithCustomLoggerAndGlobalThreadPools(
	logging_function: OrtLoggingFunction,
	logger_param: *mut ::std::os::raw::c_void,
	log_severity_level: OrtLoggingLevel,
	logid: *const ::std::os::raw::c_char,
	tp_options: *const OrtThreadingOptions,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_CUDA(
	options: *mut OrtSessionOptions,
	cuda_options: *const OrtCUDAProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_ROCM(
	options: *mut OrtSessionOptions,
	rocm_options: *const OrtROCMProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_OpenVINO(
	options: *mut OrtSessionOptions,
	provider_options: *const OrtOpenVINOProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalDenormalAsZero(tp_options: *mut OrtThreadingOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateArenaCfg(
	max_mem: usize,
	arena_extend_strategy: ::std::os::raw::c_int,
	initial_chunk_size_bytes: ::std::os::raw::c_int,
	max_dead_bytes_per_chunk: ::std::os::raw::c_int,
	out: *mut *mut OrtArenaCfg
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseArenaCfg(input: *mut OrtArenaCfg) {}

unsafe extern "system" fn ModelMetadataGetGraphDescription(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_TensorRT(
	options: *mut OrtSessionOptions,
	tensorrt_options: *const OrtTensorRTProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetCurrentGpuDeviceId(device_id: ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCurrentGpuDeviceId(device_id: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttributeArray_float(
	info: *const OrtKernelInfo,
	name: *const ::std::os::raw::c_char,
	out: *mut f32,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttributeArray_int64(
	info: *const OrtKernelInfo,
	name: *const ::std::os::raw::c_char,
	out: *mut i64,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateArenaCfgV2(
	arena_config_keys: *const *const ::std::os::raw::c_char,
	arena_config_values: *const usize,
	num_keys: usize,
	out: *mut *mut OrtArenaCfg
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddRunConfigEntry(
	options: *mut OrtRunOptions,
	config_key: *const ::std::os::raw::c_char,
	config_value: *const ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreatePrepackedWeightsContainer(out: *mut *mut OrtPrepackedWeightsContainer) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleasePrepackedWeightsContainer(input: *mut OrtPrepackedWeightsContainer) {}

unsafe extern "system" fn CreateSessionWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_path: *const ortchar,
	options: *const OrtSessionOptions,
	prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSessionFromArrayWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_data: *const ::std::os::raw::c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_TensorRT_V2(
	options: *mut OrtSessionOptions,
	tensorrt_options: *const OrtTensorRTProviderOptionsV2
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateTensorRTProviderOptions(out: *mut *mut OrtTensorRTProviderOptionsV2) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateTensorRTProviderOptions(
	tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorRTProviderOptionsAsString(
	tensorrt_options: *const OrtTensorRTProviderOptionsV2,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseTensorRTProviderOptions(input: *mut OrtTensorRTProviderOptionsV2) {}

unsafe extern "system" fn EnableOrtCustomOps(options: *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterAllocator(env: *mut OrtEnv, allocator: *mut OrtAllocator) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UnregisterAllocator(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn IsSparseTensor(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSparseTensorAsOrtValue(
	allocator: *mut OrtAllocator,
	dense_shape: *const i64,
	dense_shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillSparseTensorCoo(
	ort_value: *mut OrtValue,
	data_mem_info: *const OrtMemoryInfo,
	values_shape: *const i64,
	values_shape_len: usize,
	values: *const ::std::os::raw::c_void,
	indices_data: *const i64,
	indices_num: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillSparseTensorCsr(
	ort_value: *mut OrtValue,
	data_mem_info: *const OrtMemoryInfo,
	values_shape: *const i64,
	values_shape_len: usize,
	values: *const ::std::os::raw::c_void,
	inner_indices_data: *const i64,
	inner_indices_num: usize,
	outer_indices_data: *const i64,
	outer_indices_num: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillSparseTensorBlockSparse(
	ort_value: *mut OrtValue,
	data_mem_info: *const OrtMemoryInfo,
	values_shape: *const i64,
	values_shape_len: usize,
	values: *const ::std::os::raw::c_void,
	indices_shape_data: *const i64,
	indices_shape_len: usize,
	indices_data: *const i32
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSparseTensorWithValuesAsOrtValue(
	info: *const OrtMemoryInfo,
	p_data: *mut ::std::os::raw::c_void,
	dense_shape: *const i64,
	dense_shape_len: usize,
	values_shape: *const i64,
	values_shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UseCooIndices(ort_value: *mut OrtValue, indices_data: *mut i64, indices_num: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UseCsrIndices(
	ort_value: *mut OrtValue,
	inner_data: *mut i64,
	inner_num: usize,
	outer_data: *mut i64,
	outer_num: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UseBlockSparseIndices(
	ort_value: *mut OrtValue,
	indices_shape: *const i64,
	indices_shape_len: usize,
	indices_data: *mut i32
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSparseTensorFormat(ort_value: *const OrtValue, out: *mut OrtSparseFormat) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSparseTensorValuesTypeAndShape(ort_value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSparseTensorValues(ort_value: *const OrtValue, out: *mut *const ::std::os::raw::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSparseTensorIndicesTypeShape(
	ort_value: *const OrtValue,
	indices_format: OrtSparseIndicesFormat,
	out: *mut *mut OrtTensorTypeAndShapeInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSparseTensorIndices(
	ort_value: *const OrtValue,
	indices_format: OrtSparseIndicesFormat,
	num_indices: *mut usize,
	indices: *mut *const ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn HasValue(value: *const OrtValue, out: *mut ::std::os::raw::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetGPUComputeStream(context: *const OrtKernelContext, out: *mut *mut ::std::os::raw::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorMemoryInfo(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	// `MemoryInfo` is #[repr(transparent)], so &MemoryInfo is &Device.
	*mem_info = (tensor.device() as *const Device).cast();
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetExecutionProviderApi(
	provider_name: *const ::std::os::raw::c_char,
	version: u32,
	provider_api: *mut *const ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsSetCustomCreateThreadFn(
	options: *mut OrtSessionOptions,
	ort_custom_create_thread_fn: OrtCustomCreateThreadFn
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsSetCustomThreadCreationOptions(
	options: *mut OrtSessionOptions,
	ort_custom_thread_creation_options: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsSetCustomJoinThreadFn(
	options: *mut OrtSessionOptions,
	ort_custom_join_thread_fn: OrtCustomJoinThreadFn
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalCustomCreateThreadFn(
	tp_options: *mut OrtThreadingOptions,
	ort_custom_create_thread_fn: OrtCustomCreateThreadFn
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalCustomThreadCreationOptions(
	tp_options: *mut OrtThreadingOptions,
	ort_custom_thread_creation_options: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalCustomJoinThreadFn(tp_options: *mut OrtThreadingOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SynchronizeBoundInputs(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SynchronizeBoundOutputs(binding_ptr: *mut OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_CUDA_V2(
	options: *mut OrtSessionOptions,
	cuda_options: *const OrtCUDAProviderOptionsV2
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateCUDAProviderOptions(out: *mut *mut OrtCUDAProviderOptionsV2) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateCUDAProviderOptions(
	cuda_options: *mut OrtCUDAProviderOptionsV2,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCUDAProviderOptionsAsString(
	cuda_options: *const OrtCUDAProviderOptionsV2,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseCUDAProviderOptions(input: *mut OrtCUDAProviderOptionsV2) {}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_MIGraphX(
	options: *mut OrtSessionOptions,
	migraphx_options: *const OrtMIGraphXProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddExternalInitializers(
	options: *mut OrtSessionOptions,
	initializer_names: *const *const ::std::os::raw::c_char,
	initializers: *const *const OrtValue,
	initializers_num: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateOpAttr(
	name: *const ::std::os::raw::c_char,
	data: *const ::std::os::raw::c_void,
	len: ::std::os::raw::c_int,
	type_: OrtOpAttrType,
	op_attr: *mut *mut OrtOpAttr
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseOpAttr(input: *mut OrtOpAttr) {}

unsafe extern "system" fn CreateOp(
	info: *const OrtKernelInfo,
	op_name: *const ::std::os::raw::c_char,
	domain: *const ::std::os::raw::c_char,
	version: ::std::os::raw::c_int,
	type_constraint_names: *mut *const ::std::os::raw::c_char,
	type_constraint_values: *const ONNXTensorElementDataType,
	type_constraint_count: ::std::os::raw::c_int,
	attr_values: *const *const OrtOpAttr,
	attr_count: ::std::os::raw::c_int,
	input_count: ::std::os::raw::c_int,
	output_count: ::std::os::raw::c_int,
	ort_op: *mut *mut OrtOp
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn InvokeOp(
	context: *const OrtKernelContext,
	ort_op: *const OrtOp,
	input_values: *const *const OrtValue,
	input_count: ::std::os::raw::c_int,
	output_values: *const *mut OrtValue,
	output_count: ::std::os::raw::c_int
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseOp(input: *mut OrtOp) {}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider(
	options: *mut OrtSessionOptions,
	provider_name: *const ::std::os::raw::c_char,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CopyKernelInfo(info: *const OrtKernelInfo, info_copy: *mut *mut OrtKernelInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseKernelInfo(input: *mut OrtKernelInfo) {}

unsafe extern "system" fn GetTrainingApi(version: u32) -> *const OrtTrainingApi {
	ptr::null()
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_CANN(
	options: *mut OrtSessionOptions,
	cann_options: *const OrtCANNProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateCANNProviderOptions(out: *mut *mut OrtCANNProviderOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateCANNProviderOptions(
	cann_options: *mut OrtCANNProviderOptions,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCANNProviderOptionsAsString(
	cann_options: *const OrtCANNProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseCANNProviderOptions(input: *mut OrtCANNProviderOptions) {}

unsafe extern "system" fn MemoryInfoGetDeviceType(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType) {
	let memory_info = unsafe { &*ptr.cast::<MemoryInfo>() };
	*out = memory_info.device_type();
}

unsafe extern "system" fn UpdateEnvWithCustomLogLevel(ort_env: *mut OrtEnv, log_severity_level: OrtLoggingLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalIntraOpThreadAffinity(tp_options: *mut OrtThreadingOptions, affinity_string: *const ::std::os::raw::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterCustomOpsLibrary_V2(options: *mut OrtSessionOptions, library_name: *const ortchar) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterCustomOpsUsingFunction(
	options: *mut OrtSessionOptions,
	registration_func_name: *const ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetInputCount(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetOutputCount(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetInputName(
	info: *const OrtKernelInfo,
	index: usize,
	out: *mut ::std::os::raw::c_char,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetOutputName(
	info: *const OrtKernelInfo,
	index: usize,
	out: *mut ::std::os::raw::c_char,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetInputTypeInfo(info: *const OrtKernelInfo, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetOutputTypeInfo(info: *const OrtKernelInfo, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_tensor(
	info: *const OrtKernelInfo,
	name: *const ::std::os::raw::c_char,
	allocator: *mut OrtAllocator,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn HasSessionConfigEntry(
	options: *const OrtSessionOptions,
	config_key: *const ::std::os::raw::c_char,
	out: *mut ::std::os::raw::c_int
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSessionConfigEntry(
	options: *const OrtSessionOptions,
	config_key: *const ::std::os::raw::c_char,
	config_value: *mut ::std::os::raw::c_char,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_Dnnl(
	options: *mut OrtSessionOptions,
	dnnl_options: *const OrtDnnlProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateDnnlProviderOptions(out: *mut *mut OrtDnnlProviderOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateDnnlProviderOptions(
	dnnl_options: *mut OrtDnnlProviderOptions,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetDnnlProviderOptionsAsString(
	dnnl_options: *const OrtDnnlProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseDnnlProviderOptions(input: *mut OrtDnnlProviderOptions) {}

unsafe extern "system" fn KernelInfo_GetNodeName(info: *const OrtKernelInfo, out: *mut ::std::os::raw::c_char, size: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetLogger(info: *const OrtKernelInfo, logger: *mut *const OrtLogger) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetLogger(context: *const OrtKernelContext, logger: *mut *const OrtLogger) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn Logger_LogMessage(
	logger: *const OrtLogger,
	log_severity_level: OrtLoggingLevel,
	message: *const ::std::os::raw::c_char,
	file_path: *const ortchar,
	line_number: ::std::os::raw::c_int,
	func_name: *const ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn Logger_GetLoggingSeverityLevel(logger: *const OrtLogger, out: *mut OrtLoggingLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetConstantInput_tensor(
	info: *const OrtKernelInfo,
	index: usize,
	is_constant: *mut ::std::os::raw::c_int,
	out: *mut *const OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CastTypeInfoToOptionalTypeInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtOptionalTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetOptionalContainedTypeInfo(optional_type_info: *const OrtOptionalTypeInfo, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetResizedStringTensorElementBuffer(
	value: *mut OrtValue,
	index: usize,
	length_in_bytes: usize,
	buffer: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetAllocator(
	context: *const OrtKernelContext,
	mem_info: *const OrtMemoryInfo,
	out: *mut *mut OrtAllocator
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetBuildInfoString() -> *const ::std::os::raw::c_char {
	concat!("ORT Build Info: backend=ort-candle, version=", env!("CARGO_PKG_VERSION"), "\0")
		.as_ptr()
		.cast()
}

unsafe extern "system" fn CreateROCMProviderOptions(out: *mut *mut OrtROCMProviderOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateROCMProviderOptions(
	rocm_options: *mut OrtROCMProviderOptions,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetROCMProviderOptionsAsString(
	rocm_options: *const OrtROCMProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::std::os::raw::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseROCMProviderOptions(input: *mut OrtROCMProviderOptions) {}

unsafe extern "system" fn CreateAndRegisterAllocatorV2(
	env: *mut OrtEnv,
	provider_type: *const ::std::os::raw::c_char,
	mem_info: *const OrtMemoryInfo,
	arena_cfg: *const OrtArenaCfg,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunAsync(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: *const *const ::std::os::raw::c_char,
	input: *const *const OrtValue,
	input_len: usize,
	output_names: *const *const ::std::os::raw::c_char,
	output_names_len: usize,
	output: *mut *mut OrtValue,
	run_async_callback: RunAsyncCallbackFn,
	user_data: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateTensorRTProviderOptionsWithValue(
	tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
	key: *const ::std::os::raw::c_char,
	value: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorRTProviderOptionsByName(
	tensorrt_options: *const OrtTensorRTProviderOptionsV2,
	key: *const ::std::os::raw::c_char,
	ptr: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateCUDAProviderOptionsWithValue(
	cuda_options: *mut OrtCUDAProviderOptionsV2,
	key: *const ::std::os::raw::c_char,
	value: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCUDAProviderOptionsByName(
	cuda_options: *const OrtCUDAProviderOptionsV2,
	key: *const ::std::os::raw::c_char,
	ptr: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetResource(
	context: *const OrtKernelContext,
	resouce_version: ::std::os::raw::c_int,
	resource_id: ::std::os::raw::c_int,
	resource: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetUserLoggingFunction(
	options: *mut OrtSessionOptions,
	user_logging_function: OrtLoggingFunction,
	user_logging_param: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ShapeInferContext_GetInputCount(context: *const OrtShapeInferContext, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ShapeInferContext_GetInputTypeShape(
	context: *const OrtShapeInferContext,
	index: usize,
	info: *mut *mut OrtTensorTypeAndShapeInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ShapeInferContext_GetAttribute(
	context: *const OrtShapeInferContext,
	attr_name: *const ::std::os::raw::c_char,
	attr: *mut *const OrtOpAttr
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ShapeInferContext_SetOutputTypeShape(
	context: *const OrtShapeInferContext,
	index: usize,
	info: *const OrtTensorTypeAndShapeInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSymbolicDimensions(
	info: *mut OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const ::std::os::raw::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReadOpAttr(
	op_attr: *const OrtOpAttr,
	type_: OrtOpAttrType,
	data: *mut ::std::os::raw::c_void,
	len: usize,
	out: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetDeterministicCompute(options: *mut OrtSessionOptions, value: bool) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_ParallelFor(
	context: *const OrtKernelContext,
	fn_: unsafe extern "system" fn(arg1: *mut ::std::os::raw::c_void, arg2: usize),
	total: usize,
	num_batch: usize,
	usr_data: *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_OpenVINO_V2(
	options: *mut OrtSessionOptions,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_VitisAI(
	options: *mut OrtSessionOptions,
	provider_options_keys: *const *const ::std::os::raw::c_char,
	provider_options_values: *const *const ::std::os::raw::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetScratchBuffer(
	context: *const OrtKernelContext,
	mem_info: *const OrtMemoryInfo,
	count_or_bytes: usize,
	out: *mut *mut ::std::os::raw::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAllocator(info: *const OrtKernelInfo, mem_type: OrtMemType, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddExternalInitializersFromMemory(
	options: *mut OrtSessionOptions,
	external_initializer_file_names: *const *const ortchar,
	external_initializer_file_buffer_array: *const *mut ::std::os::raw::c_char,
	external_initializer_file_lengths: *const usize,
	num_external_initializer_files: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateLoraAdapter(adapter_file_path: *const ortchar, allocator: *mut OrtAllocator, out: *mut *mut OrtLoraAdapter) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateLoraAdapterFromArray(
	bytes: *const ::std::os::raw::c_void,
	num_bytes: usize,
	allocator: *mut OrtAllocator,
	out: *mut *mut OrtLoraAdapter
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseLoraAdapter(input: *mut OrtLoraAdapter) {}

unsafe extern "system" fn RunOptionsAddActiveLoraAdapter(options: *mut OrtRunOptions, adapter: *const OrtLoraAdapter) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetEpDynamicOptions(
	sess: *mut OrtSession,
	keys: *const *const ::std::os::raw::c_char,
	values: *const *const ::std::os::raw::c_char,
	kv_len: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
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
		SetOptimizedModelFilePath,
		CloneSessionOptions,
		SetSessionExecutionMode,
		EnableProfiling,
		DisableProfiling,
		EnableMemPattern,
		DisableMemPattern,
		EnableCpuMemArena,
		DisableCpuMemArena,
		SetSessionLogId,
		SetSessionLogVerbosityLevel,
		SetSessionLogSeverityLevel,
		SetSessionGraphOptimizationLevel,
		SetIntraOpNumThreads,
		SetInterOpNumThreads,
		CreateCustomOpDomain,
		CustomOpDomain_Add,
		AddCustomOpDomain,
		RegisterCustomOpsLibrary,
		SessionGetInputCount,
		SessionGetOutputCount,
		SessionGetOverridableInitializerCount,
		SessionGetInputTypeInfo,
		SessionGetOutputTypeInfo,
		SessionGetOverridableInitializerTypeInfo,
		SessionGetInputName,
		SessionGetOutputName,
		SessionGetOverridableInitializerName,
		CreateRunOptions,
		RunOptionsSetRunLogVerbosityLevel,
		RunOptionsSetRunLogSeverityLevel,
		RunOptionsSetRunTag,
		RunOptionsGetRunLogVerbosityLevel,
		RunOptionsGetRunLogSeverityLevel,
		RunOptionsGetRunTag,
		RunOptionsSetTerminate,
		RunOptionsUnsetTerminate,
		CreateTensorAsOrtValue,
		CreateTensorWithDataAsOrtValue,
		IsTensor,
		GetTensorMutableData,
		FillStringTensor,
		GetStringTensorDataLength,
		GetStringTensorContent,
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
		AddFreeDimensionOverride,
		GetValue,
		GetValueCount,
		CreateValue,
		CreateOpaqueValue,
		GetOpaqueValue,
		KernelInfoGetAttribute_float,
		KernelInfoGetAttribute_int64,
		KernelInfoGetAttribute_string,
		KernelContext_GetInputCount,
		KernelContext_GetOutputCount,
		KernelContext_GetInput,
		KernelContext_GetOutput,
		ReleaseEnv,
		ReleaseStatus,
		ReleaseMemoryInfo,
		ReleaseSession,
		ReleaseValue,
		ReleaseRunOptions,
		ReleaseTypeInfo,
		ReleaseTensorTypeAndShapeInfo,
		ReleaseSessionOptions,
		ReleaseCustomOpDomain,
		GetDenotationFromTypeInfo,
		CastTypeInfoToMapTypeInfo,
		CastTypeInfoToSequenceTypeInfo,
		GetMapKeyType,
		GetMapValueType,
		GetSequenceElementType,
		ReleaseMapTypeInfo,
		ReleaseSequenceTypeInfo,
		SessionEndProfiling,
		SessionGetModelMetadata,
		ModelMetadataGetProducerName,
		ModelMetadataGetGraphName,
		ModelMetadataGetDomain,
		ModelMetadataGetDescription,
		ModelMetadataLookupCustomMetadataMap,
		ModelMetadataGetVersion,
		ReleaseModelMetadata,
		CreateEnvWithGlobalThreadPools,
		DisablePerSessionThreads,
		CreateThreadingOptions,
		ReleaseThreadingOptions,
		ModelMetadataGetCustomMetadataMapKeys,
		AddFreeDimensionOverrideByName,
		GetAvailableProviders,
		ReleaseAvailableProviders,
		GetStringTensorElementLength,
		GetStringTensorElement,
		FillStringTensorElement,
		AddSessionConfigEntry,
		CreateAllocator,
		ReleaseAllocator,
		RunWithBinding,
		CreateIoBinding,
		ReleaseIoBinding,
		BindInput,
		BindOutput,
		BindOutputToDevice,
		GetBoundOutputNames,
		GetBoundOutputValues,
		ClearBoundInputs,
		ClearBoundOutputs,
		TensorAt,
		CreateAndRegisterAllocator,
		SetLanguageProjection,
		SessionGetProfilingStartTimeNs,
		SetGlobalIntraOpNumThreads,
		SetGlobalInterOpNumThreads,
		SetGlobalSpinControl,
		AddInitializer,
		CreateEnvWithCustomLoggerAndGlobalThreadPools,
		SessionOptionsAppendExecutionProvider_CUDA,
		SessionOptionsAppendExecutionProvider_ROCM,
		SessionOptionsAppendExecutionProvider_OpenVINO,
		SetGlobalDenormalAsZero,
		CreateArenaCfg,
		ReleaseArenaCfg,
		ModelMetadataGetGraphDescription,
		SessionOptionsAppendExecutionProvider_TensorRT,
		SetCurrentGpuDeviceId,
		GetCurrentGpuDeviceId,
		KernelInfoGetAttributeArray_float,
		KernelInfoGetAttributeArray_int64,
		CreateArenaCfgV2,
		AddRunConfigEntry,
		CreatePrepackedWeightsContainer,
		ReleasePrepackedWeightsContainer,
		CreateSessionWithPrepackedWeightsContainer,
		CreateSessionFromArrayWithPrepackedWeightsContainer,
		SessionOptionsAppendExecutionProvider_TensorRT_V2,
		CreateTensorRTProviderOptions,
		UpdateTensorRTProviderOptions,
		GetTensorRTProviderOptionsAsString,
		ReleaseTensorRTProviderOptions,
		EnableOrtCustomOps,
		RegisterAllocator,
		UnregisterAllocator,
		IsSparseTensor,
		CreateSparseTensorAsOrtValue,
		FillSparseTensorCoo,
		FillSparseTensorCsr,
		FillSparseTensorBlockSparse,
		CreateSparseTensorWithValuesAsOrtValue,
		UseCooIndices,
		UseCsrIndices,
		UseBlockSparseIndices,
		GetSparseTensorFormat,
		GetSparseTensorValuesTypeAndShape,
		GetSparseTensorValues,
		GetSparseTensorIndicesTypeShape,
		GetSparseTensorIndices,
		HasValue,
		KernelContext_GetGPUComputeStream,
		GetTensorMemoryInfo,
		GetExecutionProviderApi,
		SessionOptionsSetCustomCreateThreadFn,
		SessionOptionsSetCustomThreadCreationOptions,
		SessionOptionsSetCustomJoinThreadFn,
		SetGlobalCustomCreateThreadFn,
		SetGlobalCustomThreadCreationOptions,
		SetGlobalCustomJoinThreadFn,
		SynchronizeBoundInputs,
		SynchronizeBoundOutputs,
		SessionOptionsAppendExecutionProvider_CUDA_V2,
		CreateCUDAProviderOptions,
		UpdateCUDAProviderOptions,
		GetCUDAProviderOptionsAsString,
		ReleaseCUDAProviderOptions,
		SessionOptionsAppendExecutionProvider_MIGraphX,
		AddExternalInitializers,
		CreateOpAttr,
		ReleaseOpAttr,
		CreateOp,
		InvokeOp,
		ReleaseOp,
		SessionOptionsAppendExecutionProvider,
		CopyKernelInfo,
		ReleaseKernelInfo,
		GetTrainingApi,
		SessionOptionsAppendExecutionProvider_CANN,
		CreateCANNProviderOptions,
		UpdateCANNProviderOptions,
		GetCANNProviderOptionsAsString,
		ReleaseCANNProviderOptions,
		MemoryInfoGetDeviceType,
		UpdateEnvWithCustomLogLevel,
		SetGlobalIntraOpThreadAffinity,
		RegisterCustomOpsLibrary_V2,
		RegisterCustomOpsUsingFunction,
		KernelInfo_GetInputCount,
		KernelInfo_GetOutputCount,
		KernelInfo_GetInputName,
		KernelInfo_GetOutputName,
		KernelInfo_GetInputTypeInfo,
		KernelInfo_GetOutputTypeInfo,
		KernelInfoGetAttribute_tensor,
		HasSessionConfigEntry,
		GetSessionConfigEntry,
		SessionOptionsAppendExecutionProvider_Dnnl,
		CreateDnnlProviderOptions,
		UpdateDnnlProviderOptions,
		GetDnnlProviderOptionsAsString,
		ReleaseDnnlProviderOptions,
		KernelInfo_GetNodeName,
		KernelInfo_GetLogger,
		KernelContext_GetLogger,
		Logger_LogMessage,
		Logger_GetLoggingSeverityLevel,
		KernelInfoGetConstantInput_tensor,
		CastTypeInfoToOptionalTypeInfo,
		GetOptionalContainedTypeInfo,
		GetResizedStringTensorElementBuffer,
		KernelContext_GetAllocator,
		GetBuildInfoString,
		CreateROCMProviderOptions,
		UpdateROCMProviderOptions,
		GetROCMProviderOptionsAsString,
		ReleaseROCMProviderOptions,
		CreateAndRegisterAllocatorV2,
		RunAsync,
		UpdateTensorRTProviderOptionsWithValue,
		GetTensorRTProviderOptionsByName,
		UpdateCUDAProviderOptionsWithValue,
		GetCUDAProviderOptionsByName,
		KernelContext_GetResource,
		SetUserLoggingFunction,
		ShapeInferContext_GetInputCount,
		ShapeInferContext_GetInputTypeShape,
		ShapeInferContext_GetAttribute,
		ShapeInferContext_SetOutputTypeShape,
		SetSymbolicDimensions,
		ReadOpAttr,
		SetDeterministicCompute,
		KernelContext_ParallelFor,
		SessionOptionsAppendExecutionProvider_OpenVINO_V2,
		SessionOptionsAppendExecutionProvider_VitisAI,
		KernelContext_GetScratchBuffer,
		KernelInfoGetAllocator,
		AddExternalInitializersFromMemory,
		CreateLoraAdapter,
		CreateLoraAdapterFromArray,
		ReleaseLoraAdapter,
		RunOptionsAddActiveLoraAdapter,
		SetEpDynamicOptions
	}
}
