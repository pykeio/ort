#![allow(non_snake_case, unused)]

use alloc::{boxed::Box, ffi::CString, string::String};
use core::{ffi::c_char, ptr};

use crate::*;

#[derive(Debug, Clone)]
pub struct Error {
	pub code: OrtErrorCode,
	message: CString
}

impl Error {
	pub fn new(code: OrtErrorCode, message: impl Into<String>) -> Self {
		Self {
			code,
			message: CString::new(message.into()).unwrap()
		}
	}

	pub fn into_sys(self) -> OrtStatusPtr {
		OrtStatusPtr((Box::leak(Box::new(self)) as *mut Error).cast())
	}

	pub fn new_sys(code: OrtErrorCode, message: impl Into<String>) -> OrtStatusPtr {
		Self::new(code, message).into_sys()
	}

	#[inline]
	pub fn message(&self) -> &str {
		self.message.as_c_str().to_str().unwrap()
	}

	#[inline]
	pub fn message_ptr(&self) -> *const c_char {
		self.message.as_ptr()
	}

	pub unsafe fn cast_from_sys<'e>(status: *const OrtStatus) -> &'e Error {
		unsafe { &*status.cast::<Error>() }
	}

	pub unsafe fn consume_sys(status: *mut OrtStatus) -> Box<Error> {
		unsafe { Box::from_raw(status.cast::<Error>()) }
	}
}

unsafe extern "system" fn CreateStatus(code: OrtErrorCode, msg: *const ::core::ffi::c_char) -> OrtStatusPtr {
	let msg = unsafe { CString::from_raw(msg.cast_mut()) };
	Error::new_sys(code, msg.to_string_lossy())
}

unsafe extern "system" fn GetErrorCode(status: *const OrtStatus) -> OrtErrorCode {
	unsafe { Error::cast_from_sys(status) }.code
}

unsafe extern "system" fn GetErrorMessage(status: *const OrtStatus) -> *const ::core::ffi::c_char {
	unsafe { Error::cast_from_sys(status) }.message_ptr()
}

unsafe extern "system" fn CreateEnv(log_severity_level: OrtLoggingLevel, logid: *const ::core::ffi::c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateEnvWithCustomLogger(
	logging_function: OrtLoggingFunction,
	logger_param: *mut ::core::ffi::c_void,
	log_severity_level: OrtLoggingLevel,
	logid: *const ::core::ffi::c_char,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn EnableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn DisableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "system" fn CreateSession(
	env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(target_arch = "wasm32")]
unsafe fn CreateSession(
	env: *const OrtEnv,
	model_path: &str,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> core::pin::Pin<alloc::boxed::Box<dyn core::future::Future<Output = OrtStatusPtr>>> {
	Box::pin(async { Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented") })
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "system" fn CreateSessionFromArray(
	env: *const OrtEnv,
	model_data: *const ::core::ffi::c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(target_arch = "wasm32")]
unsafe fn CreateSessionFromArray(
	env: *const OrtEnv,
	model_data: &[u8],
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> core::pin::Pin<alloc::boxed::Box<dyn core::future::Future<Output = OrtStatusPtr>>> {
	Box::pin(async { Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented") })
}

unsafe extern "system" fn Run(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: *const *const ::core::ffi::c_char,
	inputs: *const *const OrtValue,
	input_len: usize,
	output_names: *const *const ::core::ffi::c_char,
	output_names_len: usize,
	output_ptrs: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSessionOptions(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetOptimizedModelFilePath(options: *mut OrtSessionOptions, optimized_model_filepath: *const os_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionExecutionMode(options: *mut OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn EnableProfiling(options: *mut OrtSessionOptions, profile_file_prefix: *const os_char) -> OrtStatusPtr {
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

unsafe extern "system" fn SetSessionLogId(options: *mut OrtSessionOptions, logid: *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionLogVerbosityLevel(options: *mut OrtSessionOptions, session_log_verbosity_level: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionLogSeverityLevel(options: *mut OrtSessionOptions, session_log_severity_level: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetSessionGraphOptimizationLevel(options: *mut OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetIntraOpNumThreads(options: *mut OrtSessionOptions, intra_op_num_threads: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetInterOpNumThreads(options: *mut OrtSessionOptions, inter_op_num_threads: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateCustomOpDomain(domain: *const ::core::ffi::c_char, out: *mut *mut OrtCustomOpDomain) -> OrtStatusPtr {
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
	library_path: *const ::core::ffi::c_char,
	library_handle: *mut *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOverridableInitializerCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOverridableInitializerTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetOverridableInitializerName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateRunOptions(out: *mut *mut OrtRunOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunLogVerbosityLevel(options: *mut OrtRunOptions, log_verbosity_level: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunLogSeverityLevel(options: *mut OrtRunOptions, log_severity_level: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsSetRunTag(options: *mut OrtRunOptions, run_tag: *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunLogVerbosityLevel(options: *const OrtRunOptions, log_verbosity_level: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunLogSeverityLevel(options: *const OrtRunOptions, log_severity_level: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RunOptionsGetRunTag(options: *const OrtRunOptions, run_tag: *mut *const ::core::ffi::c_char) -> OrtStatusPtr {
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
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateTensorWithDataAsOrtValue(
	info: *const OrtMemoryInfo,
	p_data: *mut ::core::ffi::c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn IsTensor(value: *const OrtValue, out: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorMutableData(value: *mut OrtValue, out: *mut *mut ::core::ffi::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillStringTensor(value: *mut OrtValue, s: *const *const ::core::ffi::c_char, s_len: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorDataLength(value: *const OrtValue, len: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorContent(
	value: *const OrtValue,
	s: *mut ::core::ffi::c_void,
	s_len: usize,
	offsets: *mut usize,
	offsets_len: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CastTypeInfoToTensorInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetOnnxTypeFromTypeInfo(type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateTensorTypeAndShapeInfo(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetTensorElementType(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetDimensions(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorElementType(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetDimensionsCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetDimensions(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSymbolicDimensions(
	info: *const OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const ::core::ffi::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorShapeElementCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorTypeAndShape(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTypeInfo(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetValueType(value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateMemoryInfo(
	name: *const ::core::ffi::c_char,
	type_: OrtAllocatorType,
	id: ::core::ffi::c_int,
	mem_type: OrtMemType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateCpuMemoryInfo(type_: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CompareMemoryInfo(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn MemoryInfoGetName(ptr: *const OrtMemoryInfo, out: *mut *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn MemoryInfoGetId(ptr: *const OrtMemoryInfo, out: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn MemoryInfoGetMemType(ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn MemoryInfoGetType(ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr {
	unsafe { *out = OrtAllocatorType::OrtDeviceAllocator };
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorAlloc(ort_allocator: *mut OrtAllocator, size: usize, out: *mut *mut ::core::ffi::c_void) -> OrtStatusPtr {
	unsafe { *out = (&*ort_allocator).Alloc.unwrap()(ort_allocator, size) };
	if unsafe { *out }.is_null() {
		return Error::new_sys(OrtErrorCode::ORT_RUNTIME_EXCEPTION, "Allocation failed");
	}
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorFree(ort_allocator: *mut OrtAllocator, p: *mut ::core::ffi::c_void) -> OrtStatusPtr {
	unsafe { (&*ort_allocator).Free.unwrap()(ort_allocator, p) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn AllocatorGetInfo(ort_allocator: *const OrtAllocator, out: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	unsafe { *out = (&*ort_allocator).Info.unwrap()(ort_allocator) };
	OrtStatusPtr::default()
}

unsafe extern "system" fn GetAllocatorWithDefaultOptions(out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddFreeDimensionOverride(
	options: *mut OrtSessionOptions,
	dim_denotation: *const ::core::ffi::c_char,
	dim_value: i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetValue(value: *const OrtValue, index: ::core::ffi::c_int, allocator: *mut OrtAllocator, out: *mut *mut OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetValueCount(value: *const OrtValue, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateValue(in_: *const *const OrtValue, num_values: usize, value_type: ONNXType, out: *mut *mut OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateOpaqueValue(
	domain_name: *const ::core::ffi::c_char,
	type_name: *const ::core::ffi::c_char,
	data_container: *const ::core::ffi::c_void,
	data_container_size: usize,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetOpaqueValue(
	domain_name: *const ::core::ffi::c_char,
	type_name: *const ::core::ffi::c_char,
	in_: *const OrtValue,
	data_container: *mut ::core::ffi::c_void,
	data_container_size: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_float(info: *const OrtKernelInfo, name: *const ::core::ffi::c_char, out: *mut f32) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_int64(info: *const OrtKernelInfo, name: *const ::core::ffi::c_char, out: *mut i64) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttribute_string(
	info: *const OrtKernelInfo,
	name: *const ::core::ffi::c_char,
	out: *mut ::core::ffi::c_char,
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

unsafe extern "system" fn ReleaseEnv(input: *mut OrtEnv) {}

unsafe extern "system" fn ReleaseStatus(input: *mut OrtStatus) {
	drop(unsafe { Error::consume_sys(input) });
}

unsafe extern "system" fn ReleaseMemoryInfo(input: *mut OrtMemoryInfo) {}

unsafe extern "system" fn ReleaseSession(input: *mut OrtSession) {}

unsafe extern "system" fn ReleaseValue(input: *mut OrtValue) {}

unsafe extern "system" fn ReleaseRunOptions(input: *mut OrtRunOptions) {}

unsafe extern "system" fn ReleaseTypeInfo(input: *mut OrtTypeInfo) {}

unsafe extern "system" fn ReleaseTensorTypeAndShapeInfo(input: *mut OrtTensorTypeAndShapeInfo) {}

unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {}

unsafe extern "system" fn ReleaseCustomOpDomain(input: *mut OrtCustomOpDomain) {}

unsafe extern "system" fn GetDenotationFromTypeInfo(
	type_info: *const OrtTypeInfo,
	denotation: *mut *const ::core::ffi::c_char,
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

unsafe extern "system" fn SessionEndProfiling(session: *mut OrtSession, allocator: *mut OrtAllocator, out: *mut *mut ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionGetModelMetadata(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetProducerName(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetGraphName(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetDomain(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetDescription(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataLookupCustomMetadataMap(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	key: *const ::core::ffi::c_char,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ModelMetadataGetVersion(model_metadata: *const OrtModelMetadata, value: *mut i64) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseModelMetadata(input: *mut OrtModelMetadata) {}

unsafe extern "system" fn CreateEnvWithGlobalThreadPools(
	log_severity_level: OrtLoggingLevel,
	logid: *const ::core::ffi::c_char,
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
	keys: *mut *mut *mut ::core::ffi::c_char,
	num_keys: *mut i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddFreeDimensionOverrideByName(
	options: *mut OrtSessionOptions,
	dim_name: *const ::core::ffi::c_char,
	dim_value: i64
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetAvailableProviders(out_ptr: *mut *mut *mut ::core::ffi::c_char, provider_length: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseAvailableProviders(ptr: *mut *mut ::core::ffi::c_char, providers_length: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorElementLength(value: *const OrtValue, index: usize, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetStringTensorElement(value: *const OrtValue, s_len: usize, index: usize, s: *mut ::core::ffi::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn FillStringTensorElement(value: *mut OrtValue, s: *const ::core::ffi::c_char, index: usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddSessionConfigEntry(
	options: *mut OrtSessionOptions,
	config_key: *const ::core::ffi::c_char,
	config_value: *const ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateAllocator(session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {}

unsafe extern "system" fn RunWithBinding(session: *mut OrtSession, run_options: *const OrtRunOptions, binding_ptr: *const OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateIoBinding(session: *mut OrtSession, out: *mut *mut OrtIoBinding) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseIoBinding(input: *mut OrtIoBinding) {}

unsafe extern "system" fn BindInput(binding_ptr: *mut OrtIoBinding, name: *const ::core::ffi::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn BindOutput(binding_ptr: *mut OrtIoBinding, name: *const ::core::ffi::c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn BindOutputToDevice(
	binding_ptr: *mut OrtIoBinding,
	name: *const ::core::ffi::c_char,
	mem_info_ptr: *const OrtMemoryInfo
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetBoundOutputNames(
	binding_ptr: *const OrtIoBinding,
	allocator: *mut OrtAllocator,
	buffer: *mut *mut ::core::ffi::c_char,
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
	out: *mut *mut ::core::ffi::c_void
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

unsafe extern "system" fn SetGlobalIntraOpNumThreads(tp_options: *mut OrtThreadingOptions, intra_op_num_threads: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalInterOpNumThreads(tp_options: *mut OrtThreadingOptions, inter_op_num_threads: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalSpinControl(tp_options: *mut OrtThreadingOptions, allow_spinning: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddInitializer(options: *mut OrtSessionOptions, name: *const ::core::ffi::c_char, val: *const OrtValue) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateEnvWithCustomLoggerAndGlobalThreadPools(
	logging_function: OrtLoggingFunction,
	logger_param: *mut ::core::ffi::c_void,
	log_severity_level: OrtLoggingLevel,
	logid: *const ::core::ffi::c_char,
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
	arena_extend_strategy: ::core::ffi::c_int,
	initial_chunk_size_bytes: ::core::ffi::c_int,
	max_dead_bytes_per_chunk: ::core::ffi::c_int,
	out: *mut *mut OrtArenaCfg
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseArenaCfg(input: *mut OrtArenaCfg) {}

unsafe extern "system" fn ModelMetadataGetGraphDescription(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	value: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_TensorRT(
	options: *mut OrtSessionOptions,
	tensorrt_options: *const OrtTensorRTProviderOptions
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetCurrentGpuDeviceId(device_id: ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCurrentGpuDeviceId(device_id: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttributeArray_float(
	info: *const OrtKernelInfo,
	name: *const ::core::ffi::c_char,
	out: *mut f32,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetAttributeArray_int64(
	info: *const OrtKernelInfo,
	name: *const ::core::ffi::c_char,
	out: *mut i64,
	size: *mut usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateArenaCfgV2(
	arena_config_keys: *const *const ::core::ffi::c_char,
	arena_config_values: *const usize,
	num_keys: usize,
	out: *mut *mut OrtArenaCfg
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn AddRunConfigEntry(
	options: *mut OrtRunOptions,
	config_key: *const ::core::ffi::c_char,
	config_value: *const ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreatePrepackedWeightsContainer(out: *mut *mut OrtPrepackedWeightsContainer) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleasePrepackedWeightsContainer(input: *mut OrtPrepackedWeightsContainer) {}

unsafe extern "system" fn CreateSessionWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSessionFromArrayWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_data: *const ::core::ffi::c_void,
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
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorRTProviderOptionsAsString(
	tensorrt_options: *const OrtTensorRTProviderOptionsV2,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::core::ffi::c_char
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

unsafe extern "system" fn IsSparseTensor(value: *const OrtValue, out: *mut ::core::ffi::c_int) -> OrtStatusPtr {
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
	values: *const ::core::ffi::c_void,
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
	values: *const ::core::ffi::c_void,
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
	values: *const ::core::ffi::c_void,
	indices_shape_data: *const i64,
	indices_shape_len: usize,
	indices_data: *const i32
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateSparseTensorWithValuesAsOrtValue(
	info: *const OrtMemoryInfo,
	p_data: *mut ::core::ffi::c_void,
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

unsafe extern "system" fn GetSparseTensorValues(ort_value: *const OrtValue, out: *mut *const ::core::ffi::c_void) -> OrtStatusPtr {
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
	indices: *mut *const ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn HasValue(value: *const OrtValue, out: *mut ::core::ffi::c_int) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetGPUComputeStream(context: *const OrtKernelContext, out: *mut *mut ::core::ffi::c_void) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorMemoryInfo(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetExecutionProviderApi(
	provider_name: *const ::core::ffi::c_char,
	version: u32,
	provider_api: *mut *const ::core::ffi::c_void
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
	ort_custom_thread_creation_options: *mut ::core::ffi::c_void
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
	ort_custom_thread_creation_options: *mut ::core::ffi::c_void
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
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCUDAProviderOptionsAsString(
	cuda_options: *const OrtCUDAProviderOptionsV2,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::core::ffi::c_char
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
	initializer_names: *const *const ::core::ffi::c_char,
	initializers: *const *const OrtValue,
	initializers_num: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn CreateOpAttr(
	name: *const ::core::ffi::c_char,
	data: *const ::core::ffi::c_void,
	len: ::core::ffi::c_int,
	type_: OrtOpAttrType,
	op_attr: *mut *mut OrtOpAttr
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseOpAttr(input: *mut OrtOpAttr) {}

unsafe extern "system" fn CreateOp(
	info: *const OrtKernelInfo,
	op_name: *const ::core::ffi::c_char,
	domain: *const ::core::ffi::c_char,
	version: ::core::ffi::c_int,
	type_constraint_names: *mut *const ::core::ffi::c_char,
	type_constraint_values: *const ONNXTensorElementDataType,
	type_constraint_count: ::core::ffi::c_int,
	attr_values: *const *const OrtOpAttr,
	attr_count: ::core::ffi::c_int,
	input_count: ::core::ffi::c_int,
	output_count: ::core::ffi::c_int,
	ort_op: *mut *mut OrtOp
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn InvokeOp(
	context: *const OrtKernelContext,
	ort_op: *const OrtOp,
	input_values: *const *const OrtValue,
	input_count: ::core::ffi::c_int,
	output_values: *const *mut OrtValue,
	output_count: ::core::ffi::c_int
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseOp(input: *mut OrtOp) {}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider(
	options: *mut OrtSessionOptions,
	provider_name: *const ::core::ffi::c_char,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
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
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCANNProviderOptionsAsString(
	cann_options: *const OrtCANNProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseCANNProviderOptions(input: *mut OrtCANNProviderOptions) {}

unsafe extern "system" fn MemoryInfoGetDeviceType(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType) {
	unsafe { *out = OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU };
}

unsafe extern "system" fn UpdateEnvWithCustomLogLevel(ort_env: *mut OrtEnv, log_severity_level: OrtLoggingLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetGlobalIntraOpThreadAffinity(tp_options: *mut OrtThreadingOptions, affinity_string: *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterCustomOpsLibrary_V2(options: *mut OrtSessionOptions, library_name: *const os_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn RegisterCustomOpsUsingFunction(options: *mut OrtSessionOptions, registration_func_name: *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetInputCount(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetOutputCount(info: *const OrtKernelInfo, out: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetInputName(info: *const OrtKernelInfo, index: usize, out: *mut ::core::ffi::c_char, size: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfo_GetOutputName(info: *const OrtKernelInfo, index: usize, out: *mut ::core::ffi::c_char, size: *mut usize) -> OrtStatusPtr {
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
	name: *const ::core::ffi::c_char,
	allocator: *mut OrtAllocator,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn HasSessionConfigEntry(
	options: *const OrtSessionOptions,
	config_key: *const ::core::ffi::c_char,
	out: *mut ::core::ffi::c_int
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetSessionConfigEntry(
	options: *const OrtSessionOptions,
	config_key: *const ::core::ffi::c_char,
	config_value: *mut ::core::ffi::c_char,
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
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetDnnlProviderOptionsAsString(
	dnnl_options: *const OrtDnnlProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseDnnlProviderOptions(input: *mut OrtDnnlProviderOptions) {}

unsafe extern "system" fn KernelInfo_GetNodeName(info: *const OrtKernelInfo, out: *mut ::core::ffi::c_char, size: *mut usize) -> OrtStatusPtr {
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
	message: *const ::core::ffi::c_char,
	file_path: *const os_char,
	line_number: ::core::ffi::c_int,
	func_name: *const ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn Logger_GetLoggingSeverityLevel(logger: *const OrtLogger, out: *mut OrtLoggingLevel) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelInfoGetConstantInput_tensor(
	info: *const OrtKernelInfo,
	index: usize,
	is_constant: *mut ::core::ffi::c_int,
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
	buffer: *mut *mut ::core::ffi::c_char
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

unsafe extern "system" fn GetBuildInfoString() -> *const ::core::ffi::c_char {
	c"ORT Build Info: ort-sys stub".as_ptr().cast()
}

unsafe extern "system" fn CreateROCMProviderOptions(out: *mut *mut OrtROCMProviderOptions) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateROCMProviderOptions(
	rocm_options: *mut OrtROCMProviderOptions,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetROCMProviderOptionsAsString(
	rocm_options: *const OrtROCMProviderOptions,
	allocator: *mut OrtAllocator,
	ptr: *mut *mut ::core::ffi::c_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReleaseROCMProviderOptions(input: *mut OrtROCMProviderOptions) {}

unsafe extern "system" fn CreateAndRegisterAllocatorV2(
	env: *mut OrtEnv,
	provider_type: *const ::core::ffi::c_char,
	mem_info: *const OrtMemoryInfo,
	arena_cfg: *const OrtArenaCfg,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "system" fn RunAsync(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: *const *const ::core::ffi::c_char,
	input: *const *const OrtValue,
	input_len: usize,
	output_names: *const *const ::core::ffi::c_char,
	output_names_len: usize,
	output: *mut *mut OrtValue,
	run_async_callback: RunAsyncCallbackFn,
	user_data: *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(target_arch = "wasm32")]
unsafe fn RunAsync(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: &[&str],
	inputs: &[*const OrtValue],
	output_names: &[&str],
	outputs: &mut [*mut OrtValue]
) -> core::pin::Pin<alloc::boxed::Box<dyn core::future::Future<Output = OrtStatusPtr>>> {
	Box::pin(async { Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented") })
}

unsafe extern "system" fn UpdateTensorRTProviderOptionsWithValue(
	tensorrt_options: *mut OrtTensorRTProviderOptionsV2,
	key: *const ::core::ffi::c_char,
	value: *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetTensorRTProviderOptionsByName(
	tensorrt_options: *const OrtTensorRTProviderOptionsV2,
	key: *const ::core::ffi::c_char,
	ptr: *mut *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn UpdateCUDAProviderOptionsWithValue(
	cuda_options: *mut OrtCUDAProviderOptionsV2,
	key: *const ::core::ffi::c_char,
	value: *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn GetCUDAProviderOptionsByName(
	cuda_options: *const OrtCUDAProviderOptionsV2,
	key: *const ::core::ffi::c_char,
	ptr: *mut *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn KernelContext_GetResource(
	context: *const OrtKernelContext,
	resouce_version: ::core::ffi::c_int,
	resource_id: ::core::ffi::c_int,
	resource: *mut *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SetUserLoggingFunction(
	options: *mut OrtSessionOptions,
	user_logging_function: OrtLoggingFunction,
	user_logging_param: *mut ::core::ffi::c_void
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
	attr_name: *const ::core::ffi::c_char,
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
	dim_params: *mut *const ::core::ffi::c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn ReadOpAttr(
	op_attr: *const OrtOpAttr,
	type_: OrtOpAttrType,
	data: *mut ::core::ffi::c_void,
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
	fn_: unsafe extern "system" fn(arg1: *mut ::core::ffi::c_void, arg2: usize),
	total: usize,
	num_batch: usize,
	usr_data: *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

unsafe extern "system" fn SessionOptionsAppendExecutionProvider_OpenVINO_V2(
	options: *mut OrtSessionOptions,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-18")]
unsafe extern "system" fn SessionOptionsAppendExecutionProvider_VitisAI(
	options: *mut OrtSessionOptions,
	provider_options_keys: *const *const ::core::ffi::c_char,
	provider_options_values: *const *const ::core::ffi::c_char,
	num_keys: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-18")]
unsafe extern "system" fn KernelContext_GetScratchBuffer(
	context: *const OrtKernelContext,
	mem_info: *const OrtMemoryInfo,
	count_or_bytes: usize,
	out: *mut *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-18")]
unsafe extern "system" fn KernelInfoGetAllocator(info: *const OrtKernelInfo, mem_type: OrtMemType, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-18")]
unsafe extern "system" fn AddExternalInitializersFromMemory(
	options: *mut OrtSessionOptions,
	external_initializer_file_names: *const *const os_char,
	external_initializer_file_buffer_array: *const *mut ::core::ffi::c_char,
	external_initializer_file_lengths: *const usize,
	num_external_initializer_files: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-20")]
unsafe extern "system" fn CreateLoraAdapter(adapter_file_path: *const os_char, allocator: *mut OrtAllocator, out: *mut *mut OrtLoraAdapter) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-20")]
unsafe extern "system" fn CreateLoraAdapterFromArray(
	bytes: *const ::core::ffi::c_void,
	num_bytes: usize,
	allocator: *mut OrtAllocator,
	out: *mut *mut OrtLoraAdapter
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-20")]
unsafe extern "system" fn ReleaseLoraAdapter(input: *mut OrtLoraAdapter) {}

#[cfg(feature = "api-20")]
unsafe extern "system" fn RunOptionsAddActiveLoraAdapter(options: *mut OrtRunOptions, adapter: *const OrtLoraAdapter) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-20")]
unsafe extern "system" fn SetEpDynamicOptions(
	sess: *mut OrtSession,
	keys: *const *const ::core::ffi::c_char,
	values: *const *const ::core::ffi::c_char,
	kv_len: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn ReleaseValueInfo(input: *mut OrtValueInfo) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn ReleaseNode(input: *mut OrtNode) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn ReleaseGraph(input: *mut OrtGraph) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn ReleaseModel(input: *mut OrtModel) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetValueInfoName(value_info: *const OrtValueInfo, name: *mut *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetValueInfoTypeInfo(value_info: *const OrtValueInfo, type_info: *mut *const OrtTypeInfo) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetModelEditorApi() -> *const OrtModelEditorApi {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn CreateTensorWithDataAndDeleterAsOrtValue(
	deleter: *mut OrtAllocator,
	p_data: *mut ::core::ffi::c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	r#type: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn SessionOptionsSetLoadCancellationFlag(options: *mut OrtSessionOptions, cancel: bool) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetCompileApi() -> *const OrtCompileApi {
	ptr::null_mut()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn CreateKeyValuePairs(out: *mut *mut OrtKeyValuePairs) {
	unsafe { *out = ptr::null_mut() };
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn AddKeyValuePair(kvps: *mut OrtKeyValuePairs, key: *const ::core::ffi::c_char, value: *const ::core::ffi::c_char) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetKeyValue(kvps: *const OrtKeyValuePairs, key: *const ::core::ffi::c_char) -> *const ::core::ffi::c_char {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetKeyValuePairs(
	kvps: *const OrtKeyValuePairs,
	keys: *mut *const *const ::core::ffi::c_char,
	values: *mut *const *const ::core::ffi::c_char,
	num_entries: *mut usize
) {
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn RemoveKeyValuePair(kvps: *mut OrtKeyValuePairs, key: *const ::core::ffi::c_char) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn ReleaseKeyValuePairs(input: *mut OrtKeyValuePairs) {}

#[cfg(feature = "api-22")]
unsafe extern "system" fn RegisterExecutionProviderLibrary(
	env: *mut OrtEnv,
	registration_name: *const ::core::ffi::c_char,
	path: *const os_char
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn UnregisterExecutionProviderLibrary(env: *mut OrtEnv, registration_name: *const ::core::ffi::c_char) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetEpDevices(env: *const OrtEnv, ep_devices: *mut *const *const OrtEpDevice, num_ep_devices: *mut usize) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn SessionOptionsAppendExecutionProvider_V2(
	session_options: *mut OrtSessionOptions,
	env: *mut OrtEnv,
	ep_devices: *const *const OrtEpDevice,
	num_ep_devices: usize,
	ep_option_keys: *const *const ::core::ffi::c_char,
	ep_option_vals: *const *const ::core::ffi::c_char,
	num_ep_options: usize
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn SessionOptionsSetEpSelectionPolicy(
	session_options: *mut OrtSessionOptions,
	policy: OrtExecutionProviderDevicePolicy
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn SessionOptionsSetEpSelectionPolicyDelegate(
	session_options: *mut OrtSessionOptions,
	delegate: EpSelectionDelegate,
	delegate_state: *mut ::core::ffi::c_void
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "Unimplemented")
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn HardwareDevice_Type(device: *const OrtHardwareDevice) -> OrtHardwareDeviceType {
	OrtHardwareDeviceType::OrtHardwareDeviceType_CPU
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn HardwareDevice_VendorId(device: *const OrtHardwareDevice) -> u32 {
	0
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn HardwareDevice_Vendor(device: *const OrtHardwareDevice) -> *const ::core::ffi::c_char {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn HardwareDevice_DeviceId(device: *const OrtHardwareDevice) -> u32 {
	0
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn HardwareDevice_Metadata(device: *const OrtHardwareDevice) -> *const OrtKeyValuePairs {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn EpDevice_EpName(ep_device: *const OrtEpDevice) -> *const ::core::ffi::c_char {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn EpDevice_EpVendor(ep_device: *const OrtEpDevice) -> *const ::core::ffi::c_char {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn EpDevice_EpMetadata(ep_device: *const OrtEpDevice) -> *const OrtKeyValuePairs {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn EpDevice_EpOptions(ep_device: *const OrtEpDevice) -> *const OrtKeyValuePairs {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn EpDevice_Device(ep_device: *const OrtEpDevice) -> *const OrtHardwareDevice {
	ptr::null()
}

#[cfg(feature = "api-22")]
unsafe extern "system" fn GetEpApi() -> *const OrtEpApi {
	ptr::null()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetTensorSizeInBytes(ort_value: *const OrtValue, size: *mut usize) -> OrtStatusPtr {
	unsafe { *size = 0 };
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn AllocatorGetStats(ort_allocator: *const OrtAllocator, out: *mut *mut OrtKeyValuePairs) -> OrtStatusPtr {
	unsafe { *out = ptr::null_mut() };
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn CreateMemoryInfo_V2(
	name: *const c_char,
	device_type: OrtMemoryInfoDeviceType,
	vendor_id: u32,
	device_id: i32,
	mem_type: OrtDeviceMemoryType,
	alignment: usize,
	allocator_type: OrtAllocatorType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	unsafe { *out = ptr::null_mut() };
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn MemoryInfoGetDeviceMemType(ptr: *const OrtMemoryInfo) -> OrtDeviceMemoryType {
	OrtDeviceMemoryType::OrtDeviceMemoryType_HOST_ACCESSIBLE
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn MemoryInfoGetVendorId(ptr: *const OrtMemoryInfo) -> u32 {
	0
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_GetValueProducer(
	value_info: *const OrtValueInfo,
	producer_node: *mut *const OrtNode,
	producer_output_index: *mut usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_GetValueNumConsumers(value_info: *const OrtValueInfo, num_consumers: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_GetValueConsumers(
	value_info: *const OrtValueInfo,
	nodes: *mut *const OrtNode,
	input_indices: *mut i64,
	num_consumers: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_GetInitializerValue(value_info: *const OrtValueInfo, initializer_value: *mut *const OrtValue) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_GetExternalInitializerInfo(value_info: *const OrtValueInfo, info: *mut *mut OrtExternalInitializerInfo) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_IsRequiredGraphInput(value_info: *const OrtValueInfo, is_required_graph_input: *mut bool) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_IsOptionalGraphInput(value_info: *const OrtValueInfo, is_optional_graph_input: *mut bool) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_IsGraphOutput(value_info: *const OrtValueInfo, is_graph_output: *mut bool) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_IsConstantInitializer(value_info: *const OrtValueInfo, is_constant_initializer: *mut bool) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ValueInfo_IsFromOuterScope(value_info: *const OrtValueInfo, is_from_outer_scope: *mut bool) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetName(graph: *const OrtGraph, graph_name: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetModelPath(graph: *const OrtGraph, model_path: *mut *const os_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetOnnxIRVersion(graph: *const OrtGraph, onnx_ir_version: *mut i64) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNumOperatorSets(graph: *const OrtGraph, num_operator_sets: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetOperatorSets(
	graph: *const OrtGraph,
	domains: *mut *const c_char,
	opset_versions: *mut i64,
	num_operator_sets: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNumInputs(graph: *const OrtGraph, num_inputs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetInputs(graph: *const OrtGraph, inputs: *mut *const OrtValueInfo, num_inputs: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNumOutputs(graph: *const OrtGraph, num_outputs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetOutputs(graph: *const OrtGraph, outputs: *mut *const OrtValueInfo, num_outputs: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNumInitializers(graph: *const OrtGraph, num_initializers: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetInitializers(graph: *const OrtGraph, initializers: *mut *const OrtValueInfo, num_initializers: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNumNodes(graph: *const OrtGraph, num_nodes: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetNodes(graph: *const OrtGraph, nodes: *mut *const OrtNode, num_nodes: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetParentNode(graph: *const OrtGraph, node: *mut *const OrtNode) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetGraphView(
	src_graph: *const OrtGraph,
	nodes: *mut *const OrtNode,
	num_nodes: usize,
	dst_graph: *mut *mut OrtGraph
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetId(node: *const OrtNode, node_id: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetName(node: *const OrtNode, node_name: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetOperatorType(node: *const OrtNode, operator_type: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetDomain(node: *const OrtNode, domain_name: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetSinceVersion(node: *const OrtNode, since_version: *mut i32) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetNumInputs(node: *const OrtNode, num_inputs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetInputs(node: *const OrtNode, inputs: *mut *const OrtValueInfo, num_inputs: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetNumOutputs(node: *const OrtNode, num_outputs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetOutputs(node: *const OrtNode, outputs: *mut *const OrtValueInfo, num_outputs: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetNumImplicitInputs(node: *const OrtNode, num_implicit_inputs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetImplicitInputs(node: *const OrtNode, implicit_inputs: *mut *const OrtValueInfo, num_implicit_inputs: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetNumAttributes(node: *const OrtNode, num_attributes: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetAttributes(node: *const OrtNode, attributes: *mut *const OrtOpAttr, num_attributes: usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetAttributeByName(node: *const OrtNode, attribute_name: *const c_char, attribute: *mut *const OrtOpAttr) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn OpAttr_GetTensorAttributeAsOrtValue(attribute: *const OrtOpAttr, attr_tensor: *mut *mut OrtValue) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn OpAttr_GetType(attribute: *const OrtOpAttr, r#type: *mut OrtOpAttrType) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn OpAttr_GetName(attribute: *const OrtOpAttr, name: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetNumSubgraphs(node: *const OrtNode, num_subgraphs: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetSubgraphs(
	node: *const OrtNode,
	subgraphs: *mut *const OrtGraph,
	num_subgraphs: *mut usize,
	attribute_names: *mut *const c_char
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetGraph(node: *const OrtNode, graph: *mut *const OrtGraph) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Node_GetEpName(node: *const OrtNode, out: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ReleaseExternalInitializerInfo(value: *mut OrtExternalInitializerInfo) {}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ExternalInitializerInfo_GetFilePath(info: *const OrtExternalInitializerInfo) -> *const os_char {
	ptr::null()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ExternalInitializerInfo_GetFileOffset(info: *const OrtExternalInitializerInfo) -> i64 {
	0
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ExternalInitializerInfo_GetByteSize(info: *const OrtExternalInitializerInfo) -> usize {
	0
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetRunConfigEntry(options: *const OrtRunOptions, config_key: *const c_char) -> *const c_char {
	ptr::null()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn EpDevice_MemoryInfo(ep_device: *const OrtEpDevice, memory_type: OrtDeviceMemoryType) -> *const OrtMemoryInfo {
	ptr::null()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn CreateSharedAllocator(
	env: *mut OrtEnv,
	ep_device: *const OrtEpDevice,
	mem_type: OrtDeviceMemoryType,
	allocator_type: OrtAllocatorType,
	allocator_options: *const OrtKeyValuePairs,
	allocator: *mut *mut OrtAllocator
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetSharedAllocator(env: *mut OrtEnv, mem_info: *const OrtMemoryInfo, allocator: *mut *mut OrtAllocator) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ReleaseSharedAllocator(env: *mut OrtEnv, ep_device: *const OrtEpDevice, mem_type: OrtDeviceMemoryType) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetTensorData(value: *const OrtValue, out: *mut *const c_void) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetSessionOptionsConfigEntries(options: *const OrtSessionOptions, out: *mut *mut OrtKeyValuePairs) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn SessionGetMemoryInfoForInputs(
	session: *const OrtSession,
	inputs_memory_info: *mut *const OrtMemoryInfo,
	num_inputs: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn SessionGetMemoryInfoForOutputs(
	session: *const OrtSession,
	outputs_memory_info: *mut *const OrtMemoryInfo,
	num_outputs: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn SessionGetEpDeviceForInputs(
	session: *const OrtSession,
	inputs_ep_devices: *mut *const OrtEpDevice,
	num_inputs: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn CreateSyncStreamForEpDevice(
	ep_device: *const OrtEpDevice,
	stream_options: *const OrtKeyValuePairs,
	stream: *mut *mut OrtSyncStream
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn SyncStream_GetHandle(stream: *mut OrtSyncStream) -> *mut c_void {
	ptr::null_mut()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn ReleaseSyncStream(value: *mut OrtSyncStream) {}

#[cfg(feature = "api-23")]
unsafe extern "system" fn CopyTensors(
	env: *mut OrtEnv,
	src_tensors: *const *const OrtValue,
	dst_tensors: *mut *const OrtValue,
	stream: *mut OrtSyncStream,
	num_tensors: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn Graph_GetModelMetadata(graph: *mut OrtGraph, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn GetModelCompatibilityForEpDevices(
	ep_devices: *const *const OrtEpDevice,
	num_ep_devices: usize,
	compatibility_info: *const c_char,
	out_status: *mut OrtCompiledModelCompatibility
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-23")]
unsafe extern "system" fn CreateExternalInitializerInfo(
	filepath: *const os_char,
	file_offset: i64,
	byte_size: usize,
	out: *mut *mut OrtExternalInitializerInfo
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn TensorTypeAndShape_HasShape(info: *const OrtTensorTypeAndShapeInfo) -> bool {
	false
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn KernelInfo_GetConfigEntries(info: *const OrtKernelInfo, out: *mut *mut OrtKeyValuePairs) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn KernelInfo_GetOperatorDomain(info: *const OrtKernelInfo, out: *mut c_char, size: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn KernelInfo_GetOperatorType(info: *const OrtKernelInfo, out: *mut c_char, size: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetInteropApi() -> *const OrtInteropApi {
	ptr::null()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn SessionGetEpDeviceForOutputs(
	session: *const OrtSession,
	outputs_ep_devices: *mut *const OrtEpDevice,
	num_outputs: usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetNumHardwareDevices(env: *const OrtEnv, num_devices: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetHardwareDevices(env: *const OrtEnv, devices: *mut *const OrtHardwareDevice, num_devices: *mut usize) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetHardwareDeviceEpIncompatibilityDetails(
	env: *const OrtEnv,
	ep_name: *const c_char,
	hw: *const OrtHardwareDevice,
	details: *mut *mut OrtDeviceEpIncompatibilityDetails
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn DeviceEpIncompatibilityDetails_GetReasonsBitmask(
	details: *const OrtDeviceEpIncompatibilityDetails,
	reasons_bitmask: *mut u32
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn DeviceEpIncompatibilityDetails_GetNotes(
	details: *const OrtDeviceEpIncompatibilityDetails,
	notes: *mut *const c_char
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn DeviceEpIncompatibilityDetails_GetErrorCode(details: *const OrtDeviceEpIncompatibilityDetails, error_code: *mut i32) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn ReleaseDeviceEpIncompatibilityDetails(value: *mut OrtDeviceEpIncompatibilityDetails) {}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetCompatibilityInfoFromModel(
	model_path: *const os_char,
	ep_type: *const c_char,
	allocator: *mut OrtAllocator,
	compatibility_info: *mut *mut c_char
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetCompatibilityInfoFromModelBytes(
	model_data: *const c_void,
	model_data_length: usize,
	ep_type: *const c_char,
	allocator: *mut OrtAllocator,
	compatibility_info: *mut *mut c_char
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn CreateEnvWithOptions(options: *const OrtEnvCreationOptions, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn Session_GetEpGraphAssignmentInfo(
	session: *const OrtSession,
	ep_subgraphs: *mut *const *const OrtEpAssignedSubgraph,
	num_ep_subgraphs: *mut usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn EpAssignedSubgraph_GetEpName(ep_subgraph: *const OrtEpAssignedSubgraph, out: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn EpAssignedSubgraph_GetNodes(
	ep_subgraph: *const OrtEpAssignedSubgraph,
	ep_nodes: *mut *const *const OrtEpAssignedNode,
	num_ep_nodes: *mut usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn EpAssignedNode_GetName(ep_node: *const OrtEpAssignedNode, out: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn EpAssignedNode_GetDomain(ep_node: *const OrtEpAssignedNode, out: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn EpAssignedNode_GetOperatorType(ep_node: *const OrtEpAssignedNode, out: *mut *const c_char) -> OrtStatusPtr {
	OrtStatusPtr::default()
}

#[cfg(feature = "api-24")]
unsafe extern "system" fn RunOptionsSetSyncStream(options: *mut OrtRunOptions, sync_stream: *mut OrtSyncStream) {}

#[cfg(feature = "api-24")]
unsafe extern "system" fn GetTensorElementTypeAndShapeDataReference(
	value: *const OrtValue,
	elem_type: *mut ONNXTensorElementDataType,
	shape_data: *mut *const i64,
	shape_data_count: *mut usize
) -> OrtStatusPtr {
	OrtStatusPtr::default()
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
		#[cfg(feature = "api-18")]
		SessionOptionsAppendExecutionProvider_VitisAI,
		#[cfg(feature = "api-18")]
		KernelContext_GetScratchBuffer,
		#[cfg(feature = "api-18")]
		KernelInfoGetAllocator,
		#[cfg(feature = "api-18")]
		AddExternalInitializersFromMemory,
		#[cfg(feature = "api-20")]
		CreateLoraAdapter,
		#[cfg(feature = "api-20")]
		CreateLoraAdapterFromArray,
		#[cfg(feature = "api-20")]
		ReleaseLoraAdapter,
		#[cfg(feature = "api-20")]
		RunOptionsAddActiveLoraAdapter,
		#[cfg(feature = "api-20")]
		SetEpDynamicOptions,
		#[cfg(feature = "api-22")]
		ReleaseValueInfo,
		#[cfg(feature = "api-22")]
		ReleaseNode,
		#[cfg(feature = "api-22")]
		ReleaseGraph,
		#[cfg(feature = "api-22")]
		ReleaseModel,
		#[cfg(feature = "api-22")]
		GetValueInfoName,
		#[cfg(feature = "api-22")]
		GetValueInfoTypeInfo,
		#[cfg(feature = "api-22")]
		GetModelEditorApi,
		#[cfg(feature = "api-22")]
		CreateTensorWithDataAndDeleterAsOrtValue,
		#[cfg(feature = "api-22")]
		SessionOptionsSetLoadCancellationFlag,
		#[cfg(feature = "api-22")]
		GetCompileApi,
		#[cfg(feature = "api-22")]
		CreateKeyValuePairs,
		#[cfg(feature = "api-22")]
		AddKeyValuePair,
		#[cfg(feature = "api-22")]
		GetKeyValue,
		#[cfg(feature = "api-22")]
		GetKeyValuePairs,
		#[cfg(feature = "api-22")]
		RemoveKeyValuePair,
		#[cfg(feature = "api-22")]
		ReleaseKeyValuePairs,
		#[cfg(feature = "api-22")]
		RegisterExecutionProviderLibrary,
		#[cfg(feature = "api-22")]
		UnregisterExecutionProviderLibrary,
		#[cfg(feature = "api-22")]
		GetEpDevices,
		#[cfg(feature = "api-22")]
		SessionOptionsAppendExecutionProvider_V2,
		#[cfg(feature = "api-22")]
		SessionOptionsSetEpSelectionPolicy,
		#[cfg(feature = "api-22")]
		SessionOptionsSetEpSelectionPolicyDelegate,
		#[cfg(feature = "api-22")]
		HardwareDevice_Type,
		#[cfg(feature = "api-22")]
		HardwareDevice_VendorId,
		#[cfg(feature = "api-22")]
		HardwareDevice_Vendor,
		#[cfg(feature = "api-22")]
		HardwareDevice_DeviceId,
		#[cfg(feature = "api-22")]
		HardwareDevice_Metadata,
		#[cfg(feature = "api-22")]
		EpDevice_EpName,
		#[cfg(feature = "api-22")]
		EpDevice_EpVendor,
		#[cfg(feature = "api-22")]
		EpDevice_EpMetadata,
		#[cfg(feature = "api-22")]
		EpDevice_EpOptions,
		#[cfg(feature = "api-22")]
		EpDevice_Device,
		#[cfg(feature = "api-22")]
		GetEpApi,
		#[cfg(feature = "api-23")]
		GetTensorSizeInBytes,
		#[cfg(feature = "api-23")]
		AllocatorGetStats,
		#[cfg(feature = "api-23")]
		CreateMemoryInfo_V2,
		#[cfg(feature = "api-23")]
		MemoryInfoGetDeviceMemType,
		#[cfg(feature = "api-23")]
		MemoryInfoGetVendorId,
		#[cfg(feature = "api-23")]
		ValueInfo_GetValueProducer,
		#[cfg(feature = "api-23")]
		ValueInfo_GetValueNumConsumers,
		#[cfg(feature = "api-23")]
		ValueInfo_GetValueConsumers,
		#[cfg(feature = "api-23")]
		ValueInfo_GetInitializerValue,
		#[cfg(feature = "api-23")]
		ValueInfo_GetExternalInitializerInfo,
		#[cfg(feature = "api-23")]
		ValueInfo_IsRequiredGraphInput,
		#[cfg(feature = "api-23")]
		ValueInfo_IsOptionalGraphInput,
		#[cfg(feature = "api-23")]
		ValueInfo_IsGraphOutput,
		#[cfg(feature = "api-23")]
		ValueInfo_IsConstantInitializer,
		#[cfg(feature = "api-23")]
		ValueInfo_IsFromOuterScope,
		#[cfg(feature = "api-23")]
		Graph_GetName,
		#[cfg(feature = "api-23")]
		Graph_GetModelPath,
		#[cfg(feature = "api-23")]
		Graph_GetOnnxIRVersion,
		#[cfg(feature = "api-23")]
		Graph_GetNumOperatorSets,
		#[cfg(feature = "api-23")]
		Graph_GetOperatorSets,
		#[cfg(feature = "api-23")]
		Graph_GetNumInputs,
		#[cfg(feature = "api-23")]
		Graph_GetInputs,
		#[cfg(feature = "api-23")]
		Graph_GetNumOutputs,
		#[cfg(feature = "api-23")]
		Graph_GetOutputs,
		#[cfg(feature = "api-23")]
		Graph_GetNumInitializers,
		#[cfg(feature = "api-23")]
		Graph_GetInitializers,
		#[cfg(feature = "api-23")]
		Graph_GetNumNodes,
		#[cfg(feature = "api-23")]
		Graph_GetNodes,
		#[cfg(feature = "api-23")]
		Graph_GetParentNode,
		#[cfg(feature = "api-23")]
		Graph_GetGraphView,
		#[cfg(feature = "api-23")]
		Node_GetId,
		#[cfg(feature = "api-23")]
		Node_GetName,
		#[cfg(feature = "api-23")]
		Node_GetOperatorType,
		#[cfg(feature = "api-23")]
		Node_GetDomain,
		#[cfg(feature = "api-23")]
		Node_GetSinceVersion,
		#[cfg(feature = "api-23")]
		Node_GetNumInputs,
		#[cfg(feature = "api-23")]
		Node_GetInputs,
		#[cfg(feature = "api-23")]
		Node_GetNumOutputs,
		#[cfg(feature = "api-23")]
		Node_GetOutputs,
		#[cfg(feature = "api-23")]
		Node_GetNumImplicitInputs,
		#[cfg(feature = "api-23")]
		Node_GetImplicitInputs,
		#[cfg(feature = "api-23")]
		Node_GetNumAttributes,
		#[cfg(feature = "api-23")]
		Node_GetAttributes,
		#[cfg(feature = "api-23")]
		Node_GetAttributeByName,
		#[cfg(feature = "api-23")]
		OpAttr_GetTensorAttributeAsOrtValue,
		#[cfg(feature = "api-23")]
		OpAttr_GetType,
		#[cfg(feature = "api-23")]
		OpAttr_GetName,
		#[cfg(feature = "api-23")]
		Node_GetNumSubgraphs,
		#[cfg(feature = "api-23")]
		Node_GetSubgraphs,
		#[cfg(feature = "api-23")]
		Node_GetGraph,
		#[cfg(feature = "api-23")]
		Node_GetEpName,
		#[cfg(feature = "api-23")]
		ReleaseExternalInitializerInfo,
		#[cfg(feature = "api-23")]
		ExternalInitializerInfo_GetFilePath,
		#[cfg(feature = "api-23")]
		ExternalInitializerInfo_GetFileOffset,
		#[cfg(feature = "api-23")]
		ExternalInitializerInfo_GetByteSize,
		#[cfg(feature = "api-23")]
		GetRunConfigEntry,
		#[cfg(feature = "api-23")]
		EpDevice_MemoryInfo,
		#[cfg(feature = "api-23")]
		CreateSharedAllocator,
		#[cfg(feature = "api-23")]
		GetSharedAllocator,
		#[cfg(feature = "api-23")]
		ReleaseSharedAllocator,
		#[cfg(feature = "api-23")]
		GetTensorData,
		#[cfg(feature = "api-23")]
		GetSessionOptionsConfigEntries,
		#[cfg(feature = "api-23")]
		SessionGetMemoryInfoForInputs,
		#[cfg(feature = "api-23")]
		SessionGetMemoryInfoForOutputs,
		#[cfg(feature = "api-23")]
		SessionGetEpDeviceForInputs,
		#[cfg(feature = "api-23")]
		CreateSyncStreamForEpDevice,
		#[cfg(feature = "api-23")]
		SyncStream_GetHandle,
		#[cfg(feature = "api-23")]
		ReleaseSyncStream,
		#[cfg(feature = "api-23")]
		CopyTensors,
		#[cfg(feature = "api-23")]
		Graph_GetModelMetadata,
		#[cfg(feature = "api-23")]
		GetModelCompatibilityForEpDevices,
		#[cfg(feature = "api-23")]
		CreateExternalInitializerInfo,
		#[cfg(feature = "api-24")]
		TensorTypeAndShape_HasShape,
		#[cfg(feature = "api-24")]
		KernelInfo_GetConfigEntries,
		#[cfg(feature = "api-24")]
		KernelInfo_GetOperatorDomain,
		#[cfg(feature = "api-24")]
		KernelInfo_GetOperatorType,
		#[cfg(feature = "api-24")]
		GetInteropApi,
		#[cfg(feature = "api-24")]
		SessionGetEpDeviceForOutputs,
		#[cfg(feature = "api-24")]
		GetNumHardwareDevices,
		#[cfg(feature = "api-24")]
		GetHardwareDevices,
		#[cfg(feature = "api-24")]
		GetHardwareDeviceEpIncompatibilityDetails,
		#[cfg(feature = "api-24")]
		DeviceEpIncompatibilityDetails_GetReasonsBitmask,
		#[cfg(feature = "api-24")]
		DeviceEpIncompatibilityDetails_GetNotes,
		#[cfg(feature = "api-24")]
		DeviceEpIncompatibilityDetails_GetErrorCode,
		#[cfg(feature = "api-24")]
		ReleaseDeviceEpIncompatibilityDetails,
		#[cfg(feature = "api-24")]
		GetCompatibilityInfoFromModel,
		#[cfg(feature = "api-24")]
		GetCompatibilityInfoFromModelBytes,
		#[cfg(feature = "api-24")]
		CreateEnvWithOptions,
		#[cfg(feature = "api-24")]
		Session_GetEpGraphAssignmentInfo,
		#[cfg(feature = "api-24")]
		EpAssignedSubgraph_GetEpName,
		#[cfg(feature = "api-24")]
		EpAssignedSubgraph_GetNodes,
		#[cfg(feature = "api-24")]
		EpAssignedNode_GetName,
		#[cfg(feature = "api-24")]
		EpAssignedNode_GetDomain,
		#[cfg(feature = "api-24")]
		EpAssignedNode_GetOperatorType,
		#[cfg(feature = "api-24")]
		RunOptionsSetSyncStream,
		#[cfg(feature = "api-24")]
		GetTensorElementTypeAndShapeDataReference
	}
}
