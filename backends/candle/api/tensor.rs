use std::ptr::NonNull;

use ort_sys::{stub::Error, *};

use crate::{
	memory::{Allocator, DEFAULT_CPU_MEMORY_INFO, MemoryInfo},
	tensor::{Tensor, TypeInfo},
	util::{element_size, num_elements, read_vec, trys}
};

pub(super) unsafe extern "system" fn CreateTensorAsOrtValue(
	allocator: *mut OrtAllocator,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let memory_info = if allocator.is_null() {
		&DEFAULT_CPU_MEMORY_INFO
	} else {
		&unsafe { &*allocator.cast::<Allocator>() }.memory_info
	};
	let tensor = trys!(Tensor::new(type_, unsafe { read_vec(shape, shape_len) }, memory_info));
	unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateTensorWithDataAsOrtValue(
	info: *const OrtMemoryInfo,
	p_data: *mut c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	if !info.is_null() && !unsafe { &*info.cast::<MemoryInfo>() }.is_cpu() {
		return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "ort-candle can only create tensors from CPU buffers");
	}
	let tensor = trys!(Tensor::new_borrowed(type_, unsafe { read_vec(shape, shape_len) }, p_data.cast(), p_data_len, None));
	unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateTensorWithDataAndDeleterAsOrtValue(
	deleter: *mut OrtAllocator,
	p_data: *mut c_void,
	p_data_len: usize,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let tensor = trys!(Tensor::new_borrowed(type_, unsafe { read_vec(shape, shape_len) }, p_data.cast(), p_data_len, NonNull::new(deleter)));
	unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn IsTensor(_value: *const OrtValue, out: *mut c_int) -> OrtStatusPtr {
	unsafe { out.write(1) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetValueType(_value: *const OrtValue, out: *mut ONNXType) -> OrtStatusPtr {
	unsafe { out.write(ONNXType::ONNX_TYPE_TENSOR) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn HasValue(_value: *const OrtValue, out: *mut c_int) -> OrtStatusPtr {
	unsafe { out.write(1) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorMemoryInfo(value: *const OrtValue, mem_info: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { mem_info.write((&tensor.memory_info as *const MemoryInfo).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTypeInfo(value: *const OrtValue, out: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(TypeInfo::new(tensor.dtype, tensor.shape.clone()).into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorTypeAndShape(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(TypeInfo::new(tensor.dtype, tensor.shape.clone()).into_sys().cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorElementTypeAndShapeDataReference(
	value: *const OrtValue,
	elem_type: *mut ONNXTensorElementDataType,
	shape_data: *mut *const i64,
	shape_data_count: *mut usize
) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe {
		elem_type.write(tensor.dtype);
		shape_data.write(tensor.shape.as_ptr());
		shape_data_count.write(tensor.shape.len());
	};
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorMutableData(value: *mut OrtValue, out: *mut *mut c_void) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(trys!(tensor.data_ptr()).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorData(value: *const OrtValue, out: *mut *const c_void) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(trys!(tensor.data_ptr()).cast_const().cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorSizeInBytes(value: *const OrtValue, size: *mut usize) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	let byte_len = trys!(num_elements(&tensor.shape).and_then(|x| {
		x.checked_mul(element_size(tensor.dtype))
			.ok_or_else(|| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "tensor is too large"))
	}));
	unsafe { size.write(byte_len) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn TensorAt(
	value: *mut OrtValue,
	location_values: *const i64,
	location_values_count: usize,
	out: *mut *mut c_void
) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	if location_values_count != tensor.shape.len() {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "location dimensions do not match shape size");
	}
	let location = unsafe { read_vec(location_values, location_values_count) };
	let mut offset = 0usize;
	let mut stride = 1usize;
	for (&index, &dim) in location.iter().zip(&tensor.shape).rev() {
		if index < 0 || index >= dim {
			return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "invalid location range");
		}
		offset += index as usize * stride;
		stride *= dim as usize;
	}
	unsafe { out.write(trys!(tensor.data_ptr()).add(offset * element_size(tensor.dtype)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CopyTensors(
	_env: *mut OrtEnv,
	src_tensors: *const *const OrtValue,
	dst_tensors: *mut *const OrtValue,
	_stream: *mut OrtSyncStream,
	num_tensors: usize
) -> OrtStatusPtr {
	for i in 0..num_tensors {
		let (src, dst) = unsafe { (*src_tensors.add(i), *dst_tensors.add(i)) };
		if src == dst {
			continue;
		}
		let src = unsafe { &*src.cast::<Tensor>() };
		let dst = unsafe { &mut *dst.cast_mut().cast::<Tensor>() };
		let src = trys!(src.to_candle());
		trys!(dst.copy_from(&src));
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseValue(input: *mut OrtValue) {
	drop(unsafe { Box::<Tensor>::from_raw(input.cast()) });
}
