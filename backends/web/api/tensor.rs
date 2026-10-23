use core::ptr;

use ort_sys::{stub::Error, *};

use crate::{
	binding,
	memory::MemoryInfo,
	tensor::{Tensor, TypeInfo, dtype_to_onnx, onnx_to_dtype, tensor_byte_size},
	util::{num_elements_i64, read_vec, str_from_ptr, trys, value_to_string}
};

fn shape_to_dims(shape: &[i64]) -> Result<Vec<i32>, Error> {
	shape
		.iter()
		.map(|d| i32::try_from(*d).ok().filter(|d| *d >= 0))
		.collect::<Option<Vec<_>>>()
		.ok_or_else(|| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("invalid shape {shape:?}")))
}

pub(super) unsafe extern "system" fn CreateTensorAsOrtValue(
	_allocator: *mut OrtAllocator,
	shape: *const i64,
	shape_len: usize,
	type_: ONNXTensorElementDataType,
	out: *mut *mut OrtValue
) -> OrtStatusPtr {
	let dims = trys!(shape_to_dims(&unsafe { read_vec(shape, shape_len) }));
	let Some(dtype) = onnx_to_dtype(type_) else {
		return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("unsupported or invalid data type {type_:?}"));
	};

	match Tensor::new_owned(dtype, &dims) {
		Ok(tensor) => {
			unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_FAIL, value_to_string(&e))
	}
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
		return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, "ort-web can only create tensors from CPU buffers");
	}
	let dims = trys!(shape_to_dims(&unsafe { read_vec(shape, shape_len) }));
	let Some(dtype) = onnx_to_dtype(type_) else {
		return Error::new_sys(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("unsupported or invalid data type {type_:?}"));
	};

	match unsafe { Tensor::from_ptr(dtype, p_data, p_data_len, &dims) } {
		Ok(tensor) => {
			unsafe { out.write((Box::leak(Box::new(tensor)) as *mut Tensor).cast()) };
			OrtStatusPtr::default()
		}
		Err(e) => Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, value_to_string(&e))
	}
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
	unsafe { out.write(TypeInfo::from_tensor(tensor).into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorTypeAndShape(value: *const OrtValue, out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	unsafe { out.write(TypeInfo::from_tensor(tensor).into_sys().cast()) };
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
		elem_type.write(dtype_to_onnx(tensor.dtype()));
		shape_data.write(tensor.shape.as_ptr());
		shape_data_count.write(tensor.shape.len());
	};
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorMutableData(value: *mut OrtValue, out: *mut *mut c_void) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast::<Tensor>() };
	unsafe { out.write(trys!(tensor.data_ptr())) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorData(value: *const OrtValue, out: *mut *const c_void) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast_mut().cast::<Tensor>() };
	unsafe { out.write(trys!(tensor.data_ptr()).cast_const()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorSizeInBytes(value: *const OrtValue, size: *mut usize) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	match tensor_byte_size(tensor.dtype(), num_elements_i64(&tensor.shape)) {
		Some(byte_size) => {
			unsafe { size.write(byte_size) };
			OrtStatusPtr::default()
		}
		None => Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "GetTensorSizeInBytes does not support string tensors")
	}
}

pub(super) unsafe extern "system" fn TensorAt(
	value: *mut OrtValue,
	location_values: *const i64,
	location_values_count: usize,
	out: *mut *mut c_void
) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast::<Tensor>() };
	let element_size = match tensor.dtype() {
		binding::DataType::String => return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "TensorAt does not support string tensors"),
		binding::DataType::Int4 | binding::DataType::Uint4 => {
			return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "TensorAt does not support sub-byte packed types like int4");
		}
		dtype => tensor_byte_size(dtype, 1).unwrap_or(0)
	};
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
	unsafe { out.write(trys!(tensor.data_ptr()).cast::<u8>().add(offset * element_size).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn FillStringTensor(value: *mut OrtValue, s: *const *const c_char, s_len: usize) -> OrtStatusPtr {
	let tensor = unsafe { &mut *value.cast::<Tensor>() };
	let array = trys!(tensor.string_array());
	if array.length() as usize != s_len {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "input array doesn't equal tensor size");
	}
	for i in 0..s_len {
		let string = trys!(unsafe { str_from_ptr(*s.add(i)) });
		array.set(i as u32, js_sys::JsString::from(string).into());
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetStringTensorDataLength(value: *const OrtValue, len: *mut usize) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	let strings = trys!(tensor.strings());
	unsafe { len.write(strings.iter().map(|s| s.len()).sum()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetStringTensorContent(
	value: *const OrtValue,
	s: *mut c_void,
	s_len: usize,
	offsets: *mut usize,
	offsets_len: usize
) -> OrtStatusPtr {
	let tensor = unsafe { &*value.cast::<Tensor>() };
	let strings = trys!(tensor.strings());
	if offsets_len != strings.len() {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "offsets buffer is not equal to tensor size");
	}
	let total_len: usize = strings.iter().map(|s| s.len()).sum();
	if s_len < total_len {
		return Error::new_sys(OrtErrorCode::ORT_FAIL, "output buffer is too small");
	}
	let mut offset = 0;
	for (i, string) in strings.iter().enumerate() {
		unsafe {
			ptr::copy_nonoverlapping(string.as_ptr(), s.cast::<u8>().add(offset), string.len());
			offsets.add(i).write(offset);
		}
		offset += string.len();
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseValue(input: *mut OrtValue) {
	drop(unsafe { Box::<Tensor>::from_raw(input.cast()) });
}
