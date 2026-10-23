use core::ffi::CStr;

use ort_sys::*;

use crate::{
	tensor::TypeInfo,
	util::{num_elements_i64, read_vec}
};

pub(super) unsafe extern "system" fn CastTypeInfoToTensorInfo(type_info: *const OrtTypeInfo, out: *mut *const OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	unsafe { out.write(type_info.cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetOnnxTypeFromTypeInfo(_type_info: *const OrtTypeInfo, out: *mut ONNXType) -> OrtStatusPtr {
	unsafe { out.write(ONNXType::ONNX_TYPE_TENSOR) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateTensorTypeAndShapeInfo(out: *mut *mut OrtTensorTypeAndShapeInfo) -> OrtStatusPtr {
	unsafe {
		out.write(
			TypeInfo::new(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, Vec::new())
				.into_sys()
				.cast()
		)
	};
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetTensorElementType(info: *mut OrtTensorTypeAndShapeInfo, type_: ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	info.dtype = type_;
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetDimensions(info: *mut OrtTensorTypeAndShapeInfo, dim_values: *const i64, dim_count: usize) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	info.shape = Some(unsafe { read_vec(dim_values, dim_count) });
	info.symbolic_dims.resize(dim_count, Default::default());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SetSymbolicDimensions(
	info: *mut OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	let info = unsafe { &mut *info.cast::<TypeInfo>() };
	info.symbolic_dims = (0..dim_params_length)
		.map(|i| unsafe { CStr::from_ptr(*dim_params.add(i)) }.to_owned())
		.collect();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorElementType(info: *const OrtTensorTypeAndShapeInfo, out: *mut ONNXTensorElementDataType) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	unsafe { out.write(info.dtype) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetDimensionsCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	unsafe { out.write(info.shape().len()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetDimensions(info: *const OrtTensorTypeAndShapeInfo, dim_values: *mut i64, dim_values_length: usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	for (i, dim) in info.shape().iter().enumerate().take(dim_values_length) {
		unsafe { dim_values.add(i).write(*dim) };
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetSymbolicDimensions(
	info: *const OrtTensorTypeAndShapeInfo,
	dim_params: *mut *const c_char,
	dim_params_length: usize
) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	for i in 0..dim_params_length {
		unsafe { dim_params.add(i).write(info.symbolic_dims.get(i).map_or(c"".as_ptr(), |s| s.as_ptr())) };
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetTensorShapeElementCount(info: *const OrtTensorTypeAndShapeInfo, out: *mut usize) -> OrtStatusPtr {
	let info = unsafe { &*info.cast::<TypeInfo>() };
	let Some(shape) = info.shape.as_ref() else {
		unsafe { out.write(0) };
		return OrtStatusPtr::default();
	};
	unsafe { out.write(num_elements_i64(shape)) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn TensorTypeAndShape_HasShape(info: *const OrtTensorTypeAndShapeInfo) -> bool {
	unsafe { &*info.cast::<TypeInfo>() }.shape.is_some()
}

pub(super) unsafe extern "system" fn ReleaseTypeInfo(input: *mut OrtTypeInfo) {
	drop(unsafe { TypeInfo::consume_sys(input) });
}

pub(super) unsafe extern "system" fn ReleaseTensorTypeAndShapeInfo(input: *mut OrtTensorTypeAndShapeInfo) {
	drop(unsafe { TypeInfo::consume_sys(input.cast()) });
}
