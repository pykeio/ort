use core::fmt::Display;
use std::{
	ffi::{CStr, c_char},
	slice
};

use candle_core::DType;
use ort_sys::{ONNXTensorElementDataType, OrtErrorCode, stub::Error};

macro_rules! trys {
	($e:expr) => {
		match $e {
			Ok(v) => v,
			Err(e) => return crate::Error::into_sys(e)
		}
	};
}
pub(crate) use trys;

pub(crate) fn candle_error(e: impl Display) -> Error {
	Error::new(OrtErrorCode::ORT_FAIL, e.to_string())
}

pub(crate) fn convert_sys_to_dtype(sys: ONNXTensorElementDataType) -> Result<DType, Error> {
	match sys {
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Ok(DType::U8),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Ok(DType::U8),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Ok(DType::U32),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Ok(DType::I16),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Ok(DType::I32),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Ok(DType::I64),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => Ok(DType::BF16),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Ok(DType::F16),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Ok(DType::F32),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Ok(DType::F64),
		ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => Ok(DType::F8E4M3),
		_ => Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("Element type {sys:?} not supported by candle")))
	}
}

pub(crate) fn convert_dtype_to_sys(dtype: DType) -> Result<ONNXTensorElementDataType, Error> {
	match dtype {
		DType::U8 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
		DType::U32 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
		DType::I16 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
		DType::I32 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
		DType::I64 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
		DType::BF16 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
		DType::F16 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
		DType::F32 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
		DType::F64 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
		DType::F8E4M3 => Ok(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN),
		_ => Err(Error::new(OrtErrorCode::ORT_FAIL, format!("Element type {dtype:?} not supported by ONNX Runtime")))
	}
}

pub(crate) fn element_size(dtype: ONNXTensorElementDataType) -> usize {
	convert_sys_to_dtype(dtype).map_or(0, |d| d.size_in_bytes())
}

pub(crate) fn num_elements(shape: &[i64]) -> Result<usize, Error> {
	let mut size = 1usize;
	for dim in shape {
		if *dim < 0 {
			return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid shape {shape:?}")));
		}
		size = size
			.checked_mul(*dim as usize)
			.ok_or_else(|| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Shape {shape:?} is too large")))?;
	}
	Ok(size)
}

pub(crate) fn convert_proto_to_sys(data_type: i32) -> Result<ONNXTensorElementDataType, Error> {
	let sys = match data_type {
		1 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		2 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
		3 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
		4 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
		5 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
		6 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
		7 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		8 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
		9 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
		10 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
		11 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
		12 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
		13 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
		16 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
		17 => ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
		_ => return Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("Element type {data_type} not supported by ONNX Runtime")))
	};
	Ok(sys)
}

pub(crate) unsafe fn str_from_ptr<'s>(ptr: *const c_char) -> Result<&'s str, Error> {
	if ptr.is_null() {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "string ptr should not be null"));
	}
	unsafe { CStr::from_ptr(ptr) }
		.to_str()
		.map_err(|_| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "string is not valid UTF-8"))
}

pub(crate) unsafe fn read_vec<T: Clone>(ptr: *const T, len: usize) -> Vec<T> {
	if len == 0 { Vec::new() } else { unsafe { slice::from_raw_parts(ptr, len) }.to_vec() }
}
