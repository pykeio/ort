use ort_sys::OrtErrorCode;
use tract_onnx::{Onnx, prelude::DatumType};

mod api;
pub(crate) mod error;
mod memory;
mod session;
mod tensor;

pub use self::api::api;
use self::error::Error;

pub(crate) struct Environment {
	pub onnx: Onnx
}

impl Environment {
	pub fn new_sys() -> *mut ort_sys::OrtEnv {
		(Box::leak(Box::new(Self { onnx: tract_onnx::onnx() })) as *mut Environment).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtEnv) -> Box<Environment> {
		Box::from_raw(ptr.cast::<Environment>())
	}
}

fn convert_sys_to_datum_type(sys: ort_sys::ONNXTensorElementDataType) -> Result<DatumType, Error> {
	match sys {
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Ok(DatumType::Bool),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Ok(DatumType::U8),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Ok(DatumType::U16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Ok(DatumType::U32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Ok(DatumType::U64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => Ok(DatumType::I8),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Ok(DatumType::I16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Ok(DatumType::I32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Ok(DatumType::I64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Ok(DatumType::F16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Ok(DatumType::F32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Ok(DatumType::F64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Ok(DatumType::String),
		_ => Err(Error::new(OrtErrorCode::ORT_FAIL, "Element type not supported by tract"))
	}
}

fn convert_datum_type_to_sys(dtype: DatumType) -> ort_sys::ONNXTensorElementDataType {
	match dtype {
		DatumType::Bool => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
		DatumType::U8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
		DatumType::U16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
		DatumType::U32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
		DatumType::U64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
		DatumType::I8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
		DatumType::I16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
		DatumType::I32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
		DatumType::I64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		DatumType::F16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
		DatumType::F32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		DatumType::F64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
		DatumType::String => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
		_ => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
	}
}
