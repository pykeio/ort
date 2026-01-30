use candle_core::DType;
use ort_sys::OrtErrorCode;

mod api;
mod memory;
mod session;
mod tensor;

pub(crate) use ort_sys::stub::Error;

pub use self::api::api;

pub(crate) struct Environment {}

impl Environment {
	pub fn new_sys() -> *mut ort_sys::OrtEnv {
		(Box::leak(Box::new(Self {})) as *mut Environment).cast()
	}

	pub unsafe fn cast_from_sys<'e>(ptr: *const ort_sys::OrtEnv) -> &'e Environment {
		unsafe { &*ptr.cast::<Environment>() }
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtEnv) -> Box<Environment> {
		Box::from_raw(ptr.cast::<Environment>())
	}
}

fn convert_sys_to_dtype(sys: ort_sys::ONNXTensorElementDataType) -> Result<DType, Error> {
	match sys {
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Ok(DType::U8),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Ok(DType::U32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Ok(DType::I16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Ok(DType::I32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Ok(DType::I64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => Ok(DType::BF16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Ok(DType::F16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Ok(DType::F32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Ok(DType::F64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => Ok(DType::F8E4M3),
		_ => Err(Error::new(OrtErrorCode::ORT_FAIL, "Element type not supported by candle"))
	}
}

fn convert_dtype_to_sys(dtype: DType) -> Result<ort_sys::ONNXTensorElementDataType, Error> {
	match dtype {
		DType::U8 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8),
		DType::U32 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32),
		DType::I16 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16),
		DType::I32 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
		DType::I64 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
		DType::BF16 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16),
		DType::F16 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
		DType::F32 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
		DType::F64 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE),
		DType::F8E4M3 => Ok(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN),
		_ => Err(Error::new(OrtErrorCode::ORT_FAIL, "Element type not supported by ONNX Runtime"))
	}
}
