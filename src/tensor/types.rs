#[cfg(feature = "ndarray")]
use std::ptr;

#[cfg(feature = "ndarray")]
use crate::{ortsys, Error, Result};

/// Enum mapping ONNX Runtime's supported tensor data types.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TensorElementType {
	/// 32-bit floating point number, equivalent to Rust's `f32`.
	Float32,
	/// Unsigned 8-bit integer, equivalent to Rust's `u8`.
	Uint8,
	/// Signed 8-bit integer, equivalent to Rust's `i8`.
	Int8,
	/// Unsigned 16-bit integer, equivalent to Rust's `u16`.
	Uint16,
	/// Signed 16-bit integer, equivalent to Rust's `i16`.
	Int16,
	/// Signed 32-bit integer, equivalent to Rust's `i32`.
	Int32,
	/// Signed 64-bit integer, equivalent to Rust's `i64`.
	Int64,
	/// String, equivalent to Rust's `String`.
	String,
	/// Boolean, equivalent to Rust's `bool`.
	Bool,
	/// 16-bit floating point number, equivalent to [`half::f16`] (requires the `half` feature).
	#[cfg(feature = "half")]
	#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
	Float16,
	/// 64-bit floating point number, equivalent to Rust's `f64`. Also known as `double`.
	Float64,
	/// Unsigned 32-bit integer, equivalent to Rust's `u32`.
	Uint32,
	/// Unsigned 64-bit integer, equivalent to Rust's `u64`.
	Uint64,
	/// Brain 16-bit floating point number, equivalent to [`half::bf16`] (requires the `half` feature).
	#[cfg(feature = "half")]
	#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
	Bfloat16
}

impl From<TensorElementType> for ort_sys::ONNXTensorElementDataType {
	fn from(val: TensorElementType) -> Self {
		match val {
			TensorElementType::Float32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			TensorElementType::Uint8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			TensorElementType::Int8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
			TensorElementType::Uint16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
			TensorElementType::Int16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
			TensorElementType::Int32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
			TensorElementType::Int64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
			TensorElementType::String => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
			TensorElementType::Bool => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
			#[cfg(feature = "half")]
			TensorElementType::Float16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
			TensorElementType::Float64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
			TensorElementType::Uint32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
			TensorElementType::Uint64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
			#[cfg(feature = "half")]
			TensorElementType::Bfloat16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
		}
	}
}
impl From<ort_sys::ONNXTensorElementDataType> for TensorElementType {
	fn from(val: ort_sys::ONNXTensorElementDataType) -> Self {
		match val {
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => TensorElementType::Float32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => TensorElementType::Uint8,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => TensorElementType::Int8,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => TensorElementType::Uint16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => TensorElementType::Int16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => TensorElementType::Int32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => TensorElementType::Int64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => TensorElementType::String,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => TensorElementType::Bool,
			#[cfg(feature = "half")]
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => TensorElementType::Float16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => TensorElementType::Float64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => TensorElementType::Uint32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => TensorElementType::Uint64,
			#[cfg(feature = "half")]
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => TensorElementType::Bfloat16,
			_ => panic!("Invalid ONNXTensorElementDataType value")
		}
	}
}

/// Trait used to map Rust types (for example `f32`) to ONNX tensor element data types (for example `Float`).
pub trait IntoTensorElementType {
	/// Returns the ONNX tensor element data type corresponding to the given Rust type.
	fn into_tensor_element_type() -> TensorElementType;

	crate::private_trait!();
}

pub trait PrimitiveTensorElementType: IntoTensorElementType {
	crate::private_trait!();
}

macro_rules! impl_type_trait {
	($type_:ty, $variant:ident) => {
		impl IntoTensorElementType for $type_ {
			fn into_tensor_element_type() -> TensorElementType {
				TensorElementType::$variant
			}

			crate::private_impl!();
		}

		impl PrimitiveTensorElementType for $type_ {
			crate::private_impl!();
		}
	};
}

impl_type_trait!(f32, Float32);
impl_type_trait!(u8, Uint8);
impl_type_trait!(i8, Int8);
impl_type_trait!(u16, Uint16);
impl_type_trait!(i16, Int16);
impl_type_trait!(i32, Int32);
impl_type_trait!(i64, Int64);
impl_type_trait!(bool, Bool);
#[cfg(feature = "half")]
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
impl_type_trait!(half::f16, Float16);
impl_type_trait!(f64, Float64);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
#[cfg(feature = "half")]
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
impl_type_trait!(half::bf16, Bfloat16);

impl IntoTensorElementType for String {
	fn into_tensor_element_type() -> TensorElementType {
		TensorElementType::String
	}

	crate::private_impl!();
}

/// Adapter for common Rust string types to ONNX strings.
pub trait Utf8Data {
	/// Returns the contents of this value as a slice of UTF-8 bytes.
	fn as_utf8_bytes(&self) -> &[u8];
}

impl Utf8Data for String {
	fn as_utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<'a> Utf8Data for &'a str {
	fn as_utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

/// Construct an [`ndarray::ArrayView`] for an ORT tensor.
///
/// Only to be used on types whose Rust in-memory representation matches ONNX Runtime's (e.g. primitive numeric types
/// like u32)
#[cfg(feature = "ndarray")]
pub(crate) fn extract_primitive_array<'t, T>(shape: ndarray::IxDyn, tensor: *mut ort_sys::OrtValue) -> Result<ndarray::ArrayViewD<'t, T>> {
	// Get pointer to output tensor values
	let mut output_array_ptr: *mut T = ptr::null_mut();
	let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
	let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
	ortsys![unsafe GetTensorMutableData(tensor, output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

	let array_view = unsafe { ndarray::ArrayView::from_shape_ptr(shape, output_array_ptr) };
	Ok(array_view)
}

/// Construct an [`ndarray::ArrayViewMut`] for an ORT tensor.
///
/// Only to be used on types whose Rust in-memory representation matches ONNX Runtime's (e.g. primitive numeric types
/// like u32)
#[cfg(feature = "ndarray")]
pub(crate) fn extract_primitive_array_mut<'t, T>(shape: ndarray::IxDyn, tensor: *mut ort_sys::OrtValue) -> Result<ndarray::ArrayViewMutD<'t, T>> {
	// Get pointer to output tensor values
	let mut output_array_ptr: *mut T = ptr::null_mut();
	let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
	let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
	ortsys![unsafe GetTensorMutableData(tensor, output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

	let array_view = unsafe { ndarray::ArrayViewMut::from_shape_ptr(shape, output_array_ptr) };
	Ok(array_view)
}
