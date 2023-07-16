//! Module containing tensor types.
//!
//! Two main types of tensors are available.
//!
//! The first one, [`OrtTensor`], is an _owned_ tensor that is backed by [`ndarray`](https://crates.io/crates/ndarray).
//! This kind of tensor is used to pass input data for the inference.
//!
//! The second one, [`OrtOwnedTensor`], is used internally to pass to the ONNX Runtime inference execution to place its
//! output values. Once "extracted" from the runtime environment, this tensor will contain an [`ndarray::ArrayView`]
//! containing _a view_ of the data. When going out of scope, this tensor will free the required memory on the C side.
//!
//! **NOTE**: Tensors are not meant to be created directly. When performing inference, the
//! [`Session::run`](crate::Session::run) method takes an `ndarray::Array` as input (taking ownership of it) and will
//! convert it internally to an [`OrtTensor`]. After inference, a [`OrtOwnedTensor`] will be returned by the method
//! which can be derefed into its internal [`ndarray::ArrayView`].

pub mod ndarray_tensor;
pub mod ort_owned_tensor;

use std::{ffi, fmt, ptr, result, string};

pub use self::ndarray_tensor::NdArrayExtensions;
pub use self::ort_owned_tensor::OrtOwnedTensor;
use super::{ortsys, sys, OrtError, OrtResult};

/// Enum mapping ONNX Runtime's supported tensor data types.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum TensorElementDataType {
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
	/// 16-bit floating point number, equivalent to `half::f16` (requires the `half` crate).
	#[cfg(feature = "half")]
	Float16,
	/// 64-bit floating point number, equivalent to Rust's `f64`. Also known as `double`.
	Float64,
	/// Unsigned 32-bit integer, equivalent to Rust's `u32`.
	Uint32,
	/// Unsigned 64-bit integer, equivalent to Rust's `u64`.
	Uint64,
	// /// Complex 64-bit floating point number, equivalent to Rust's `num_complex::Complex<f64>`.
	// Complex64,
	// TODO: `num_complex` crate doesn't support i128 provided by the `decimal` crate.
	// /// Complex 128-bit floating point number, equivalent to Rust's `num_complex::Complex<f128>`.
	// Complex128,
	/// Brain 16-bit floating point number, equivalent to `half::bf16` (requires the `half` crate).
	#[cfg(feature = "half")]
	Bfloat16
}

impl From<TensorElementDataType> for sys::ONNXTensorElementDataType {
	fn from(val: TensorElementDataType) -> Self {
		match val {
			TensorElementDataType::Float32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			TensorElementDataType::Uint8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			TensorElementDataType::Int8 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
			TensorElementDataType::Uint16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
			TensorElementDataType::Int16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
			TensorElementDataType::Int32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
			TensorElementDataType::Int64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
			TensorElementDataType::String => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
			TensorElementDataType::Bool => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
			#[cfg(feature = "half")]
			TensorElementDataType::Float16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
			TensorElementDataType::Float64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
			TensorElementDataType::Uint32 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
			TensorElementDataType::Uint64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
			// TensorElementDataType::Complex64 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
			// TensorElementDataType::Complex128 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 => sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
		}
	}
}
impl From<sys::ONNXTensorElementDataType> for TensorElementDataType {
	fn from(val: sys::ONNXTensorElementDataType) -> Self {
		match val {
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => TensorElementDataType::Float32,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => TensorElementDataType::Uint8,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => TensorElementDataType::Int8,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => TensorElementDataType::Uint16,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => TensorElementDataType::Int16,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => TensorElementDataType::Int32,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => TensorElementDataType::Int64,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => TensorElementDataType::String,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => TensorElementDataType::Bool,
			#[cfg(feature = "half")]
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => TensorElementDataType::Float16,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => TensorElementDataType::Float64,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => TensorElementDataType::Uint32,
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => TensorElementDataType::Uint64,
			// sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => TensorElementDataType::Complex64,
			// sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => TensorElementDataType::Complex128,
			#[cfg(feature = "half")]
			sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => TensorElementDataType::Bfloat16,
			_ => panic!("Invalid ONNXTensorElementDataType value")
		}
	}
}

/// Trait used to map Rust types (for example `f32`) to ONNX tensor element data types (for example `Float`).
pub trait IntoTensorElementDataType {
	/// Returns the ONNX tensor element data type corresponding to the given Rust type.
	fn tensor_element_data_type() -> TensorElementDataType;

	/// If the type is `String`, returns `Some` with UTF-8 contents, else `None`.
	fn try_utf8_bytes(&self) -> Option<&[u8]>;
}

macro_rules! impl_type_trait {
	($type_:ty, $variant:ident) => {
		impl IntoTensorElementDataType for $type_ {
			fn tensor_element_data_type() -> TensorElementDataType {
				TensorElementDataType::$variant
			}

			fn try_utf8_bytes(&self) -> Option<&[u8]> {
				None
			}
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
impl_type_trait!(half::f16, Float16);
impl_type_trait!(f64, Float64);
impl_type_trait!(u32, Uint32);
impl_type_trait!(u64, Uint64);
// impl_type_trait!(num_complex::Complex<f64>, Complex64);
// impl_type_trait!(num_complex::Complex<f128>, Complex128);
#[cfg(feature = "half")]
impl_type_trait!(half::bf16, Bfloat16);

/// Adapter for common Rust string types to ONNX strings.
///
/// It should be easy to use both [`String`] and `&str` as [`TensorElementDataType::String`] data, but
/// we can't define an automatic implementation for anything that implements [`AsRef<str>`] as it
/// would conflict with the implementations of [`IntoTensorElementDataType`] for primitive numeric
/// types (which might implement [`AsRef<str>`] at some point in the future).
pub trait Utf8Data {
	/// Returns the contents of this value as a slice of UTF-8 bytes.
	fn utf8_bytes(&self) -> &[u8];
}

impl Utf8Data for String {
	fn utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<'a> Utf8Data for &'a str {
	fn utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}

impl<T: Utf8Data> IntoTensorElementDataType for T {
	fn tensor_element_data_type() -> TensorElementDataType {
		TensorElementDataType::String
	}

	fn try_utf8_bytes(&self) -> Option<&[u8]> {
		Some(self.utf8_bytes())
	}
}

/// Trait used to map ONNX Runtime types to Rust types.
pub trait TensorDataToType: Sized + fmt::Debug + Clone {
	/// The tensor element type that this type can extract from.
	fn tensor_element_data_type() -> TensorElementDataType;

	/// Extract an `ArrayView` from the ORT-owned tensor.
	fn extract_data<'t, D>(shape: D, tensor_element_len: usize, tensor_ptr: *mut sys::OrtValue) -> OrtResult<TensorData<'t, Self, D>>
	where
		D: ndarray::Dimension;
}

/// Represents the possible ways tensor data can be accessed.
///
/// This should only be used internally.
#[derive(Debug)]
pub enum TensorData<'t, T, D>
where
	D: ndarray::Dimension
{
	/// Data residing in ONNX Runtime's tensor, in which case the `'t` lifetime is what makes this valid.
	/// This is used for data types whose in-memory form from ONNX Runtime is compatible with Rust's, like
	/// primitive numeric types.
	TensorPtr {
		/// The pointer ONNX Runtime produced. Kept alive so that `array_view` is valid.
		ptr: *mut sys::OrtValue,
		/// A view into `ptr`.
		array_view: ndarray::ArrayView<'t, T, D>
	},
	/// String data is output differently by ONNX, and is of course also variable size, so it cannot
	/// use the same simple pointer representation.
	// Since `'t` outlives this struct, the 't lifetime is more than we need, but no harm done.
	Strings {
		/// Owned Strings copied out of ONNX Runtime's output.
		strings: ndarray::Array<T, D>
	}
}

/// Implements [`TensorDataToType`] for primitives which can use `GetTensorMutableData`.
macro_rules! impl_prim_type_from_ort_trait {
	($type_: ty, $variant: ident) => {
		impl TensorDataToType for $type_ {
			fn tensor_element_data_type() -> TensorElementDataType {
				TensorElementDataType::$variant
			}

			fn extract_data<'t, D>(shape: D, _tensor_element_len: usize, tensor_ptr: *mut sys::OrtValue) -> OrtResult<TensorData<'t, Self, D>>
			where
				D: ndarray::Dimension
			{
				extract_primitive_array(shape, tensor_ptr).map(|v| TensorData::TensorPtr { ptr: tensor_ptr, array_view: v })
			}
		}
	};
}

/// Construct an [`ndarray::ArrayView`] for an ORT tensor.
///
/// Only to be used on types whose Rust in-memory representation matches ONNX Runtime's (e.g. primitive numeric types
/// like u32)
fn extract_primitive_array<'t, D, T: TensorDataToType>(shape: D, tensor: *mut sys::OrtValue) -> OrtResult<ndarray::ArrayView<'t, T, D>>
where
	D: ndarray::Dimension
{
	// Get pointer to output tensor values
	let mut output_array_ptr: *mut T = ptr::null_mut();
	let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
	let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr as *mut *mut std::ffi::c_void;
	ortsys![unsafe GetTensorMutableData(tensor, output_array_ptr_ptr_void) -> OrtError::GetTensorMutableData; nonNull(output_array_ptr)];

	let array_view = unsafe { ndarray::ArrayView::from_shape_ptr(shape, output_array_ptr) };
	Ok(array_view)
}

#[cfg(feature = "half")]
impl_prim_type_from_ort_trait!(half::f16, Float16);
#[cfg(feature = "half")]
impl_prim_type_from_ort_trait!(half::bf16, Bfloat16);
impl_prim_type_from_ort_trait!(f32, Float32);
impl_prim_type_from_ort_trait!(f64, Float64);
impl_prim_type_from_ort_trait!(u8, Uint8);
impl_prim_type_from_ort_trait!(u16, Uint16);
impl_prim_type_from_ort_trait!(u32, Uint32);
impl_prim_type_from_ort_trait!(u64, Uint64);
impl_prim_type_from_ort_trait!(i8, Int8);
impl_prim_type_from_ort_trait!(i16, Int16);
impl_prim_type_from_ort_trait!(i32, Int32);
impl_prim_type_from_ort_trait!(i64, Int64);
impl_prim_type_from_ort_trait!(bool, Bool);

impl TensorDataToType for String {
	fn tensor_element_data_type() -> TensorElementDataType {
		TensorElementDataType::String
	}

	#[allow(clippy::not_unsafe_ptr_arg_deref)]
	fn extract_data<'t, D: ndarray::Dimension>(shape: D, tensor_element_len: usize, tensor_ptr: *mut sys::OrtValue) -> OrtResult<TensorData<'t, Self, D>> {
		// Total length of string data, not including \0 suffix
		let mut total_length = 0;
		ortsys![unsafe GetStringTensorDataLength(tensor_ptr, &mut total_length) -> OrtError::GetStringTensorDataLength];

		// In the JNI impl of this, tensor_element_len was included in addition to total_length,
		// but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
		// don't seem to be written to in practice either.
		// If the string data actually did go farther, it would panic below when using the offset
		// data to get slices for each string.
		let mut string_contents = vec![0u8; total_length as _];
		// one extra slot so that the total length can go in the last one, making all per-string
		// length calculations easy
		let mut offsets = vec![0; tensor_element_len + 1];

		ortsys![unsafe GetStringTensorContent(tensor_ptr, string_contents.as_mut_ptr() as *mut ffi::c_void, total_length as _, offsets.as_mut_ptr(), tensor_element_len as _) -> OrtError::GetStringTensorContent];

		// final offset = overall length so that per-string length calculations work for the last string
		debug_assert_eq!(0, offsets[tensor_element_len]);
		offsets[tensor_element_len] = total_length;

		let strings = offsets
            // offsets has 1 extra offset past the end so that all windows work
            .windows(2)
            .map(|w| {
                let slice = &string_contents[w[0] as _..w[1] as _];
                String::from_utf8(slice.into())
            })
            .collect::<result::Result<Vec<String>, string::FromUtf8Error>>()
            .map_err(OrtError::StringFromUtf8Error)?;

		let array = ndarray::Array::from_shape_vec(shape, strings).expect("Shape extracted from tensor didn't match tensor contents");

		Ok(TensorData::Strings { strings: array })
	}
}
