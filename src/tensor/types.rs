use alloc::string::String;
use core::fmt;

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
	/// 16-bit floating point number, equivalent to [`half::f16`] (with the `half` feature).
	Float16,
	/// 64-bit floating point number, equivalent to Rust's `f64`. Also known as `double`.
	Float64,
	/// Unsigned 32-bit integer, equivalent to Rust's `u32`.
	Uint32,
	/// Unsigned 64-bit integer, equivalent to Rust's `u64`.
	Uint64,
	/// Brain 16-bit floating point number, equivalent to [`half::bf16`] (with the `half` feature).
	Bfloat16,
	Complex64,
	Complex128,
	/// 8-bit floating point number with 4 exponent bits and 3 mantissa bits, with only NaN values and no infinite
	/// values.
	Float8E4M3FN,
	/// 8-bit floating point number with 4 exponent bits and 3 mantissa bits, with only NaN values, no infinite
	/// values, and no negative zero.
	Float8E4M3FNUZ,
	/// 8-bit floating point number with 5 exponent bits and 2 mantissa bits.
	Float8E5M2,
	/// 8-bit floating point number with 5 exponent bits and 2 mantissa bits, with only NaN values, no infinite
	/// values, and no negative zero.
	Float8E5M2FNUZ,
	/// 4-bit unsigned integer.
	Uint4,
	/// 4-bit signed integer.
	Int4,
	Undefined
}

impl TensorElementType {
	/// Returns the size in bytes that a container of this type occupies according to its total capacity.
	pub fn byte_size(&self, container_capacity: usize) -> usize {
		match self {
			TensorElementType::Uint4 | TensorElementType::Int4 => container_capacity / 2,
			TensorElementType::Bool | TensorElementType::Int8 | TensorElementType::Uint8 => container_capacity,
			TensorElementType::Int16 | TensorElementType::Uint16 => container_capacity * 2,
			TensorElementType::Int32 | TensorElementType::Uint32 => container_capacity * 4,
			TensorElementType::Int64 | TensorElementType::Uint64 => container_capacity * 8,
			TensorElementType::String => 0, // unsure what to do about this...
			TensorElementType::Float8E4M3FN | TensorElementType::Float8E4M3FNUZ | TensorElementType::Float8E5M2 | TensorElementType::Float8E5M2FNUZ => {
				container_capacity * 4
			}
			TensorElementType::Float16 | TensorElementType::Bfloat16 => container_capacity * 2,
			TensorElementType::Float32 => container_capacity * 4,
			TensorElementType::Float64 => container_capacity * 8,
			TensorElementType::Complex64 => container_capacity * 8,
			TensorElementType::Complex128 => container_capacity * 16,
			TensorElementType::Undefined => 0
		}
	}
}

impl fmt::Display for TensorElementType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			TensorElementType::Bool => "bool",
			TensorElementType::Bfloat16 => "bf16",
			TensorElementType::Float16 => "f16",
			TensorElementType::Float32 => "f32",
			TensorElementType::Float64 => "f64",
			TensorElementType::Int16 => "i16",
			TensorElementType::Int32 => "i32",
			TensorElementType::Int64 => "i64",
			TensorElementType::Int8 => "i8",
			TensorElementType::Int4 => "i4",
			TensorElementType::String => "String",
			TensorElementType::Uint16 => "u16",
			TensorElementType::Uint32 => "u32",
			TensorElementType::Uint64 => "u64",
			TensorElementType::Uint8 => "u8",
			TensorElementType::Uint4 => "u4",
			TensorElementType::Complex64 => "c64",
			TensorElementType::Complex128 => "c128",
			// these really need more memorable (and easier to type) names. like Gerald or perhaps Alexa
			TensorElementType::Float8E4M3FN => "f8_e4m3fn",
			TensorElementType::Float8E4M3FNUZ => "f8_e4m3fnuz",
			TensorElementType::Float8E5M2 => "f8_e5m2",
			TensorElementType::Float8E5M2FNUZ => "f8_e5m2fnuz",
			TensorElementType::Undefined => "undefined"
		})
	}
}

impl From<TensorElementType> for ort_sys::ONNXTensorElementDataType {
	fn from(val: TensorElementType) -> Self {
		match val {
			TensorElementType::Undefined => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
			TensorElementType::Float32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
			TensorElementType::Uint8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
			TensorElementType::Int8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
			TensorElementType::Uint16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
			TensorElementType::Int16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
			TensorElementType::Int32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
			TensorElementType::Int64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
			TensorElementType::String => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
			TensorElementType::Bool => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
			TensorElementType::Float16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
			TensorElementType::Float64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
			TensorElementType::Uint32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
			TensorElementType::Uint64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
			TensorElementType::Bfloat16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
			TensorElementType::Int4 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
			TensorElementType::Uint4 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
			TensorElementType::Complex64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
			TensorElementType::Complex128 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
			TensorElementType::Float8E4M3FN => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
			TensorElementType::Float8E4M3FNUZ => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
			TensorElementType::Float8E5M2 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
			TensorElementType::Float8E5M2FNUZ => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ
		}
	}
}
impl From<ort_sys::ONNXTensorElementDataType> for TensorElementType {
	fn from(val: ort_sys::ONNXTensorElementDataType) -> Self {
		match val {
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED => TensorElementType::Undefined,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => TensorElementType::Float32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => TensorElementType::Uint8,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => TensorElementType::Int8,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => TensorElementType::Uint16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => TensorElementType::Int16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => TensorElementType::Int32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => TensorElementType::Int64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => TensorElementType::String,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => TensorElementType::Bool,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => TensorElementType::Float16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => TensorElementType::Float64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => TensorElementType::Uint32,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => TensorElementType::Uint64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 => TensorElementType::Bfloat16,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 => TensorElementType::Int4,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 => TensorElementType::Uint4,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64 => TensorElementType::Complex64,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128 => TensorElementType::Complex128,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN => TensorElementType::Float8E4M3FN,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ => TensorElementType::Float8E4M3FNUZ,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 => TensorElementType::Float8E5M2,
			ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ => TensorElementType::Float8E5M2FNUZ
		}
	}
}

/// Trait used to map Rust types (for example `f32`) to ONNX tensor element data types (for example `Float`).
pub trait IntoTensorElementType {
	/// Returns the ONNX tensor element data type corresponding to the given Rust type.
	fn into_tensor_element_type() -> TensorElementType;

	private_trait!();
}

/// A superset of [`IntoTensorElementType`] that represents traits whose underlying memory is identical between Rust and
/// C++ (i.e., every type except `String`).
pub trait PrimitiveTensorElementType: IntoTensorElementType {
	private_trait!();
}

macro_rules! impl_type_trait {
	($type_:ty, $variant:ident) => {
		impl IntoTensorElementType for $type_ {
			fn into_tensor_element_type() -> TensorElementType {
				TensorElementType::$variant
			}

			private_impl!();
		}

		impl PrimitiveTensorElementType for $type_ {
			private_impl!();
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
#[cfg(feature = "num-complex")]
#[cfg_attr(docsrs, doc(cfg(feature = "num-complex")))]
impl_type_trait!(num_complex::Complex32, Complex64);
#[cfg(feature = "num-complex")]
#[cfg_attr(docsrs, doc(cfg(feature = "num-complex")))]
impl_type_trait!(num_complex::Complex64, Complex128);

impl IntoTensorElementType for String {
	fn into_tensor_element_type() -> TensorElementType {
		TensorElementType::String
	}

	private_impl!();
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

impl Utf8Data for &str {
	fn as_utf8_bytes(&self) -> &[u8] {
		self.as_bytes()
	}
}
