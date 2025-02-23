use alloc::{
	boxed::Box,
	ffi::CString,
	string::{String, ToString},
	vec,
	vec::Vec
};
use core::{
	ffi::{CStr, c_char},
	fmt, ptr
};

use crate::{ortsys, tensor::TensorElementType};

/// The type of a [`Value`][super::Value], or a session input/output.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{session::Session, value::{ValueType, Tensor}, tensor::TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // `ValueType`s can be obtained from session inputs/outputs:
/// let input = &session.inputs[0];
/// assert_eq!(input.input_type, ValueType::Tensor {
/// 	ty: TensorElementType::Float32,
/// 	// Our model has 3 dynamic dimensions, represented by -1
/// 	dimensions: vec![-1, -1, -1, 3],
/// 	// Dynamic dimensions may also have names.
/// 	dimension_symbols: vec![
/// 		Some("unk__31".to_string()),
/// 		Some("unk__32".to_string()),
/// 		Some("unk__33".to_string()),
/// 		None
/// 	]
/// });
///
/// // ...or by `Value`s created in Rust or output by a session.
/// let value = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
/// assert_eq!(value.dtype(), &ValueType::Tensor {
/// 	ty: TensorElementType::Int64,
/// 	dimensions: vec![5],
/// 	dimension_symbols: vec![None]
/// });
/// # 	Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ValueType {
	/// Value is a tensor/multi-dimensional array.
	Tensor {
		/// Element type of the tensor.
		ty: TensorElementType,
		/// Dimensions of the tensor. If an exact dimension is not known (i.e. a dynamic dimension as part of an
		/// [`Input`]/[`Output`]), the dimension will be `-1`.
		///
		/// Actual tensor values, which have a known dimension, will always have positive (>1) dimensions.
		///
		/// [`Input`]: crate::session::Input
		/// [`Output`]: crate::session::Output
		dimensions: Vec<i64>,
		dimension_symbols: Vec<Option<String>>
	},
	/// A sequence (vector) of other `Value`s.
	///
	/// [Per ONNX spec](https://onnx.ai/onnx/intro/concepts.html#other-types), only sequences of tensors and maps are allowed.
	Sequence(Box<ValueType>),
	/// A map/dictionary from one element type to another.
	Map {
		/// The map key type. Allowed types are:
		/// - [`TensorElementType::Int8`]
		/// - [`TensorElementType::Int16`]
		/// - [`TensorElementType::Int32`]
		/// - [`TensorElementType::Int64`]
		/// - [`TensorElementType::Uint8`]
		/// - [`TensorElementType::Uint16`]
		/// - [`TensorElementType::Uint32`]
		/// - [`TensorElementType::Uint64`]
		/// - [`TensorElementType::String`]
		key: TensorElementType,
		/// The map value type.
		value: TensorElementType
	},
	/// An optional value, which may or may not contain a [`Value`][super::Value].
	Optional(Box<ValueType>)
}

impl ValueType {
	pub(crate) fn from_type_info(typeinfo_ptr: *mut ort_sys::OrtTypeInfo) -> Self {
		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty).expect("infallible")];
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr).expect("infallible")];
				unsafe { extract_data_type_from_tensor_info(info_ptr) }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr).expect("infallible")];

				let mut element_type_info: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe GetSequenceElementType(info_ptr, &mut element_type_info).expect("infallible")];

				let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
				ortsys![unsafe GetOnnxTypeFromTypeInfo(element_type_info, &mut ty).expect("infallible")];

				match ty {
					ort_sys::ONNXType::ONNX_TYPE_TENSOR => {
						let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
						ortsys![unsafe CastTypeInfoToTensorInfo(element_type_info, &mut info_ptr).expect("infallible")];
						let ty = unsafe { extract_data_type_from_tensor_info(info_ptr) };
						ValueType::Sequence(Box::new(ty))
					}
					ort_sys::ONNXType::ONNX_TYPE_MAP => {
						let mut info_ptr: *const ort_sys::OrtMapTypeInfo = ptr::null_mut();
						ortsys![unsafe CastTypeInfoToMapTypeInfo(element_type_info, &mut info_ptr).expect("infallible")];
						let ty = unsafe { extract_data_type_from_map_info(info_ptr) };
						ValueType::Sequence(Box::new(ty))
					}
					_ => unreachable!()
				}
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr).expect("infallible")];
				unsafe { extract_data_type_from_map_info(info_ptr) }
			}
			ort_sys::ONNXType::ONNX_TYPE_OPTIONAL => {
				let mut info_ptr: *const ort_sys::OrtOptionalTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToOptionalTypeInfo(typeinfo_ptr, &mut info_ptr).expect("infallible")];

				let mut contained_type: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe GetOptionalContainedTypeInfo(info_ptr, &mut contained_type).expect("infallible")];

				ValueType::Optional(Box::new(ValueType::from_type_info(contained_type)))
			}
			_ => unreachable!()
		};
		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		io_type
	}

	pub(crate) fn to_tensor_type_info(&self) -> Option<*mut ort_sys::OrtTensorTypeAndShapeInfo> {
		match self {
			Self::Tensor { ty, dimensions, dimension_symbols } => {
				let mut info_ptr = ptr::null_mut();
				ortsys![unsafe CreateTensorTypeAndShapeInfo(&mut info_ptr).expect("infallible")];
				ortsys![unsafe SetTensorElementType(info_ptr, (*ty).into()).expect("infallible")];
				ortsys![unsafe SetDimensions(info_ptr, dimensions.as_ptr(), dimensions.len()).expect("infallible")];
				let dimension_symbols: Vec<*const c_char> = dimension_symbols
					.iter()
					.cloned()
					.map(|s| CString::new(s.unwrap_or_default()))
					.map(|s| s.map_or(ptr::null(), |s| s.into_raw().cast_const()))
					.collect();
				ortsys![unsafe SetSymbolicDimensions(info_ptr, dimension_symbols.as_ptr().cast_mut(), dimension_symbols.len()).expect("infallible")];
				for p in dimension_symbols {
					if !p.is_null() {
						drop(unsafe { CString::from_raw(p.cast_mut().cast()) });
					}
				}
				Some(info_ptr)
			}
			_ => None
		}
	}

	/// Returns the dimensions of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let value = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
	/// assert_eq!(value.dtype().tensor_dimensions(), Some(&vec![5]));
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn tensor_dimensions(&self) -> Option<&Vec<i64>> {
		match self {
			ValueType::Tensor { dimensions, .. } => Some(dimensions),
			_ => None
		}
	}

	/// Returns the element type of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use ort::{tensor::TensorElementType, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let value = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
	/// assert_eq!(value.dtype().tensor_type(), Some(TensorElementType::Int64));
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn tensor_type(&self) -> Option<TensorElementType> {
		match self {
			ValueType::Tensor { ty, .. } => Some(*ty),
			_ => None
		}
	}

	/// Returns `true` if this value type is a tensor.
	#[inline]
	#[must_use]
	pub fn is_tensor(&self) -> bool {
		matches!(self, ValueType::Tensor { .. })
	}

	/// Returns `true` if this value type is a sequence.
	#[inline]
	#[must_use]
	pub fn is_sequence(&self) -> bool {
		matches!(self, ValueType::Sequence { .. })
	}

	/// Returns `true` if this value type is a map.
	#[inline]
	#[must_use]
	pub fn is_map(&self) -> bool {
		matches!(self, ValueType::Map { .. })
	}
}

impl fmt::Display for ValueType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			ValueType::Tensor { ty, dimensions, dimension_symbols } => {
				write!(
					f,
					"Tensor<{ty}>({})",
					dimensions
						.iter()
						.enumerate()
						.map(|(i, c)| if *c == -1 {
							dimension_symbols[i].clone().unwrap_or_else(|| String::from("dyn"))
						} else {
							c.to_string()
						})
						.collect::<Vec<_>>()
						.join(", ")
				)
			}
			ValueType::Map { key, value } => write!(f, "Map<{key}, {value}>"),
			ValueType::Sequence(inner) => write!(f, "Sequence<{inner}>"),
			ValueType::Optional(inner) => write!(f, "Option<{inner}>")
		}
	}
}

pub(crate) unsafe fn extract_data_type_from_tensor_info(info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo) -> ValueType {
	let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![unsafe GetTensorElementType(info_ptr, &mut type_sys).expect("infallible")];
	assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	let mut num_dims = 0;
	ortsys![unsafe GetDimensionsCount(info_ptr, &mut num_dims).expect("infallible")];

	let mut node_dims: Vec<i64> = vec![0; num_dims];
	ortsys![unsafe GetDimensions(info_ptr, node_dims.as_mut_ptr(), num_dims).expect("infallible")];

	let mut symbolic_dims: Vec<*const c_char> = vec![ptr::null(); num_dims];
	ortsys![unsafe GetSymbolicDimensions(info_ptr, symbolic_dims.as_mut_ptr(), num_dims).expect("infallible")];

	let dimension_symbols = symbolic_dims
		.into_iter()
		.map(|c| {
			if !c.is_null() && unsafe { *c } != 0 {
				unsafe { CStr::from_ptr(c) }.to_str().ok().map(str::to_string)
			} else {
				None
			}
		})
		.collect();

	ValueType::Tensor {
		ty: type_sys.into(),
		dimensions: node_dims,
		dimension_symbols
	}
}

unsafe fn extract_data_type_from_map_info(info_ptr: *const ort_sys::OrtMapTypeInfo) -> ValueType {
	let mut key_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![unsafe GetMapKeyType(info_ptr, &mut key_type_sys).expect("infallible")];
	assert_ne!(key_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	let mut value_type_info: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
	ortsys![unsafe GetMapValueType(info_ptr, &mut value_type_info).expect("infallible")];
	let mut value_info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
	ortsys![unsafe CastTypeInfoToTensorInfo(value_type_info, &mut value_info_ptr).expect("infallible")];
	let mut value_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![unsafe GetTensorElementType(value_info_ptr, &mut value_type_sys).expect("infallible")];
	assert_ne!(value_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	ValueType::Map {
		key: key_type_sys.into(),
		value: value_type_sys.into()
	}
}

#[cfg(test)]
mod tests {
	use super::ValueType;
	use crate::{ortsys, tensor::TensorElementType};

	#[test]
	fn test_to_from_tensor_info() -> crate::Result<()> {
		let ty = ValueType::Tensor {
			ty: TensorElementType::Float32,
			dimensions: vec![-1, 32, 4, 32],
			dimension_symbols: vec![Some("d1".to_string()), None, None, None]
		};
		let ty_ptr = ty.to_tensor_type_info().expect("");
		let ty_d = unsafe { super::extract_data_type_from_tensor_info(ty_ptr) };
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(ty_ptr)];
		assert_eq!(ty, ty_d);

		Ok(())
	}
}
