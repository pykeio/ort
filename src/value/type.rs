use alloc::{
	boxed::Box,
	string::{String, ToString}
};
use core::{
	ffi::CStr,
	fmt,
	ptr::{self, NonNull}
};

use smallvec::{SmallVec, smallvec};

use crate::{
	Result, ortsys,
	tensor::{Shape, SymbolicDimensions, TensorElementType},
	util::{self, with_cstr_ptr_array}
};

/// The type of a [`Value`][super::Value], or a session input/output.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{session::Session, tensor::{Shape, SymbolicDimensions}, value::{ValueType, Tensor}, tensor::TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // `ValueType`s can be obtained from session inputs/outputs:
/// let input = &session.inputs[0];
/// assert_eq!(input.input_type, ValueType::Tensor {
/// 	ty: TensorElementType::Float32,
/// 	// Our model's input has 3 dynamic dimensions, represented by -1
/// 	shape: Shape::new([-1, -1, -1, 3]),
/// 	// Dynamic dimensions may also have names.
/// 	dimension_symbols: SymbolicDimensions::new([
/// 		"unk__31".to_string(),
/// 		"unk__32".to_string(),
/// 		"unk__33".to_string(),
/// 		String::default()
/// 	])
/// });
///
/// // ...or by `Value`s created in Rust or output by a session.
/// let value = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
/// assert_eq!(value.dtype(), &ValueType::Tensor {
/// 	ty: TensorElementType::Int64,
/// 	shape: Shape::new([5]),
/// 	dimension_symbols: SymbolicDimensions::new([String::default()])
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
		/// Shape of the tensor. If an exact dimension is not known (i.e. a dynamic dimension as part of an
		/// [`Input`]/[`Output`]), the dimension will be `-1`.
		///
		/// Actual tensor values (i.e. not [`Input`] or [`Output`] definitions), which have a known dimension, will
		/// always have non-negative dimensions.
		///
		/// [`Input`]: crate::session::Input
		/// [`Output`]: crate::session::Output
		shape: Shape,
		dimension_symbols: SymbolicDimensions
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
	pub(crate) unsafe fn from_type_info(typeinfo_ptr: NonNull<ort_sys::OrtTypeInfo>) -> Self {
		let _guard = util::run_on_drop(|| {
			ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr.as_ptr())];
		});

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr.as_ptr(), &mut ty).expect("infallible")];
		match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr) }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];

				let mut element_type_info: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe GetSequenceElementType(info_ptr.as_ptr(), &mut element_type_info).expect("infallible"); nonNull(element_type_info)];
				let _guard = util::run_on_drop(|| {
					ortsys![unsafe ReleaseTypeInfo(element_type_info.as_ptr())];
				});

				let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
				ortsys![unsafe GetOnnxTypeFromTypeInfo(element_type_info.as_ptr(), &mut ty).expect("infallible")];

				match ty {
					ort_sys::ONNXType::ONNX_TYPE_TENSOR => {
						let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = ptr::null_mut();
						ortsys![unsafe CastTypeInfoToTensorInfo(element_type_info.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];
						let ty = unsafe { extract_data_type_from_tensor_info(info_ptr) };
						ValueType::Sequence(Box::new(ty))
					}
					ort_sys::ONNXType::ONNX_TYPE_MAP => {
						let mut info_ptr: *const ort_sys::OrtMapTypeInfo = ptr::null_mut();
						ortsys![unsafe CastTypeInfoToMapTypeInfo(element_type_info.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];
						let ty = unsafe { extract_data_type_from_map_info(info_ptr) };
						ValueType::Sequence(Box::new(ty))
					}
					_ => unreachable!()
				}
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr) }
			}
			ort_sys::ONNXType::ONNX_TYPE_OPTIONAL => {
				let mut info_ptr: *const ort_sys::OrtOptionalTypeInfo = ptr::null_mut();
				ortsys![unsafe CastTypeInfoToOptionalTypeInfo(typeinfo_ptr.as_ptr(), &mut info_ptr).expect("infallible"); nonNull(info_ptr)];

				let mut contained_type: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe GetOptionalContainedTypeInfo(info_ptr.as_ptr(), &mut contained_type).expect("infallible"); nonNull(contained_type)];

				ValueType::Optional(Box::new(ValueType::from_type_info(contained_type)))
			}
			_ => unreachable!()
		}
	}

	pub(crate) fn to_tensor_type_info(&self) -> Option<*mut ort_sys::OrtTensorTypeAndShapeInfo> {
		match self {
			Self::Tensor { ty, shape, dimension_symbols } => {
				let mut info_ptr = ptr::null_mut();
				ortsys![unsafe CreateTensorTypeAndShapeInfo(&mut info_ptr).expect("infallible")];
				ortsys![unsafe SetTensorElementType(info_ptr, (*ty).into()).expect("infallible")];
				ortsys![unsafe SetDimensions(info_ptr, shape.as_ptr(), shape.len()).expect("infallible")];
				with_cstr_ptr_array(dimension_symbols, &|ptrs| {
					ortsys![unsafe SetSymbolicDimensions(info_ptr, ptrs.as_ptr().cast_mut(), dimension_symbols.len()).expect("infallible")];
					Ok(())
				})
				.expect("invalid dimension symbols");
				Some(info_ptr)
			}
			_ => None
		}
	}

	/// Converts this type to an [`ort_sys::OrtTypeInfo`] using the Model Editor API, so it shouldn't be used outside of
	/// `crate::editor`
	pub(crate) fn to_type_info(&self) -> Result<*mut ort_sys::OrtTypeInfo> {
		let mut info_ptr: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
		match self {
			Self::Tensor { .. } => {
				let tensor_type_info = self.to_tensor_type_info().expect("infallible");
				let _guard = util::run_on_drop(|| ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_type_info)]);
				ortsys![@editor: unsafe CreateTensorTypeInfo(tensor_type_info, &mut info_ptr)?];
			}
			Self::Map { .. } => {
				todo!();
			}
			Self::Sequence(ty) => {
				let el_type = ty.to_type_info()?;
				let _guard = util::run_on_drop(|| ortsys![unsafe ReleaseTypeInfo(el_type)]);
				ortsys![@editor: unsafe CreateSequenceTypeInfo(el_type, &mut info_ptr)?];
			}
			Self::Optional(ty) => {
				let ty = ty.to_type_info()?;
				let _guard = util::run_on_drop(|| ortsys![unsafe ReleaseTypeInfo(ty)]);
				ortsys![@editor: unsafe CreateOptionalTypeInfo(ty, &mut info_ptr)?];
			}
		}
		Ok(info_ptr)
	}

	/// Returns the shape of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use ort::value::{Tensor, DynValue};
	/// # fn main() -> ort::Result<()> {
	/// let value: DynValue = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?.into_dyn();
	///
	/// let shape = value.dtype().tensor_shape().unwrap();
	/// assert_eq!(**shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn tensor_shape(&self) -> Option<&Shape> {
		match self {
			ValueType::Tensor { shape, .. } => Some(shape),
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
			ValueType::Tensor { ty, shape, dimension_symbols } => {
				write!(f, "Tensor<{ty}>(")?;
				for (i, dimension) in shape.iter().copied().enumerate() {
					if dimension == -1 {
						let sym = &dimension_symbols[i];
						if sym.is_empty() {
							f.write_str("dyn")?;
						} else {
							f.write_str(sym)?;
						}
					} else {
						dimension.fmt(f)?;
					}
					if i != shape.len() - 1 {
						f.write_str(", ")?;
					}
				}
				f.write_str(")")?;
				Ok(())
			}
			ValueType::Map { key, value } => write!(f, "Map<{key}, {value}>"),
			ValueType::Sequence(inner) => write!(f, "Sequence<{inner}>"),
			ValueType::Optional(inner) => write!(f, "Option<{inner}>")
		}
	}
}

pub(crate) unsafe fn extract_data_type_from_tensor_info(info_ptr: NonNull<ort_sys::OrtTensorTypeAndShapeInfo>) -> ValueType {
	let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![unsafe GetTensorElementType(info_ptr.as_ptr(), &mut type_sys).expect("infallible")];
	assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	let mut num_dims = 0;
	ortsys![unsafe GetDimensionsCount(info_ptr.as_ptr(), &mut num_dims).expect("infallible")];

	let mut node_dims = Shape::empty(num_dims);
	ortsys![unsafe GetDimensions(info_ptr.as_ptr(), node_dims.as_mut_ptr(), num_dims).expect("infallible")];

	let mut symbolic_dims: SmallVec<_, 4> = smallvec![ptr::null(); num_dims];
	ortsys![unsafe GetSymbolicDimensions(info_ptr.as_ptr(), symbolic_dims.as_mut_ptr(), num_dims).expect("infallible")];

	let dimension_symbols = symbolic_dims
		.into_iter()
		.map(|c| unsafe { CStr::from_ptr(c) }.to_str().map_or_else(|_| String::new(), str::to_string))
		.collect();

	ValueType::Tensor {
		ty: type_sys.into(),
		shape: node_dims,
		dimension_symbols
	}
}

unsafe fn extract_data_type_from_map_info(info_ptr: NonNull<ort_sys::OrtMapTypeInfo>) -> ValueType {
	let mut key_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![unsafe GetMapKeyType(info_ptr.as_ptr(), &mut key_type_sys).expect("infallible")];
	assert_ne!(key_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	let mut value_type_info: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
	ortsys![unsafe GetMapValueType(info_ptr.as_ptr(), &mut value_type_info).expect("infallible")];
	let _guard = util::run_on_drop(|| {
		ortsys![unsafe ReleaseTypeInfo(value_type_info)];
	});

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
	use core::ptr::NonNull;

	use super::ValueType;
	use crate::{
		ortsys,
		tensor::{Shape, SymbolicDimensions, TensorElementType}
	};

	#[test]
	fn test_to_from_tensor_info() -> crate::Result<()> {
		let ty = ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([-1, 32, 4, 32]),
			dimension_symbols: SymbolicDimensions::new(["d1".to_string(), String::default(), String::default(), String::default()])
		};
		let ty_ptr = NonNull::new(ty.to_tensor_type_info().expect("")).expect("");
		let ty_d = unsafe { super::extract_data_type_from_tensor_info(ty_ptr) };
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(ty_ptr.as_ptr())];
		assert_eq!(ty, ty_d);

		Ok(())
	}
}
