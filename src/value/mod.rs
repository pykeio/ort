use std::{
	any::Any,
	fmt::Debug,
	marker::PhantomData,
	ops::{Deref, DerefMut},
	ptr::NonNull,
	sync::Arc
};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, CowArray, Dimension};

use crate::{
	error::status_to_result,
	memory::MemoryInfo,
	ortsys,
	session::SharedSessionInner,
	tensor::{IntoTensorElementType, TensorElementType},
	Error, Result
};

mod impl_map;
mod impl_sequence;
mod impl_tensor;

use self::impl_tensor::ToDimensions;

/// The type of a [`Value`], or a session input/output.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{Session, Value, ValueType, TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // `ValueType`s can be obtained from session inputs/outputs:
/// let input = &session.inputs[0];
/// assert_eq!(
/// 	input.input_type,
/// 	ValueType::Tensor {
/// 		ty: TensorElementType::Float32,
/// 		// Our model has 3 dynamic dimensions, represented by -1
/// 		dimensions: vec![-1, -1, -1, 3]
/// 	}
/// );
///
/// // Or by `Value`s created in Rust or output by a session.
/// let value = Value::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
/// assert_eq!(
/// 	value.dtype()?,
/// 	ValueType::Tensor {
/// 		ty: TensorElementType::Int64,
/// 		dimensions: vec![5]
/// 	}
/// );
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
		/// [`crate::Input`]/[`crate::Output`]), the dimension will be `-1`.
		///
		/// Actual tensor values, which have a known dimension, will always have positive (>1) dimensions.
		dimensions: Vec<i64>
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
	}
}

impl ValueType {
	/// Returns the dimensions of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use ort::{Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let value = Value::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
	/// assert_eq!(value.dtype()?.tensor_dimensions(), Some(&vec![5]));
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
	/// # use ort::{Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let value = Value::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
	/// assert_eq!(value.dtype()?.tensor_type(), Some(TensorElementType::Int64));
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

#[derive(Debug)]
pub(crate) enum ValueInner {
	RustOwned {
		ptr: NonNull<ort_sys::OrtValue>,
		_array: Box<dyn Any>,
		_memory_info: MemoryInfo
	},
	CppOwned {
		ptr: NonNull<ort_sys::OrtValue>,
		/// Whether to release the value pointer on drop.
		drop: bool,
		/// Hold [`SharedSessionInner`] to ensure that the value can stay alive after the main session is dropped.
		///
		/// This may be `None` if the value is created outside of a session or if the value does not need to hold onto
		/// the session reference. In the case of sequence/map values, we forego this because:
		/// - a map value can be created independently of a session, and thus we wouldn't have anything to hold on to;
		/// - this is only ever used by `ValueRef`s, whos owner value (which *is* holding the session Arc) will outlive
		///   it.
		_session: Option<Arc<SharedSessionInner>>
	}
}

/// A temporary version of [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRef<'v> {
	inner: Value,
	lifetime: PhantomData<&'v ()>
}

impl<'v> ValueRef<'v> {
	pub(crate) fn new(inner: Value) -> Self {
		ValueRef { inner, lifetime: PhantomData }
	}
}

impl<'v> Deref for ValueRef<'v> {
	type Target = Value;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

/// A mutable temporary version of [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRefMut<'v> {
	inner: Value,
	lifetime: PhantomData<&'v ()>
}

impl<'v> ValueRefMut<'v> {
	pub(crate) fn new(inner: Value) -> Self {
		ValueRefMut { inner, lifetime: PhantomData }
	}
}

impl<'v> Deref for ValueRefMut<'v> {
	type Target = Value;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<'v> DerefMut for ValueRefMut<'v> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

/// A [`Value`] contains data for inputs/outputs in ONNX Runtime graphs. [`Value`]s can hold a tensor, sequence
/// (array/vector), or map.
///
/// ## Creation
/// `Value`s can be created via methods like [`Value::from_array`], or as the output from running a [`crate::Session`].
///
/// ```
/// # use ort::{Session, Value, ValueType, TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let upsample = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // Create a value from a raw data vector
/// let value = Value::from_array(([1usize, 1, 1, 3], vec![1.0_f32, 2.0, 3.0].into_boxed_slice()))?;
///
/// // Create a value from an `ndarray::Array`
/// #[cfg(feature = "ndarray")]
/// let value = Value::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
///
/// // Get a value from a session's output
/// let value = &upsample.run(ort::inputs![value]?)?[0];
/// # 	Ok(())
/// # }
/// ```
///
/// See [`Value::from_array`] for more details on what tensor values are accepted.
///
/// ## Usage
/// You can access the data in a `Value` by using the relevant `extract` methods: [`Value::extract_tensor`] &
/// [`Value::extract_raw_tensor`], [`Value::extract_sequence`], and [`Value::extract_map`].
#[derive(Debug)]
pub struct Value {
	inner: ValueInner
}

unsafe impl Send for Value {}

impl Value {
	/// Returns the data type of this [`Value`].
	pub fn dtype(&self) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTypeInfo(self.ptr(), &mut typeinfo_ptr) -> Error::GetTypeInfo; nonNull(typeinfo_ptr)];

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty)];
		status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToSequenceTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_sequence_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr)? }
			}
			_ => unreachable!()
		};

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		Ok(io_type)
	}

	/// Construct a [`Value`] from a C++ [`ort_sys::OrtValue`] pointer.
	///
	/// If the value belongs to a session (i.e. if it is returned from [`crate::Session::run`] or
	/// [`crate::IoBinding::run`]), you must provide the [`SharedSessionInner`] (acquired from
	/// [`crate::Session::inner`]). This ensures the session is not dropped until the value is.
	///
	/// # Safety
	///
	/// - `ptr` must be a valid pointer to an [`ort_sys::OrtValue`].
	/// - `session` must be `Some` for values returned from a session.
	#[must_use]
	pub unsafe fn from_ptr(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value {
		Value {
			inner: ValueInner::CppOwned { ptr, drop: true, _session: session }
		}
	}

	/// A variant of [`Value::from_ptr`] that does not release the value upon dropping. Used in operator kernel
	/// contexts.
	#[must_use]
	pub(crate) unsafe fn from_ptr_nodrop(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value {
		Value {
			inner: ValueInner::CppOwned { ptr, drop: false, _session: session }
		}
	}

	/// Returns the underlying [`ort_sys::OrtValue`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtValue {
		match &self.inner {
			ValueInner::CppOwned { ptr, .. } | ValueInner::RustOwned { ptr, .. } => ptr.as_ptr()
		}
	}

	/// Returns `true` if this value is a tensor, or `false` if it is another type (sequence, map).
	///
	/// ```
	/// # use ort::Value;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let tensor_value = Value::from_array(([3usize], vec![1.0_f32, 2.0, 3.0].into_boxed_slice()))?;
	/// assert!(tensor_value.is_tensor()?);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn is_tensor(&self) -> Result<bool> {
		let mut result = 0;
		ortsys![unsafe IsTensor(self.ptr(), &mut result) -> Error::GetTensorElementType];
		Ok(result == 1)
	}
}

impl Drop for Value {
	fn drop(&mut self) {
		let ptr = self.ptr();
		tracing::trace!(
			"dropping {} value at {ptr:p}",
			match &self.inner {
				ValueInner::RustOwned { .. } => "rust-owned",
				ValueInner::CppOwned { .. } => "cpp-owned"
			}
		);
		if !matches!(&self.inner, ValueInner::CppOwned { drop: false, .. }) {
			ortsys![unsafe ReleaseValue(ptr)];
		}
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for Value
where
	'i: 'v
{
	type Error = Error;
	fn try_from(arr: &'i CowArray<'v, T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for Value {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

macro_rules! impl_try_from {
	(@T,I $($t:ty),+) => {
		$(
			impl<T: IntoTensorElementType + Debug + Clone + 'static, I: ToDimensions> TryFrom<$t> for Value {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Value::from_array(value)
				}
			}
		)+
	};
	(@T,D $($t:ty),+) => {
		$(
			#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
			impl<T: IntoTensorElementType + Debug + Clone + 'static, D: ndarray::Dimension + 'static> TryFrom<$t> for Value {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Value::from_array(value)
				}
			}
		)+
	};
}

#[cfg(feature = "ndarray")]
impl_try_from!(@T,D &mut ArcArray<T, D>, Array<T, D>);
impl_try_from!(@T,I (I, Arc<Box<[T]>>), (I, Vec<T>), (I, Box<[T]>), (I, &[T]));

pub(crate) unsafe fn extract_data_type_from_tensor_info(info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo) -> Result<ValueType> {
	let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(info_ptr, &mut type_sys) -> Error::GetTensorElementType];
	assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	let mut num_dims = 0;
	ortsys![GetDimensionsCount(info_ptr, &mut num_dims) -> Error::GetDimensionsCount];

	let mut node_dims: Vec<i64> = vec![0; num_dims as _];
	ortsys![GetDimensions(info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> Error::GetDimensions];

	Ok(ValueType::Tensor {
		ty: type_sys.into(),
		dimensions: node_dims
	})
}

pub(crate) unsafe fn extract_data_type_from_sequence_info(info_ptr: *const ort_sys::OrtSequenceTypeInfo) -> Result<ValueType> {
	let mut element_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
	ortsys![GetSequenceElementType(info_ptr, &mut element_type_info) -> Error::GetSequenceElementType];

	let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
	let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(element_type_info, &mut ty)];
	status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;

	match ty {
		ort_sys::ONNXType::ONNX_TYPE_TENSOR => {
			let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
			ortsys![unsafe CastTypeInfoToTensorInfo(element_type_info, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
			let ty = unsafe { extract_data_type_from_tensor_info(info_ptr)? };
			Ok(ValueType::Sequence(Box::new(ty)))
		}
		ort_sys::ONNXType::ONNX_TYPE_MAP => {
			let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
			ortsys![unsafe CastTypeInfoToMapTypeInfo(element_type_info, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
			let ty = unsafe { extract_data_type_from_map_info(info_ptr)? };
			Ok(ValueType::Sequence(Box::new(ty)))
		}
		_ => unreachable!()
	}
}

pub(crate) unsafe fn extract_data_type_from_map_info(info_ptr: *const ort_sys::OrtMapTypeInfo) -> Result<ValueType> {
	let mut key_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetMapKeyType(info_ptr, &mut key_type_sys) -> Error::GetMapKeyType];
	assert_ne!(key_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	let mut value_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
	ortsys![GetMapValueType(info_ptr, &mut value_type_info) -> Error::GetMapValueType];
	let mut value_info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
	ortsys![unsafe CastTypeInfoToTensorInfo(value_type_info, &mut value_info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(value_info_ptr)];
	let mut value_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(value_info_ptr, &mut value_type_sys) -> Error::GetTensorElementType];
	assert_ne!(value_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	Ok(ValueType::Map {
		key: key_type_sys.into(),
		value: value_type_sys.into()
	})
}
