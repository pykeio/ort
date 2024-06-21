use std::{
	any::Any,
	fmt::Debug,
	marker::PhantomData,
	ops::{Deref, DerefMut},
	ptr::NonNull,
	sync::Arc
};

mod impl_map;
mod impl_sequence;
mod impl_tensor;

pub use self::{
	impl_map::{DynMap, DynMapRef, DynMapRefMut, DynMapValueType, Map, MapRef, MapRefMut, MapValueType, MapValueTypeMarker},
	impl_sequence::{
		DynSequence, DynSequenceRef, DynSequenceRefMut, DynSequenceValueType, Sequence, SequenceRef, SequenceRefMut, SequenceValueType, SequenceValueTypeMarker
	},
	impl_tensor::{DynTensor, DynTensorRef, DynTensorRefMut, DynTensorValueType, Tensor, TensorRef, TensorRefMut, TensorValueTypeMarker}
};
use crate::{error::status_to_result, memory::MemoryInfo, ortsys, session::SharedSessionInner, tensor::TensorElementType, Error, Result};

/// The type of a [`Value`], or a session input/output.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{Session, Tensor, ValueType, TensorElementType};
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
/// let value = Tensor::from_array(([5usize], vec![1_i64, 2, 3, 4, 5].into_boxed_slice()))?;
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
		/// Hold onto the `MemoryInfo` that we create in `Value::from_array`.
		_memory_info: Option<MemoryInfo>
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

impl ValueInner {
	pub(crate) fn ptr(&self) -> *mut ort_sys::OrtValue {
		match self {
			ValueInner::CppOwned { ptr, .. } | ValueInner::RustOwned { ptr, .. } => ptr.as_ptr()
		}
	}
}

/// A temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRef<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	inner: Value<Type>,
	lifetime: PhantomData<&'v ()>
}

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRef<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRef { inner, lifetime: PhantomData }
	}

	/// Attempts to downcast a temporary dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed
	/// variant, like [`TensorRef<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(self) -> Result<ValueRef<'v, OtherType>> {
		let dt = self.dtype()?;
		if OtherType::can_downcast(&dt) { Ok(unsafe { std::mem::transmute(self) }) } else { panic!() }
	}

	pub fn into_dyn(self) -> ValueRef<'v, DynValueTypeMarker> {
		unsafe { std::mem::transmute(self) }
	}
}

impl<'v, Type: ValueTypeMarker + ?Sized> Deref for ValueRef<'v, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

/// A mutable temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRefMut<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	inner: Value<Type>,
	lifetime: PhantomData<&'v ()>
}

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRefMut<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRefMut { inner, lifetime: PhantomData }
	}

	/// Attempts to downcast a temporary mutable dynamic value (like [`DynValue`] or [`DynTensor`]) to a more
	/// strongly typed variant, like [`TensorRefMut<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(self) -> Result<ValueRefMut<'v, OtherType>> {
		let dt = self.dtype()?;
		if OtherType::can_downcast(&dt) { Ok(unsafe { std::mem::transmute(self) }) } else { panic!() }
	}

	pub fn into_dyn(self) -> ValueRefMut<'v, DynValueTypeMarker> {
		unsafe { std::mem::transmute(self) }
	}
}

impl<'v, Type: ValueTypeMarker + ?Sized> Deref for ValueRefMut<'v, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<'v, Type: ValueTypeMarker + ?Sized> DerefMut for ValueRefMut<'v, Type> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

/// A [`Value`] contains data for inputs/outputs in ONNX Runtime graphs. [`Value`]s can be a [`Tensor`], [`Sequence`]
/// (aka array/vector), or [`Map`].
///
/// ## Creation
/// Values can be created via methods like [`Tensor::from_array`], or as the output from running a [`crate::Session`].
///
/// ```
/// # use ort::{Session, Tensor, ValueType, TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let upsample = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // Create a Tensor value from a raw data vector
/// let value = Tensor::from_array(([1usize, 1, 1, 3], vec![1.0_f32, 2.0, 3.0].into_boxed_slice()))?;
///
/// // Create a Tensor value from an `ndarray::Array`
/// #[cfg(feature = "ndarray")]
/// let value = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
///
/// // Get a DynValue from a session's output
/// let value = &upsample.run(ort::inputs![value]?)?[0];
/// # 	Ok(())
/// # }
/// ```
///
/// See [`Tensor::from_array`] for more details on what tensor values are accepted.
///
/// ## Usage
/// You can access the data contained in a `Value` by using the relevant `extract` methods.
/// You can also use [`DynValue::downcast`] to attempt to convert from a [`DynValue`] to a more strongly typed value.
///
/// For dynamic values, where the type is not known at compile time, see the `try_extract_*` methods:
/// - [`Tensor::try_extract_tensor`], [`Tensor::try_extract_raw_tensor`]
/// - [`Sequence::try_extract_sequence`]
/// - [`Map::try_extract_map`]
///
/// If the type was created from Rust (via a method like [`Tensor::from_array`] or via downcasting), you can directly
/// extract the data using the infallible extract methods:
/// - [`Tensor::extract_tensor`], [`Tensor::extract_raw_tensor`]
#[derive(Debug)]
pub struct Value<Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	pub(crate) inner: Arc<ValueInner>,
	pub(crate) _markers: PhantomData<Type>
}

/// A dynamic value, which could be a [`Tensor`], [`Sequence`], or [`Map`].
///
/// To attempt to convert a dynamic value to a strongly typed value, use [`DynValue::downcast`]. You can also attempt to
/// extract data from dynamic values directly using `try_extract_*` methods; see [`Value`] for more information.
pub type DynValue = Value<DynValueTypeMarker>;

/// Marker trait used to determine what operations can and cannot be performed on a [`Value`] of a given type.
///
/// For example, [`Tensor::try_extract_tensor`] can only be used on [`Value`]s with the [`TensorValueTypeMarker`] (which
/// inherits this trait), i.e. [`Tensor`]s, [`DynTensor`]s, and [`DynValue`]s.
pub trait ValueTypeMarker: Debug {
	crate::private_trait!();
}

/// Represents a type that a [`DynValue`] can be downcast to.
pub trait DowncastableTarget: ValueTypeMarker {
	fn can_downcast(dtype: &ValueType) -> bool;

	crate::private_trait!();
}

// this implementation is used in case we want to extract `DynValue`s from a [`Sequence`]; see `try_extract_sequence`
impl DowncastableTarget for DynValueTypeMarker {
	fn can_downcast(_: &ValueType) -> bool {
		true
	}

	crate::private_impl!();
}

/// The dynamic type marker, used for values which can be of any type.
#[derive(Debug)]
pub struct DynValueTypeMarker;
impl ValueTypeMarker for DynValueTypeMarker {
	crate::private_impl!();
}
impl MapValueTypeMarker for DynValueTypeMarker {
	crate::private_impl!();
}
impl SequenceValueTypeMarker for DynValueTypeMarker {
	crate::private_impl!();
}
impl TensorValueTypeMarker for DynValueTypeMarker {
	crate::private_impl!();
}

unsafe impl Send for Value {}

impl<Type: ValueTypeMarker + ?Sized> Value<Type> {
	/// Returns the data type of this [`Value`].
	pub fn dtype(&self) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTypeInfo(self.ptr(), &mut typeinfo_ptr) -> Error::GetTypeInfo; nonNull(typeinfo_ptr)];

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty) -> Error::GetOnnxTypeFromTypeInfo];
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
	/// [`crate::Session::inner`]). This ensures the session is not dropped until any values owned by it is.
	///
	/// # Safety
	///
	/// - `ptr` must be a valid pointer to an [`ort_sys::OrtValue`].
	/// - `session` must be `Some` for values returned from a session.
	#[must_use]
	pub unsafe fn from_ptr(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value<Type> {
		Value {
			inner: Arc::new(ValueInner::CppOwned { ptr, drop: true, _session: session }),
			_markers: PhantomData
		}
	}

	/// A variant of [`Value::from_ptr`] that does not release the value upon dropping. Used in operator kernel
	/// contexts.
	#[must_use]
	pub(crate) unsafe fn from_ptr_nodrop(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value<Type> {
		Value {
			inner: Arc::new(ValueInner::CppOwned { ptr, drop: false, _session: session }),
			_markers: PhantomData
		}
	}

	/// Returns the underlying [`ort_sys::OrtValue`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtValue {
		self.inner.ptr()
	}

	/// Create a view of this value's data.
	pub fn view(&self) -> ValueRef<'_, Type> {
		ValueRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Create a mutable view of this value's data.
	pub fn view_mut(&mut self) -> ValueRefMut<'_, Type> {
		ValueRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
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

	/// Converts this value into a type-erased [`DynValue`].
	pub fn into_dyn(self) -> DynValue {
		unsafe { std::mem::transmute(self) }
	}
}

impl Value<DynValueTypeMarker> {
	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed variant,
	/// like [`Tensor<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(self) -> Result<Value<OtherType>> {
		let dt = self.dtype()?;
		if OtherType::can_downcast(&dt) { Ok(unsafe { std::mem::transmute(self) }) } else { panic!() }
	}

	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed reference
	/// variant, like [`TensorRef<T>`].
	#[inline]
	pub fn downcast_ref<OtherType: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(&self) -> Result<ValueRef<'_, OtherType>> {
		let dt = self.dtype()?;
		if OtherType::can_downcast(&dt) {
			Ok(ValueRef::new(unsafe {
				Value::from_ptr_nodrop(
					NonNull::new_unchecked(self.ptr()),
					if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
				)
			}))
		} else {
			panic!()
		}
	}

	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed
	/// mutable-reference variant, like [`TensorRefMut<T>`].
	#[inline]
	pub fn downcast_mut<OtherType: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(&mut self) -> Result<ValueRefMut<'_, OtherType>> {
		let dt = self.dtype()?;
		if OtherType::can_downcast(&dt) {
			Ok(ValueRefMut::new(unsafe {
				Value::from_ptr_nodrop(
					NonNull::new_unchecked(self.ptr()),
					if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
				)
			}))
		} else {
			panic!()
		}
	}
}

impl Drop for ValueInner {
	fn drop(&mut self) {
		let ptr = self.ptr();
		tracing::trace!(
			"dropping {} value at {ptr:p}",
			match self {
				ValueInner::RustOwned { .. } => "rust-owned",
				ValueInner::CppOwned { .. } => "cpp-owned"
			}
		);
		if !matches!(self, ValueInner::CppOwned { drop: false, .. }) {
			ortsys![unsafe ReleaseValue(ptr)];
		}
	}
}

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
