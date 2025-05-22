//! [`Value`]s are data containers used as inputs/outputs in ONNX Runtime graphs.
//!
//! The most common type of value is [`Tensor`]:
//! ```
//! # use ort::value::Tensor;
//! # fn main() -> ort::Result<()> {
//! // Create a tensor from a raw data vector
//! let tensor = Tensor::from_array(([1usize, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice()))?;
//!
//! // Create a tensor from an `ndarray::Array`
//! #[cfg(feature = "ndarray")]
//! let tensor = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
//! # 	Ok(())
//! # }
//! ```
//!
//! ONNX Runtime also supports [`Sequence`]s and [`Map`]s, though they are less commonly used.

use alloc::{boxed::Box, format, sync::Arc};
use core::{
	any::Any,
	fmt::{self, Debug},
	marker::PhantomData,
	mem::transmute,
	ops::{Deref, DerefMut},
	ptr::{self, NonNull}
};

mod impl_map;
mod impl_sequence;
mod impl_tensor;
pub(crate) mod r#type;

pub use self::{
	impl_map::{DynMap, DynMapRef, DynMapRefMut, DynMapValueType, Map, MapRef, MapRefMut, MapValueType, MapValueTypeMarker},
	impl_sequence::{
		DynSequence, DynSequenceRef, DynSequenceRefMut, DynSequenceValueType, Sequence, SequenceRef, SequenceRefMut, SequenceValueType, SequenceValueTypeMarker
	},
	impl_tensor::{
		DefiniteTensorValueTypeMarker, DynTensor, DynTensorRef, DynTensorRefMut, DynTensorValueType, OwnedTensorArrayData, Tensor, TensorArrayData,
		TensorArrayDataMut, TensorArrayDataParts, TensorRef, TensorRefMut, TensorValueType, TensorValueTypeMarker, ToShape
	},
	r#type::ValueType
};
use crate::{
	AsPointer,
	error::{Error, ErrorCode, Result},
	memory::MemoryInfo,
	ortsys,
	session::SharedSessionInner
};

#[derive(Debug)]
pub(crate) struct ValueInner {
	pub(crate) ptr: NonNull<ort_sys::OrtValue>,
	pub(crate) dtype: ValueType,
	pub(crate) memory_info: Option<MemoryInfo>,
	pub(crate) drop: bool,
	pub(crate) _backing: Option<Box<dyn Any>>
}

impl AsPointer for ValueInner {
	type Sys = ort_sys::OrtValue;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for ValueInner {
	fn drop(&mut self) {
		if self.drop {
			let ptr = self.ptr_mut();
			crate::trace!("dropping value at {ptr:p}");
			ortsys![unsafe ReleaseValue(ptr)];
		}
	}
}

/// A temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRef<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	inner: Value<Type>,
	pub(crate) upgradable: bool,
	lifetime: PhantomData<&'v ()>
}

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRef<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRef {
			// We cannot upgade a value which we cannot drop, i.e. `ValueRef`s used in operator kernels. Those only last for the
			// duration of the kernel, allowing an upgrade would allow a UAF.
			upgradable: inner.inner.drop,
			inner,
			lifetime: PhantomData
		}
	}

	/// Attempts to downcast a temporary dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed
	/// variant, like [`TensorRef<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + ?Sized>(self) -> Result<ValueRef<'v, OtherType>> {
		let dt = self.dtype();
		if OtherType::can_downcast(dt) {
			Ok(unsafe { transmute::<ValueRef<'v, Type>, ValueRef<'v, OtherType>>(self) })
		} else {
			Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot downcast &{dt} to &{}", format_value_type::<OtherType>())))
		}
	}

	/// Attempts to upgrade this `ValueRef` to an owned [`Value`] holding the same data.
	pub fn try_upgrade(self) -> Result<Value<Type>, Self> {
		if !self.upgradable {
			return Err(self);
		}

		Ok(self.inner)
	}

	pub fn into_dyn(self) -> ValueRef<'v, DynValueTypeMarker> {
		unsafe { transmute(self) }
	}
}

impl<Type: ValueTypeMarker + ?Sized> Deref for ValueRef<'_, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

/// A mutable temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRefMut<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	inner: Value<Type>,
	pub(crate) upgradable: bool,
	lifetime: PhantomData<&'v ()>
}

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRefMut<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRefMut {
			// We cannot upgade a value which we cannot drop, i.e. `ValueRef`s used in operator kernels. Those only last for the
			// duration of the kernel, allowing an upgrade would allow a UAF.
			upgradable: inner.inner.drop,
			inner,
			lifetime: PhantomData
		}
	}

	/// Attempts to downcast a temporary mutable dynamic value (like [`DynValue`] or [`DynTensor`]) to a more
	/// strongly typed variant, like [`TensorRefMut<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + ?Sized>(self) -> Result<ValueRefMut<'v, OtherType>> {
		let dt = self.dtype();
		if OtherType::can_downcast(dt) {
			Ok(unsafe { transmute::<ValueRefMut<'v, Type>, ValueRefMut<'v, OtherType>>(self) })
		} else {
			Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot downcast &mut {dt} to &mut {}", format_value_type::<OtherType>())))
		}
	}

	/// Attempts to upgrade this `ValueRefMut` to an owned [`Value`] holding the same data.
	pub fn try_upgrade(self) -> Result<Value<Type>, Self> {
		if !self.upgradable {
			return Err(self);
		}

		Ok(self.inner)
	}

	pub fn into_dyn(self) -> ValueRefMut<'v, DynValueTypeMarker> {
		unsafe { transmute(self) }
	}
}

impl<Type: ValueTypeMarker + ?Sized> Deref for ValueRefMut<'_, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl<Type: ValueTypeMarker + ?Sized> DerefMut for ValueRefMut<'_, Type> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

/// A [`Value`] contains data for inputs/outputs in ONNX Runtime graphs. [`Value`]s can be a [`Tensor`], [`Sequence`]
/// (aka array/vector), or [`Map`].
///
/// ## Creation
/// Values can be created via methods like [`Tensor::from_array`], or as the output from running a [`Session`].
///
/// ```
/// # use ort::{session::Session, value::Tensor};
/// # fn main() -> ort::Result<()> {
/// # 	let mut upsample = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// // Create a Tensor value from a raw data vector
/// let value = Tensor::from_array(([1usize, 1, 1, 3], vec![1.0_f32, 2.0, 3.0].into_boxed_slice()))?;
///
/// // Create a Tensor value from an `ndarray::Array`
/// #[cfg(feature = "ndarray")]
/// let value = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
///
/// // Get a DynValue from a session's output
/// let value = &upsample.run(ort::inputs![value])?[0];
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
/// - [`Tensor::try_extract_tensor`], [`Tensor::try_extract_array`]
/// - [`Sequence::try_extract_sequence`]
/// - [`Map::try_extract_map`]
///
/// If the type was created from Rust (via a method like [`Tensor::from_array`] or via downcasting), you can directly
/// extract the data using the infallible extract methods:
/// - [`Tensor::extract_tensor`], [`Tensor::extract_array`]
///
/// [`Session`]: crate::session::Session
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
pub trait ValueTypeMarker {
	#[doc(hidden)]
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result;

	private_trait!();
}

pub(crate) struct ValueTypeFormatter<T: ?Sized>(PhantomData<T>);

#[inline]
pub(crate) fn format_value_type<T: ValueTypeMarker + ?Sized>() -> ValueTypeFormatter<T> {
	ValueTypeFormatter(PhantomData)
}

impl<T: ValueTypeMarker + ?Sized> fmt::Display for ValueTypeFormatter<T> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		<T as ValueTypeMarker>::fmt(f)
	}
}

/// Represents a type that a [`DynValue`] can be downcast to.
pub trait DowncastableTarget: ValueTypeMarker {
	fn can_downcast(dtype: &ValueType) -> bool;

	private_trait!();
}

// this implementation is used in case we want to extract `DynValue`s from a [`Sequence`]; see `try_extract_sequence`
impl DowncastableTarget for DynValueTypeMarker {
	fn can_downcast(_: &ValueType) -> bool {
		true
	}

	private_impl!();
}

/// The dynamic type marker, used for values which can be of any type.
#[derive(Debug)]
pub struct DynValueTypeMarker;
impl ValueTypeMarker for DynValueTypeMarker {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("DynValue")
	}

	private_impl!();
}
impl MapValueTypeMarker for DynValueTypeMarker {
	private_impl!();
}
impl SequenceValueTypeMarker for DynValueTypeMarker {
	private_impl!();
}
impl TensorValueTypeMarker for DynValueTypeMarker {
	private_impl!();
}

unsafe impl<Type: ValueTypeMarker + ?Sized> Send for Value<Type> {}
unsafe impl<Type: ValueTypeMarker + ?Sized> Sync for Value<Type> {}

impl<Type: ValueTypeMarker + ?Sized> Value<Type> {
	/// Returns the data type of this [`Value`].
	pub fn dtype(&self) -> &ValueType {
		&self.inner.dtype
	}

	/// Construct a [`Value`] from a C++ [`ort_sys::OrtValue`] pointer.
	///
	/// If the value belongs to a session (i.e. if it is the result of an inference run), you must provide the
	/// [`SharedSessionInner`] (acquired from [`Session::inner`](crate::session::Session::inner)). This ensures the
	/// session is not dropped until any values owned by it is.
	///
	/// # Safety
	///
	/// - `ptr` must be a valid pointer to an [`ort_sys::OrtValue`].
	/// - `session` must be `Some` for values returned from a session.
	#[must_use]
	pub unsafe fn from_ptr(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value<Type> {
		let mut typeinfo_ptr = ptr::null_mut();
		ortsys![unsafe GetTypeInfo(ptr.as_ptr(), &mut typeinfo_ptr).expect("infallible"); nonNull(typeinfo_ptr)];
		Value {
			inner: Arc::new(ValueInner {
				ptr,
				memory_info: unsafe { MemoryInfo::from_value(ptr) },
				dtype: unsafe { ValueType::from_type_info(typeinfo_ptr) },
				drop: true,
				_backing: session.map(|v| Box::new(v) as Box<dyn Any>)
			}),
			_markers: PhantomData
		}
	}

	/// A variant of [`Value::from_ptr`] that does not release the value upon dropping. Used in operator kernel
	/// contexts.
	#[must_use]
	pub(crate) unsafe fn from_ptr_nodrop(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value<Type> {
		let mut typeinfo_ptr = ptr::null_mut();
		ortsys![unsafe GetTypeInfo(ptr.as_ptr(), &mut typeinfo_ptr).expect("infallible"); nonNull(typeinfo_ptr)];
		Value {
			inner: Arc::new(ValueInner {
				ptr,
				memory_info: unsafe { MemoryInfo::from_value(ptr) },
				dtype: unsafe { ValueType::from_type_info(typeinfo_ptr) },
				drop: false,
				_backing: session.map(|v| Box::new(v) as Box<dyn Any>)
			}),
			_markers: PhantomData
		}
	}

	/// Create a view of this value's data.
	pub fn view(&self) -> ValueRef<'_, Type> {
		ValueRef::new(Value::clone_of(self))
	}

	/// Create a mutable view of this value's data.
	pub fn view_mut(&mut self) -> ValueRefMut<'_, Type> {
		ValueRefMut::new(Value::clone_of(self))
	}

	/// Converts this value into a type-erased [`DynValue`].
	pub fn into_dyn(self) -> DynValue {
		unsafe { self.transmute_type() }
	}

	/// Returns `true` if this value is a tensor, or `false` if it is another type (sequence, map).
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let tensor_value = Tensor::from_array(([3usize], vec![1.0_f32, 2.0, 3.0].into_boxed_slice()))?;
	/// let dyn_value = tensor_value.into_dyn();
	/// assert!(dyn_value.is_tensor());
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn is_tensor(&self) -> bool {
		let mut result = 0;
		ortsys![unsafe IsTensor(self.ptr(), &mut result).expect("infallible")];
		result == 1
	}

	#[inline(always)]
	pub(crate) unsafe fn transmute_type<OtherType: ValueTypeMarker + ?Sized>(self) -> Value<OtherType> {
		unsafe { transmute::<Value<Type>, Value<OtherType>>(self) }
	}

	#[inline(always)]
	pub(crate) unsafe fn transmute_type_ref<OtherType: ValueTypeMarker + ?Sized>(&self) -> &Value<OtherType> {
		unsafe { transmute::<&Value<Type>, &Value<OtherType>>(self) }
	}

	pub(crate) fn clone_of(value: &Self) -> Self {
		Self {
			inner: Arc::clone(&value.inner),
			_markers: PhantomData
		}
	}
}

impl Value<DynValueTypeMarker> {
	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed variant,
	/// like [`Tensor<T>`].
	#[inline]
	pub fn downcast<OtherType: ValueTypeMarker + DowncastableTarget + ?Sized>(self) -> Result<Value<OtherType>> {
		let dt = self.dtype();
		if OtherType::can_downcast(dt) {
			Ok(unsafe { transmute::<Value<DynValueTypeMarker>, Value<OtherType>>(self) })
		} else {
			Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot downcast {dt} to {}", format_value_type::<OtherType>())))
		}
	}

	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed reference
	/// variant, like [`TensorRef<T>`].
	#[inline]
	pub fn downcast_ref<OtherType: ValueTypeMarker + DowncastableTarget + ?Sized>(&self) -> Result<ValueRef<'_, OtherType>> {
		let dt = self.dtype();
		if OtherType::can_downcast(dt) {
			Ok(ValueRef::new(unsafe { transmute::<DynValue, Value<OtherType>>(Value::clone_of(self)) }))
		} else {
			Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot downcast &{dt} to &{}", format_value_type::<OtherType>())))
		}
	}

	/// Attempts to downcast a dynamic value (like [`DynValue`] or [`DynTensor`]) to a more strongly typed
	/// mutable-reference variant, like [`TensorRefMut<T>`].
	#[inline]
	pub fn downcast_mut<OtherType: ValueTypeMarker + DowncastableTarget + ?Sized>(&mut self) -> Result<ValueRefMut<'_, OtherType>> {
		let dt = self.dtype();
		if OtherType::can_downcast(dt) {
			Ok(ValueRefMut::new(unsafe { transmute::<DynValue, Value<OtherType>>(Value::clone_of(self)) }))
		} else {
			Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot downcast &mut {dt} to &mut {}", format_value_type::<OtherType>())))
		}
	}
}

impl<Type: ValueTypeMarker + ?Sized> AsPointer for Value<Type> {
	type Sys = ort_sys::OrtValue;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.ptr()
	}
}

#[cfg(test)]
mod tests {
	use super::{DynTensorValueType, Map, Sequence, Tensor, TensorRef, TensorRefMut, TensorValueType};
	use crate::memory::Allocator;

	#[test]
	fn test_casting_tensor() -> crate::Result<()> {
		let tensor: Tensor<i32> = Tensor::from_array((vec![5], vec![1, 2, 3, 4, 5]))?;

		let dyn_tensor = tensor.into_dyn();
		let mut tensor: Tensor<i32> = dyn_tensor.downcast()?;

		{
			let dyn_tensor_ref = tensor.view().into_dyn();
			let tensor_ref: TensorRef<i32> = dyn_tensor_ref.downcast()?;
			assert_eq!(tensor_ref.extract_tensor(), tensor.extract_tensor());
		}
		{
			let dyn_tensor_ref = tensor.view().into_dyn();
			let tensor_ref: TensorRef<i32> = dyn_tensor_ref.downcast_ref()?;
			assert_eq!(tensor_ref.extract_tensor(), tensor.extract_tensor());
		}

		// Ensure mutating a TensorRefMut mutates the original tensor.
		{
			let mut dyn_tensor_ref = tensor.view_mut().into_dyn();
			let mut tensor_ref: TensorRefMut<i32> = dyn_tensor_ref.downcast_mut()?;
			let (_, data) = tensor_ref.extract_tensor_mut();
			data[2] = 42;
		}
		{
			let (_, data) = tensor.extract_tensor_mut();
			assert_eq!(data[2], 42);
		}

		// chain a bunch of up/downcasts
		{
			let tensor = tensor
				.into_dyn()
				.downcast::<DynTensorValueType>()?
				.into_dyn()
				.downcast::<TensorValueType<i32>>()?
				.upcast()
				.into_dyn();
			let tensor = tensor.view();
			let tensor = tensor.downcast_ref::<TensorValueType<i32>>()?;
			let (_, data) = tensor.extract_tensor();
			assert_eq!(data, [1, 2, 42, 4, 5]);
		}

		Ok(())
	}

	#[test]
	fn test_sequence_map() -> crate::Result<()> {
		let map_contents = [("meaning".to_owned(), 42.0), ("pi".to_owned(), core::f32::consts::PI)];
		let value = Sequence::new([Map::<String, f32>::new(map_contents)?])?;

		for map in value.extract_sequence(&Allocator::default()) {
			let map = map.extract_key_values().into_iter().collect::<std::collections::HashMap<_, _>>();
			assert_eq!(map["meaning"], 42.0);
			assert_eq!(map["pi"], core::f32::consts::PI);
		}

		Ok(())
	}
}
