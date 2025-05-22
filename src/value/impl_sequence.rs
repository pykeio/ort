use alloc::{boxed::Box, format, sync::Arc, vec::Vec};
use core::{
	fmt::{self, Debug, Display},
	marker::PhantomData,
	ptr::{self}
};

use super::{DowncastableTarget, Value, ValueInner, ValueRef, ValueRefMut, ValueType, ValueTypeMarker, format_value_type};
use crate::{
	AsPointer, ErrorCode,
	error::{Error, Result},
	memory::Allocator,
	ortsys
};

pub trait SequenceValueTypeMarker: ValueTypeMarker {
	private_trait!();
}

#[derive(Debug)]
pub struct DynSequenceValueType;
impl ValueTypeMarker for DynSequenceValueType {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("DynSequence")
	}

	private_impl!();
}
impl SequenceValueTypeMarker for DynSequenceValueType {
	private_impl!();
}

impl DowncastableTarget for DynSequenceValueType {
	fn can_downcast(dtype: &ValueType) -> bool {
		matches!(dtype, ValueType::Sequence { .. })
	}

	private_impl!();
}

#[derive(Debug)]
pub struct SequenceValueType<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(PhantomData<T>);
impl<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized> ValueTypeMarker for SequenceValueType<T> {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("Sequence<")?;
		format_value_type::<T>().fmt(f)?;
		f.write_str(">")
	}

	private_impl!();
}
impl<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized> SequenceValueTypeMarker for SequenceValueType<T> {
	private_impl!();
}

impl<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized> DowncastableTarget for SequenceValueType<T> {
	fn can_downcast(dtype: &ValueType) -> bool {
		match dtype {
			ValueType::Sequence(ty) => T::can_downcast(ty),
			_ => false
		}
	}

	private_impl!();
}

pub type DynSequence = Value<DynSequenceValueType>;
pub type Sequence<T> = Value<SequenceValueType<T>>;

pub type DynSequenceRef<'v> = ValueRef<'v, DynSequenceValueType>;
pub type DynSequenceRefMut<'v> = ValueRefMut<'v, DynSequenceValueType>;
pub type SequenceRef<'v, T> = ValueRef<'v, SequenceValueType<T>>;
pub type SequenceRefMut<'v, T> = ValueRefMut<'v, SequenceValueType<T>>;

impl<Type: SequenceValueTypeMarker + Sized> Value<Type> {
	pub fn try_extract_sequence<OtherType: ValueTypeMarker + DowncastableTarget + Debug + Sized>(
		&self,
		allocator: &Allocator
	) -> Result<Vec<ValueRef<'_, OtherType>>> {
		match self.dtype() {
			ValueType::Sequence(_) => {
				let mut len = 0;
				ortsys![unsafe GetValueCount(self.ptr(), &mut len)?];

				let mut vec = Vec::with_capacity(len);
				for i in 0..len {
					let mut value_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), i as _, allocator.ptr().cast_mut(), &mut value_ptr)?; nonNull(value_ptr)];

					let mut value = ValueRef::new(unsafe { Value::from_ptr(value_ptr, None) });
					value.upgradable = false;

					let value_type = value.dtype();
					if !OtherType::can_downcast(value.dtype()) {
						return Err(Error::new_with_code(
							ErrorCode::InvalidArgument,
							format!("Cannot extract Sequence<{}> from {value_type:?}", format_value_type::<OtherType>())
						));
					}

					vec.push(value);
				}
				Ok(vec)
			}
			t => Err(Error::new(format!("Cannot extract Sequence<{}> from {t}", format_value_type::<OtherType>())))
		}
	}
}

impl<T: ValueTypeMarker + DowncastableTarget + Debug + Sized + 'static> Value<SequenceValueType<T>> {
	/// Creates a [`Sequence`] from an array of [`Value<T>`].
	///
	/// This `Value<T>` must be either a [`Tensor`] or [`Map`].
	///
	/// ```
	/// # use ort::{memory::Allocator, value::{Sequence, Tensor}};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let tensor1 = Tensor::<f32>::new(&allocator, [1_usize, 128, 128, 3])?;
	/// let tensor2 = Tensor::<f32>::new(&allocator, [1_usize, 224, 224, 3])?;
	/// let value = Sequence::new([tensor1, tensor2])?;
	///
	/// for tensor in value.extract_sequence(&allocator) {
	/// 	println!("{:?}", tensor.shape());
	/// }
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// [`Tensor`]: crate::value::Tensor
	/// [`Map`]: crate::value::Map
	pub fn new(values: impl IntoIterator<Item = Value<T>>) -> Result<Self> {
		let mut value_ptr = ptr::null_mut();
		let values: Vec<Value<T>> = values.into_iter().collect();
		let value_ptrs: Vec<*const ort_sys::OrtValue> = values.iter().map(|c| c.ptr()).collect();
		ortsys![
			unsafe CreateValue(value_ptrs.as_ptr(), values.len(), ort_sys::ONNXType::ONNX_TYPE_SEQUENCE, &mut value_ptr)?;
			nonNull(value_ptr)
		];
		Ok(Value {
			inner: Arc::new(ValueInner {
				ptr: value_ptr,
				// 1. `CreateValue` enforces that we have at least 1 value
				// 2. `CreateValue` internally uses the first value to determine the element type, so we do the same here
				dtype: ValueType::Sequence(Box::new(values[0].inner.dtype.clone())),
				drop: true,
				memory_info: None,
				_backing: Some(Box::new(values))
			}),
			_markers: PhantomData
		})
	}
}

impl<T: ValueTypeMarker + DowncastableTarget + Debug + Sized> Value<SequenceValueType<T>> {
	pub fn extract_sequence(&self, allocator: &Allocator) -> Vec<ValueRef<'_, T>> {
		self.try_extract_sequence(allocator).expect("Failed to extract sequence")
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast(self) -> DynSequence {
		unsafe { self.transmute_type() }
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a reference to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast_ref(&self) -> DynSequenceRef {
		DynSequenceRef::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a mutable reference to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast_mut(&mut self) -> DynSequenceRefMut {
		DynSequenceRefMut::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}
}
