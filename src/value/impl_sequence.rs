use std::{
	fmt::Debug,
	marker::PhantomData,
	ptr::{self, NonNull}
};

use super::{UpcastableTarget, ValueInner, ValueTypeMarker};
use crate::{memory::Allocator, ortsys, Error, Result, Value, ValueRef, ValueRefMut, ValueType};

pub trait SequenceValueTypeMarker: ValueTypeMarker {}

#[derive(Debug)]
pub struct DynSequenceValueType;
impl ValueTypeMarker for DynSequenceValueType {}
impl SequenceValueTypeMarker for DynSequenceValueType {}

#[derive(Debug)]
pub struct SequenceValueType<T: ValueTypeMarker + UpcastableTarget + Debug + ?Sized>(PhantomData<T>);
impl<T: ValueTypeMarker + UpcastableTarget + Debug + ?Sized> ValueTypeMarker for SequenceValueType<T> {}
impl<T: ValueTypeMarker + UpcastableTarget + Debug + ?Sized> SequenceValueTypeMarker for SequenceValueType<T> {}

pub type DynSequence = Value<DynSequenceValueType>;
pub type Sequence<T> = Value<SequenceValueType<T>>;

pub type DynSequenceRef<'v> = ValueRef<'v, DynSequenceValueType>;
pub type DynSequenceRefMut<'v> = ValueRefMut<'v, DynSequenceValueType>;
pub type SequenceRef<'v, T> = ValueRef<'v, SequenceValueType<T>>;
pub type SequenceRefMut<'v, T> = ValueRefMut<'v, SequenceValueType<T>>;

impl<Type: SequenceValueTypeMarker + ?Sized> Value<Type> {
	pub fn try_extract_sequence<'s, OtherType: ValueTypeMarker + UpcastableTarget + Debug + ?Sized>(
		&'s self,
		allocator: &Allocator
	) -> Result<Vec<ValueRef<'s, OtherType>>> {
		match self.dtype()? {
			ValueType::Sequence(_) => {
				let mut len: ort_sys::size_t = 0;
				ortsys![unsafe GetValueCount(self.ptr(), &mut len) -> Error::ExtractSequence];

				let mut vec = Vec::with_capacity(len as usize);
				for i in 0..len {
					let mut value_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), i as _, allocator.ptr.as_ptr(), &mut value_ptr) -> Error::ExtractSequence; nonNull(value_ptr)];

					let value = ValueRef {
						inner: unsafe { Value::from_ptr(NonNull::new_unchecked(value_ptr), None) },
						lifetime: PhantomData
					};
					let value_type = value.dtype()?;
					if !OtherType::can_upcast(&value.dtype()?) {
						return Err(Error::InvalidSequenceElementType { actual: value_type });
					}

					vec.push(value);
				}
				Ok(vec)
			}
			t => Err(Error::NotSequence(t))
		}
	}
}

impl<T: ValueTypeMarker + UpcastableTarget + Debug + ?Sized> Value<SequenceValueType<T>> {
	pub fn extract_sequence<'s>(&'s self, allocator: &Allocator) -> Vec<ValueRef<'s, T>> {
		self.try_extract_sequence(allocator).unwrap()
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a type-erased [`DynSequence`].
	#[inline]
	pub fn downcast(self) -> DynSequence {
		unsafe { std::mem::transmute(self) }
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a reference to a type-erased [`DynTensor`].
	#[inline]
	pub fn downcast_ref(&self) -> DynSequenceRef {
		DynSequenceRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a mutable reference to a type-erased [`DynTensor`].
	#[inline]
	pub fn downcast_mut(&mut self) -> DynSequenceRefMut {
		DynSequenceRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}
}
