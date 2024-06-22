use std::{
	fmt::Debug,
	marker::PhantomData,
	ptr::{self, NonNull},
	sync::Arc
};

use super::{DowncastableTarget, ValueInner, ValueTypeMarker};
use crate::{memory::Allocator, ortsys, Error, Result, Value, ValueRef, ValueRefMut, ValueType};

pub trait SequenceValueTypeMarker: ValueTypeMarker {
	crate::private_trait!();
}

#[derive(Debug)]
pub struct DynSequenceValueType;
impl ValueTypeMarker for DynSequenceValueType {
	crate::private_impl!();
}
impl SequenceValueTypeMarker for DynSequenceValueType {
	crate::private_impl!();
}

#[derive(Debug)]
pub struct SequenceValueType<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized>(PhantomData<T>);
impl<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized> ValueTypeMarker for SequenceValueType<T> {
	crate::private_impl!();
}
impl<T: ValueTypeMarker + DowncastableTarget + Debug + ?Sized> SequenceValueTypeMarker for SequenceValueType<T> {
	crate::private_impl!();
}

pub type DynSequence = Value<DynSequenceValueType>;
pub type Sequence<T> = Value<SequenceValueType<T>>;

pub type DynSequenceRef<'v> = ValueRef<'v, DynSequenceValueType>;
pub type DynSequenceRefMut<'v> = ValueRefMut<'v, DynSequenceValueType>;
pub type SequenceRef<'v, T> = ValueRef<'v, SequenceValueType<T>>;
pub type SequenceRefMut<'v, T> = ValueRefMut<'v, SequenceValueType<T>>;

impl<Type: SequenceValueTypeMarker + Sized> Value<Type> {
	pub fn try_extract_sequence<'s, OtherType: ValueTypeMarker + DowncastableTarget + Debug + Sized>(
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
					if !OtherType::can_downcast(&value.dtype()?) {
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

impl<T: ValueTypeMarker + DowncastableTarget + Debug + Sized + 'static> Value<SequenceValueType<T>> {
	/// Creates a [`Sequence`] from an array of [`Value<T>`].
	///
	/// This `Value<T>` must be either a [`crate::Tensor`] or [`crate::Map`].
	///
	/// ```
	/// # use ort::{Allocator, Sequence, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let tensor1 = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
	/// let tensor2 = Tensor::<f32>::new(&allocator, [1, 224, 224, 3])?;
	/// let value = Sequence::new([tensor1, tensor2])?;
	///
	/// for tensor in value.extract_sequence(&allocator) {
	/// 	println!("{:?}", tensor.shape()?);
	/// }
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(values: impl IntoIterator<Item = Value<T>>) -> Result<Self> {
		let mut value_ptr = ptr::null_mut();
		let values: Vec<Value<T>> = values.into_iter().collect();
		let value_ptrs: Vec<*const ort_sys::OrtValue> = values.iter().map(|c| c.ptr().cast_const()).collect();
		ortsys![
			unsafe CreateValue(value_ptrs.as_ptr(), values.len() as _, ort_sys::ONNXType::ONNX_TYPE_SEQUENCE, &mut value_ptr)
				-> Error::CreateSequence;
			nonNull(value_ptr)
		];
		Ok(Value {
			inner: Arc::new(ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(values),
				_memory_info: None
			}),
			_markers: PhantomData
		})
	}
}

impl<T: ValueTypeMarker + DowncastableTarget + Debug + Sized> Value<SequenceValueType<T>> {
	pub fn extract_sequence<'s>(&'s self, allocator: &Allocator) -> Vec<ValueRef<'s, T>> {
		self.try_extract_sequence(allocator).expect("Failed to extract sequence")
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast(self) -> DynSequence {
		unsafe { std::mem::transmute(self) }
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a reference to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast_ref(&self) -> DynSequenceRef {
		DynSequenceRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Sequence<T>`] to a mutable reference to a type-erased [`DynSequence`].
	#[inline]
	pub fn upcast_mut(&mut self) -> DynSequenceRefMut {
		DynSequenceRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}
}
