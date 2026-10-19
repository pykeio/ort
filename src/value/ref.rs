use alloc::{format, sync::Arc};
use core::{marker::PhantomData, mem::transmute, ops::Deref};

use super::{DowncastableTarget, DynValueTypeMarker, Value, ValueTypeMarker, format_value_type};
use crate::error::{Error, ErrorCode, Result};

/// A temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRef<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	inner: Value<Type>,
	lifetime: PhantomData<&'v ()>
}

pub type DynValueRef<'a> = ValueRef<'a, DynValueTypeMarker>;

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRef<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRef { inner, lifetime: PhantomData }
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
		if Arc::strong_count(&self.inner.inner) != 1 || !self.inner.inner.flags.upgradable() {
			return Err(self);
		}

		Ok(self.inner)
	}

	pub fn into_dyn(self) -> DynValueRef<'v> {
		unsafe { transmute(self) }
	}
}

impl<Type: ValueTypeMarker + ?Sized> Deref for ValueRef<'_, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}
