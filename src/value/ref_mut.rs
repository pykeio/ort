use alloc::{format, sync::Arc};
use core::{marker::PhantomData, mem::transmute, ops::Deref};

use super::{DowncastableTarget, DynValueTypeMarker, Value, ValueInner, ValueTypeMarker, format_value_type, r#ref::ValueRef};
use crate::error::{Error, ErrorCode, Result};

/// A mutable temporary version of a [`Value`] with a lifetime specifier.
#[derive(Debug)]
pub struct ValueRefMut<'v, Type: ValueTypeMarker + ?Sized = DynValueTypeMarker> {
	pub(crate) inner: Value<Type>,
	lifetime: PhantomData<&'v ()>
}

pub type DynValueRefMut<'a> = ValueRefMut<'a, DynValueTypeMarker>;

impl<'v, Type: ValueTypeMarker + ?Sized> ValueRefMut<'v, Type> {
	pub(crate) fn new(inner: Value<Type>) -> Self {
		ValueRefMut { inner, lifetime: PhantomData }
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
		if Arc::strong_count(&self.inner.inner) != 1 || !self.inner.inner.flags.upgradable() {
			return Err(self);
		}

		Ok(self.inner)
	}

	pub fn into_dyn(self) -> ValueRefMut<'v, DynValueTypeMarker> {
		unsafe { transmute(self) }
	}
}

impl<'a, Type: ValueTypeMarker + ?Sized> From<ValueRefMut<'a, Type>> for ValueRef<'a, Type> {
	fn from(ref_mut: ValueRefMut<'a, Type>) -> Self {
		ValueRef::new(ref_mut.inner)
	}
}

impl<Type: ValueTypeMarker + ?Sized> Deref for ValueRefMut<'_, Type> {
	type Target = Value<Type>;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

pub trait AsValueMut<Type: ValueTypeMarker + ?Sized> {
	#[doc(hidden)]
	#[allow(private_interfaces)]
	fn value_inner(&mut self) -> &Arc<ValueInner>;

	private_trait!();
}

impl<Type: ValueTypeMarker + ?Sized> AsValueMut<Type> for Value<Type> {
	#[allow(private_interfaces)]
	fn value_inner(&mut self) -> &Arc<ValueInner> {
		&self.inner
	}

	private_impl!();
}

impl<Type: ValueTypeMarker + ?Sized> AsValueMut<Type> for &mut Value<Type> {
	#[allow(private_interfaces)]
	fn value_inner(&mut self) -> &Arc<ValueInner> {
		&self.inner
	}

	private_impl!();
}

impl<Type: ValueTypeMarker + ?Sized> AsValueMut<Type> for ValueRefMut<'_, Type> {
	#[allow(private_interfaces)]
	fn value_inner(&mut self) -> &Arc<ValueInner> {
		&self.inner.inner
	}

	private_impl!();
}

macro_rules! mut_methods {
	(impl[$($bounds:tt)*] $ty:ty {
		$(
			$(#[$m:meta])*
			$vis:vis fn $name:ident $([$($fn_bounds:tt)*])? (&mut $self:ident $(, $arg:ident: $arg_ty:ty)*) $(-> $ret:ty)?
			$body:block
		)*
	}) => {
		impl<$($bounds)*> $crate::value::Value<$ty> {
			$(
				$(#[$m])*
				$vis fn $name $(<$($fn_bounds)*>)? (&mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				$body
			)*
		}

		impl<'v, $($bounds)*> $crate::value::ValueRefMut<'v, $ty> {
			$(
				$(#[$m])*
				$vis fn $name $(<$($fn_bounds)*>)? (&mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				{
					$self.inner.$name($($arg),*)
				}
			)*
		}
	};
}
pub(crate) use mut_methods;
