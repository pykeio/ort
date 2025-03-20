use alloc::{borrow::Cow, vec::Vec};
use core::ops::Deref;

use crate::value::{DynValueTypeMarker, Value, ValueRef, ValueRefMut, ValueTypeMarker};

pub enum SessionInputValue<'v> {
	ViewMut(ValueRefMut<'v, DynValueTypeMarker>),
	View(ValueRef<'v, DynValueTypeMarker>),
	Owned(Value<DynValueTypeMarker>)
}

impl Deref for SessionInputValue<'_> {
	type Target = Value;

	fn deref(&self) -> &Self::Target {
		match self {
			SessionInputValue::ViewMut(v) => v,
			SessionInputValue::View(v) => v,
			SessionInputValue::Owned(v) => v
		}
	}
}

impl<'v, T: ValueTypeMarker + ?Sized> From<ValueRefMut<'v, T>> for SessionInputValue<'v> {
	fn from(value: ValueRefMut<'v, T>) -> Self {
		SessionInputValue::ViewMut(value.into_dyn())
	}
}
impl<'v, T: ValueTypeMarker + ?Sized> From<ValueRef<'v, T>> for SessionInputValue<'v> {
	fn from(value: ValueRef<'v, T>) -> Self {
		SessionInputValue::View(value.into_dyn())
	}
}
impl<T: ValueTypeMarker + ?Sized> From<Value<T>> for SessionInputValue<'_> {
	fn from(value: Value<T>) -> Self {
		SessionInputValue::Owned(value.into_dyn())
	}
}
impl<'v, T: ValueTypeMarker + ?Sized> From<&'v Value<T>> for SessionInputValue<'v> {
	fn from(value: &'v Value<T>) -> Self {
		SessionInputValue::View(value.view().into_dyn())
	}
}

/// The inputs to a [`Session::run`] call.
///
/// [`Session::run`]: crate::session::Session::run
pub enum SessionInputs<'i, 'v, const N: usize = 0> {
	ValueMap(Vec<(Cow<'i, str>, SessionInputValue<'v>)>),
	ValueSlice(&'i [SessionInputValue<'v>]),
	ValueArray([SessionInputValue<'v>; N])
}

#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
impl<'i, 'v, K: Into<Cow<'i, str>>, V: Into<SessionInputValue<'v>>> From<std::collections::HashMap<K, V>> for SessionInputs<'i, 'v> {
	fn from(val: std::collections::HashMap<K, V>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
	}
}

impl<'i, 'v, K: Into<Cow<'i, str>>, V: Into<SessionInputValue<'v>>> From<Vec<(K, V)>> for SessionInputs<'i, 'v> {
	fn from(val: Vec<(K, V)>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
	}
}

impl<'i, 'v> From<&'i [SessionInputValue<'v>]> for SessionInputs<'i, 'v> {
	fn from(val: &'i [SessionInputValue<'v>]) -> Self {
		SessionInputs::ValueSlice(val)
	}
}

impl<'v, const N: usize> From<[SessionInputValue<'v>; N]> for SessionInputs<'_, 'v, N> {
	fn from(val: [SessionInputValue<'v>; N]) -> Self {
		SessionInputs::ValueArray(val)
	}
}

/// Construct the inputs to a session from an array or named map of values.
///
/// # Example
///
/// ## Array of values
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{value::Tensor, session::{builder::GraphOptimizationLevel, Session}};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.commit_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs![Tensor::from_array(([5], vec![1, 2, 3, 4, 5]))?])?;
/// # 	Ok(())
/// # }
/// ```
///
/// ## Map of named tensors
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{value::Tensor, session::{builder::GraphOptimizationLevel, Session}};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.commit_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs! {
/// 	"tokens" => Tensor::from_array(([5], vec![1, 2, 3, 4, 5]))?
/// })?;
/// # 	Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! inputs {
	($($v:expr),+ $(,)?) => (
		[$($crate::__private::core::convert::Into::<$crate::session::SessionInputValue<'_>>::into($v)),+]
	);
	($($n:expr => $v:expr),+ $(,)?) => (
		vec![$(($crate::__private::alloc::borrow::Cow::<str>::from($n), $crate::session::SessionInputValue::<'_>::from($v)),)+]
	);
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

	use super::SessionInputs;
	use crate::value::{DynTensor, Tensor};

	#[test]
	#[cfg(feature = "std")]
	fn test_hashmap_static_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let shape = vec![v.len() as i64];

		let mut inputs: HashMap<&str, DynTensor> = HashMap::new();
		inputs.insert("test", Tensor::from_array((shape, v))?.upcast());
		let _ = SessionInputs::from(inputs);

		Ok(())
	}

	#[test]
	#[cfg(feature = "std")]
	fn test_hashmap_string_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let shape = vec![v.len() as i64];

		let mut inputs: HashMap<String, DynTensor> = HashMap::new();
		inputs.insert("test".to_string(), Tensor::from_array((shape, v))?.upcast());
		let _ = SessionInputs::from(inputs);

		Ok(())
	}
}
