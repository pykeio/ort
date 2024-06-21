use std::{borrow::Cow, collections::HashMap, ops::Deref};

use crate::{
	value::{DynValueTypeMarker, ValueTypeMarker},
	Value, ValueRef, ValueRefMut
};

pub enum SessionInputValue<'v> {
	ViewMut(ValueRefMut<'v, DynValueTypeMarker>),
	View(ValueRef<'v, DynValueTypeMarker>),
	Owned(Value<DynValueTypeMarker>)
}

impl<'v> Deref for SessionInputValue<'v> {
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
impl<'v, T: ValueTypeMarker + ?Sized> From<Value<T>> for SessionInputValue<'v> {
	fn from(value: Value<T>) -> Self {
		SessionInputValue::Owned(value.into_dyn())
	}
}

/// The inputs to a [`crate::Session::run`] call.
pub enum SessionInputs<'i, 'v, const N: usize = 0> {
	ValueMap(Vec<(Cow<'i, str>, SessionInputValue<'v>)>),
	ValueSlice(&'i [SessionInputValue<'v>]),
	ValueArray([SessionInputValue<'v>; N])
}

impl<'i, 'v, K: Into<Cow<'i, str>>, V: Into<SessionInputValue<'v>>> From<HashMap<K, V>> for SessionInputs<'i, 'v> {
	fn from(val: HashMap<K, V>) -> Self {
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

impl<'i, 'v, const N: usize> From<[SessionInputValue<'v>; N]> for SessionInputs<'i, 'v, N> {
	fn from(val: [SessionInputValue<'v>; N]) -> Self {
		SessionInputs::ValueArray(val)
	}
}

/// Construct the inputs to a session from an array or named map of values.
///
/// See [`Value::from_array`] for details on what types a tensor can be created from.
///
/// Note that the output of this macro is a `Result<SessionInputs, OrtError>`, so make sure to handle any potential
/// errors.
///
/// # Example
///
/// ## Array of tensors
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.commit_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs![Array1::from_vec(vec![1, 2, 3, 4, 5])]?);
/// # 	Ok(())
/// # }
/// ```
///
/// Note that string tensors must be created manually with [`crate::Tensor::from_string_array`].
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{GraphOptimizationLevel, Session, Tensor};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.commit_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs![Tensor::from_string_array(Array1::from_vec(vec!["hello", "world"]))?]?);
/// # 	Ok(())
/// # }
/// ```
///
/// ## Map of named tensors
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.commit_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs! {
/// 	"tokens" => Array1::from_vec(vec![1, 2, 3, 4, 5])
/// }?);
/// # 	Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! inputs {
	($($v:expr),+ $(,)?) => (
		(|| -> $crate::Result<_> {
			Ok([$(::std::convert::Into::<$crate::SessionInputValue<'_>>::into(::std::convert::TryInto::<$crate::DynValue>::try_into($v).map_err($crate::Error::from)?)),+])
		})()
	);
	($($n:expr => $v:expr),+ $(,)?) => (
		(|| -> $crate::Result<_> {
			Ok(vec![$(
				::std::convert::TryInto::<$crate::DynValue>::try_into($v)
					.map_err($crate::Error::from)
					.map(|v| (::std::borrow::Cow::<str>::from($n), $crate::SessionInputValue::from(v)))?,)+])
		})()
	);
}

#[cfg(test)]
mod tests {
	use std::{collections::HashMap, sync::Arc};

	use crate::{DynTensor, SessionInputs};

	#[test]
	fn test_hashmap_static_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];

		let mut inputs: HashMap<&str, DynTensor> = HashMap::new();
		inputs.insert("test", (shape, arc).try_into()?);
		let _ = SessionInputs::from(inputs);

		Ok(())
	}

	#[test]
	fn test_hashmap_string_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];

		let mut inputs: HashMap<String, DynTensor> = HashMap::new();
		inputs.insert("test".to_string(), (shape, arc).try_into()?);
		let _ = SessionInputs::from(inputs);

		Ok(())
	}
}
