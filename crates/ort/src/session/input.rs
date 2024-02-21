use std::{borrow::Cow, collections::HashMap};

use crate::Value;

/// The inputs to a [`crate::Session::run`] call.
pub enum SessionInputs<'i, const N: usize = 0> {
	ValueMap(Vec<(Cow<'i, str>, Value)>),
	ValueSlice(&'i [Value]),
	ValueArray([Value; N])
}

impl<'i, K: Into<Cow<'i, str>>> From<HashMap<K, Value>> for SessionInputs<'i> {
	fn from(val: HashMap<K, Value>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|(k, v)| (k.into(), v)).collect())
	}
}

impl<'i> From<Vec<(Cow<'i, str>, Value)>> for SessionInputs<'i> {
	fn from(val: Vec<(Cow<'i, str>, Value)>) -> Self {
		SessionInputs::ValueMap(val)
	}
}

impl<'i> From<&'i [Value]> for SessionInputs<'i> {
	fn from(val: &'i [Value]) -> Self {
		SessionInputs::ValueSlice(val)
	}
}

impl<'i, const N: usize> From<[Value; N]> for SessionInputs<'i, N> {
	fn from(val: [Value; N]) -> Self {
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
/// # 	let mut session = Session::builder()?.with_model_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs![Array1::from_vec(vec![1, 2, 3, 4, 5])]?);
/// # 	Ok(())
/// # }
/// ```
///
/// Note that string tensors must be created manually with [`Value::from_string_array`].
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{GraphOptimizationLevel, Session, Value};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # 	let mut session = Session::builder()?.with_model_from_file("model.onnx")?;
/// let _ = session
/// 	.run(ort::inputs![Value::from_string_array(session.allocator(), Array1::from_vec(vec!["hello", "world"]))?]?);
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
/// # 	let mut session = Session::builder()?.with_model_from_file("model.onnx")?;
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
			Ok([$(::std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from)?,)+])
		})()
	);
	($($n:expr => $v:expr),+ $(,)?) => (
		(|| -> $crate::Result<_> {
			Ok(vec![$(::std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from).map(|v| (($n).into(), v))?,)+])
		})()
	);
}

#[cfg(test)]
mod tests {
	use std::{collections::HashMap, sync::Arc};

	use crate::*;

	#[test]
	fn test_hashmap_static_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];

		let mut inputs = HashMap::new();
		inputs.insert("test", (shape, arc).try_into()?);
		let _ = SessionInputs::from(inputs);

		Ok(())
	}

	#[test]
	fn test_hashmap_string_keys() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];

		let mut inputs = HashMap::new();
		inputs.insert("test".to_string(), (shape, arc).try_into()?);
		let _ = SessionInputs::from(inputs);

		Ok(())
	}
}
