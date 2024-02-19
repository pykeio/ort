use std::collections::HashMap;

use crate::Value;

pub enum SessionInputKey<'i> {
	Borrowed(&'i str),
	Owned(String)
}

impl<'i> From<&'i str> for SessionInputKey<'i> {
	fn from(value: &'i str) -> Self {
		Self::Borrowed(value)
	}
}
impl<'i> From<String> for SessionInputKey<'i> {
	fn from(value: String) -> Self {
		Self::Owned(value)
	}
}

impl<'i> SessionInputKey<'i> {
	pub fn as_str(&self) -> &str {
		match self {
			Self::Borrowed(s) => s,
			Self::Owned(s) => s.as_str()
		}
	}
}

pub enum SessionInputs<'i, const N: usize = 0> {
	ValueMap(Vec<(SessionInputKey<'i>, Value)>),
	ValueSlice(&'i [Value]),
	ValueArray([Value; N])
}

impl<'i, K: Into<SessionInputKey<'i>>> From<HashMap<K, Value>> for SessionInputs<'i> {
	fn from(val: HashMap<K, Value>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|(k, v)| (k.into(), v)).collect())
	}
}

impl<'i> From<Vec<(SessionInputKey<'i>, Value)>> for SessionInputs<'i> {
	fn from(val: Vec<(SessionInputKey<'i>, Value)>) -> Self {
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

/// Construct the inputs to a session from an array or map of values.
///
/// The result of this macro is an `Result<SessionInputs, OrtError>`, so make sure you `?` on the result.
///
/// For tensors, note that using certain array structures can have performance implications.
/// - `&CowArray`, `ArrayView` will **always** be copied.
/// - `Array`, `&mut ArcArray` will only be copied **if the tensor is not contiguous** (i.e. has been reshaped).
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
/// let mut session = Session::builder()?.with_model_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs![Array1::from_vec(vec![1, 2, 3, 4, 5])]?);
/// # Ok(())
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
/// let mut session = Session::builder()?.with_model_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs! {
/// 	"tokens" => Array1::from_vec(vec![1, 2, 3, 4, 5])
/// }?);
/// # Ok(())
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
			Ok(vec![$(::std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from).map(|v| ($crate::SessionInputKey::from($n), v))?,)+])
		})()
	);
}

#[cfg(test)]
mod tests {
	use std::collections::HashMap;

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
