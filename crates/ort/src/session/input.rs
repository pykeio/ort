use std::collections::HashMap;

use compact_str::CompactString;

use crate::Value;

pub enum SessionInputs<'i, const N: usize = 0> {
	ValueMap(HashMap<CompactString, Value>),
	ValueSlice(&'i [Value]),
	ValueArray([Value; N])
}

impl<'i, K: Into<CompactString>> From<HashMap<K, Value>> for SessionInputs<'i> {
	fn from(val: HashMap<K, Value>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|c| (c.0.into(), c.1)).collect())
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
	($($n:expr => $v:expr),+ $(,)?) => {{
		[$(::std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from).map(|v| ($n, v)),)+]
			.into_iter()
			.collect::<$crate::Result<::std::collections::HashMap::<_, $crate::Value>>>()
	}};
}
