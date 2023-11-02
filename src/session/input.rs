use std::collections::HashMap;

use smallvec::SmallVec;

use crate::{IoBinding, Value};

pub enum SessionInputs<'s> {
	IoBinding(IoBinding<'s>),
	ValueMap(HashMap<&'static str, Value>),
	ValueVec(SmallVec<[Value; 4]>)
}

impl<'s> From<IoBinding<'s>> for SessionInputs<'s> {
	fn from(val: IoBinding<'s>) -> Self {
		SessionInputs::IoBinding(val)
	}
}

impl<'s> From<HashMap<&'static str, Value>> for SessionInputs<'s> {
	fn from(val: HashMap<&'static str, Value>) -> Self {
		SessionInputs::ValueMap(val)
	}
}

impl<'s> From<Vec<Value>> for SessionInputs<'s> {
	fn from(val: Vec<Value>) -> Self {
		SessionInputs::ValueVec(val.into())
	}
}

impl<'s> From<SmallVec<[Value; 4]>> for SessionInputs<'s> {
	fn from(val: SmallVec<[Value; 4]>) -> Self {
		SessionInputs::ValueVec(val)
	}
}

/// Construct the inputs to a session from an array, a map, or an IO binding.
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
/// # use ort::{Environment, LoggingLevel, GraphOptimizationLevel, SessionBuilder};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # let environment = Environment::default().into_arc();
/// let mut session = SessionBuilder::new(&environment)?.with_model_from_file("model.onnx")?;
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
/// # use ort::{Environment, LoggingLevel, GraphOptimizationLevel, SessionBuilder};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # let environment = Environment::default().into_arc();
/// let mut session = SessionBuilder::new(&environment)?.with_model_from_file("model.onnx")?;
/// let _ = session.run(ort::inputs! {
/// 	"tokens" => Array1::from_vec(vec![1, 2, 3, 4, 5])
/// }?);
/// # Ok(())
/// # }
/// ```
///
/// ## I/O Binding
///
/// ```no_run
/// # use std::{error::Error, sync::Arc};
/// # use ndarray::Array1;
/// # use ort::{Environment, LoggingLevel, GraphOptimizationLevel, SessionBuilder};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// # let environment = Environment::default().into_arc();
/// let mut session = SessionBuilder::new(&environment)?.with_model_from_file("model.onnx")?;
/// let mut binding = session.create_binding()?;
/// let _ = session.run(ort::inputs!(bind = binding)?);
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! inputs {
	(bind = $v:expr) => ($crate::Result::<_, $crate::Error>::Ok($v));
	($($v:expr),+ $(,)?) => (
		[$(std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from),)+]
			.into_iter()
			.collect::<$crate::Result<$crate::smallvec::SmallVec<_>>>()
	);
	($($n:expr => $v:expr),+ $(,)?) => {{
		[$(std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::Error::from).map(|v| ($n, v)),)+]
			.into_iter()
			.collect::<$crate::Result<std::collections::HashMap::<_, $crate::Value>>>()
	}};
}
