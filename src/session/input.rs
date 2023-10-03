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

#[macro_export]
macro_rules! inputs {
	(bind = $($v:expr),+) => ($v);
	($($v:expr),+ $(,)?) => ([$(std::convert::TryInto::<$crate::Value>::try_into($v).map_err($crate::OrtError::from),)+].into_iter().collect::<$crate::OrtResult<$crate::smallvec::SmallVec<_>>>());
	($($n:expr => $v:expr),+ $(,)?) => {{
		let mut inputs = std::collections::HashMap::<_, $crate::Value>::new();
		$(
			inputs.insert($n, std::convert::TryInto::<$crate::Value>::try_into($v)?);
		)+
		inputs
	}};
}
