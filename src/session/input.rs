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
	($($v:expr),+ $(,)?) => ($crate::smallvec::smallvec![$(std::convert::TryInto::<Value<'_>>::try_into($v)?,)+]);
	($($n:expr => $v:expr),+ $(,)?) => {{
		let mut inputs = std::collections::HashMap::<_, Value<'_>>::new();
		$(
			inputs.insert($n, std::convert::TryInto::<Value<'_>>::try_into($v)?);
		)+
		inputs
	}};
}
