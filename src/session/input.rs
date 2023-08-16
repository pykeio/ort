use std::collections::HashMap;

use crate::{IoBinding, Value};

pub enum SessionInputs<'s, 'v> {
	IoBinding(IoBinding<'s>),
	ValueMap(HashMap<String, Value<'v>>),
	ValueVec(Vec<Value<'v>>)
}

impl<'s, 'v> From<IoBinding<'s>> for SessionInputs<'s, 'v> {
	fn from(val: IoBinding<'s>) -> Self {
		SessionInputs::IoBinding(val)
	}
}

impl<'s, 'v> From<HashMap<String, Value<'v>>> for SessionInputs<'s, 'v> {
	fn from(val: HashMap<String, Value<'v>>) -> Self {
		SessionInputs::ValueMap(val)
	}
}

impl<'s, 'v> From<HashMap<&'static str, Value<'v>>> for SessionInputs<'s, 'v> {
	fn from(val: HashMap<&'static str, Value<'v>>) -> Self {
		SessionInputs::ValueMap(val.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
	}
}

impl<'s, 'v> From<Vec<Value<'v>>> for SessionInputs<'s, 'v> {
	fn from(val: Vec<Value<'v>>) -> Self {
		SessionInputs::ValueVec(val)
	}
}

#[macro_export]
macro_rules! inputs {
	(bind = $($v:expr),+) => ($v);
	($($v:expr),+ $(,)?) => (vec![$(std::convert::TryInto::<Value<'_>>::try_into($v)?,)+]);
	($($n:expr => $v:expr),+ $(,)?) => {{
		let mut inputs = std::collections::HashMap::<_, Value<'_>>::new();
		$(
			inputs.insert($n, std::convert::TryInto::<Value<'_>>::try_into($v)?);
		)+
		inputs
	}};
}
