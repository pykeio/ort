use std::collections::HashMap;

use smallvec::SmallVec;

use crate::{IoBinding, Value};

pub enum SessionInputs<'s, 'v> {
	IoBinding(IoBinding<'s>),
	ValueMap(HashMap<&'static str, Value<'v>>),
	ValueVec(SmallVec<[Value<'v>; 4]>)
}

impl<'s, 'v> From<IoBinding<'s>> for SessionInputs<'s, 'v> {
	fn from(val: IoBinding<'s>) -> Self {
		SessionInputs::IoBinding(val)
	}
}

impl<'s, 'v> From<HashMap<&'static str, Value<'v>>> for SessionInputs<'s, 'v> {
	fn from(val: HashMap<&'static str, Value<'v>>) -> Self {
		SessionInputs::ValueMap(val)
	}
}

impl<'s, 'v> From<Vec<Value<'v>>> for SessionInputs<'s, 'v> {
	fn from(val: Vec<Value<'v>>) -> Self {
		SessionInputs::ValueVec(val.into())
	}
}

impl<'s, 'v> From<SmallVec<[Value<'v>; 4]>> for SessionInputs<'s, 'v> {
	fn from(val: SmallVec<[Value<'v>; 4]>) -> Self {
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
