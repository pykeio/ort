use std::{
	collections::HashMap,
	ops::{Deref, Index}
};

use crate::Value;

pub struct SessionOutputs {
	map: HashMap<String, Value<'static>>,
	idxs: Vec<String>
}

impl SessionOutputs {
	pub(crate) fn new(output_names: Vec<String>, output_values: impl IntoIterator<Item = Value<'static>>) -> Self {
		let map = output_names.iter().cloned().zip(output_values).collect();
		Self { map, idxs: output_names }
	}

	pub(crate) fn new_empty() -> Self {
		Self {
			map: HashMap::new(),
			idxs: Vec::new()
		}
	}
}

impl Deref for SessionOutputs {
	type Target = HashMap<String, Value<'static>>;

	fn deref(&self) -> &Self::Target {
		&self.map
	}
}

impl Index<&str> for SessionOutputs {
	type Output = Value<'static>;
	fn index(&self, index: &str) -> &Self::Output {
		self.map.get(index).unwrap()
	}
}

impl Index<String> for SessionOutputs {
	type Output = Value<'static>;
	fn index(&self, index: String) -> &Self::Output {
		self.map.get(index.as_str()).unwrap()
	}
}

impl Index<usize> for SessionOutputs {
	type Output = Value<'static>;
	fn index(&self, index: usize) -> &Self::Output {
		self.map.get(&self.idxs[index]).unwrap()
	}
}
