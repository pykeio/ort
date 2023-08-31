use std::{
	collections::HashMap,
	ops::{Deref, DerefMut, Index}
};

use crate::Value;

pub struct SessionOutputs<'s> {
	map: HashMap<String, Value<'s>>,
	idxs: Vec<String>
}

impl<'s> SessionOutputs<'s> {
	pub(crate) fn new(output_names: Vec<String>, output_values: impl IntoIterator<Item = Value<'s>>) -> Self {
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

impl<'s> Deref for SessionOutputs<'s> {
	type Target = HashMap<String, Value<'s>>;

	fn deref(&self) -> &Self::Target {
		&self.map
	}
}

impl<'s> DerefMut for SessionOutputs<'s> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.map
	}
}

impl<'s> Index<&str> for SessionOutputs<'s> {
	type Output = Value<'s>;
	fn index(&self, index: &str) -> &Self::Output {
		self.map.get(index).unwrap()
	}
}

impl<'s> Index<String> for SessionOutputs<'s> {
	type Output = Value<'s>;
	fn index(&self, index: String) -> &Self::Output {
		self.map.get(index.as_str()).unwrap()
	}
}

impl<'s> Index<usize> for SessionOutputs<'s> {
	type Output = Value<'s>;
	fn index(&self, index: usize) -> &Self::Output {
		self.map.get(&self.idxs[index]).unwrap()
	}
}
