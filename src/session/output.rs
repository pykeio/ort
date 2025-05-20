use alloc::string::String;
use core::{
	iter::FusedIterator,
	mem::ManuallyDrop,
	ops::{Index, IndexMut},
	ptr
};

use smallvec::SmallVec;

use crate::{
	util::STACK_SESSION_OUTPUTS,
	value::{DynValue, ValueRef, ValueRefMut}
};

/// The outputs returned by a [`Session`] inference call.
///
/// This type allows session outputs to be retrieved by index or by name.
///
/// ```
/// # use ort::{value::TensorRef, session::{builder::GraphOptimizationLevel, Session}};
/// # fn main() -> ort::Result<()> {
/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
///
/// // get the first output
/// let output = &outputs[0];
/// // get an output by name
/// let output = &outputs["Identity:0"];
/// # 	Ok(())
/// # }
/// ```
///
/// [`Session`]: crate::session::Session
#[derive(Debug)]
pub struct SessionOutputs<'r> {
	keys: SmallVec<&'r str, { STACK_SESSION_OUTPUTS }>,
	values: SmallVec<DynValue, { STACK_SESSION_OUTPUTS }>,
	effective_len: usize
}

unsafe impl Send for SessionOutputs<'_> {}

impl<'r> SessionOutputs<'r> {
	pub(crate) fn new(output_names: SmallVec<&'r str, { STACK_SESSION_OUTPUTS }>, output_values: SmallVec<DynValue, { STACK_SESSION_OUTPUTS }>) -> Self {
		debug_assert_eq!(output_names.len(), output_values.len());
		Self {
			effective_len: output_names.len(),
			keys: output_names,
			values: output_values
		}
	}

	pub(crate) fn new_empty() -> Self {
		Self {
			effective_len: 0,
			keys: SmallVec::new(),
			values: SmallVec::new()
		}
	}

	pub fn contains_key(&self, key: impl AsRef<str>) -> bool {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for k in &self.keys {
			if &key == k {
				return true;
			}
		}
		false
	}

	pub fn get(&self, key: impl AsRef<str>) -> Option<&DynValue> {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for (i, k) in self.keys.iter().enumerate() {
			if &key == k {
				return Some(&self.values[i]);
			}
		}
		None
	}

	pub fn get_mut(&mut self, key: impl AsRef<str>) -> Option<&mut DynValue> {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for (i, k) in self.keys.iter().enumerate() {
			if &key == k {
				return Some(&mut self.values[i]);
			}
		}
		None
	}

	pub fn remove(&mut self, key: impl AsRef<str>) -> Option<DynValue> {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for (i, k) in self.keys.iter_mut().enumerate() {
			if &key == k {
				*k = "";
				self.effective_len -= 1;
				return Some(DynValue::clone_of(&self.values[i]));
			}
		}
		None
	}

	#[inline(always)]
	#[allow(clippy::len_without_is_empty)]
	pub fn len(&self) -> usize {
		self.effective_len
	}

	pub fn keys(&self) -> Keys<'_, 'r> {
		Keys {
			iter: self.keys.iter(),
			effective_len: self.effective_len
		}
	}

	pub fn values(&self) -> Values<'_, 'r> {
		Values {
			key_iter: self.keys.iter(),
			value_iter: self.values.iter(),
			effective_len: self.effective_len
		}
	}

	pub fn values_mut(&mut self) -> ValuesMut<'_, 'r> {
		ValuesMut {
			key_iter: self.keys.iter(),
			value_iter: self.values.iter_mut(),
			effective_len: self.effective_len
		}
	}

	pub fn iter(&self) -> Iter<'_, 'r> {
		Iter {
			key_iter: self.keys.iter(),
			value_iter: self.values.iter(),
			effective_len: self.effective_len
		}
	}

	pub fn iter_mut(&mut self) -> IterMut<'_, 'r> {
		IterMut {
			key_iter: self.keys.iter(),
			value_iter: self.values.iter_mut(),
			effective_len: self.effective_len
		}
	}
}

impl<'x, 'r> IntoIterator for &'x SessionOutputs<'r> {
	type IntoIter = Iter<'x, 'r>;
	type Item = (&'r str, ValueRef<'x>);

	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'x, 'r> IntoIterator for &'x mut SessionOutputs<'r> {
	type IntoIter = IterMut<'x, 'r>;
	type Item = (&'r str, ValueRefMut<'x>);

	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

impl<'r> IntoIterator for SessionOutputs<'r> {
	type IntoIter = IntoIter<'r>;
	type Item = (&'r str, DynValue);

	fn into_iter(self) -> Self::IntoIter {
		let this = ManuallyDrop::new(self);
		let keys = unsafe { ptr::read(&this.keys) }.into_iter();
		let values = unsafe { ptr::read(&this.values) }.into_iter();
		IntoIter {
			keys,
			values,
			effective_len: this.effective_len
		}
	}
}

impl Index<&str> for SessionOutputs<'_> {
	type Output = DynValue;
	fn index(&self, key: &str) -> &Self::Output {
		self.get(key).unwrap_or_else(|| panic!("no output named `{key}`"))
	}
}

impl IndexMut<&str> for SessionOutputs<'_> {
	fn index_mut(&mut self, key: &str) -> &mut Self::Output {
		self.get_mut(key).unwrap_or_else(|| panic!("no output named `{key}`"))
	}
}

impl Index<String> for SessionOutputs<'_> {
	type Output = DynValue;
	fn index(&self, key: String) -> &Self::Output {
		self.get(&key).unwrap_or_else(|| panic!("no output named `{key}`"))
	}
}

impl IndexMut<String> for SessionOutputs<'_> {
	fn index_mut(&mut self, key: String) -> &mut Self::Output {
		self.get_mut(&key).unwrap_or_else(|| panic!("no output named `{key}`"))
	}
}

impl Index<usize> for SessionOutputs<'_> {
	type Output = DynValue;
	fn index(&self, index: usize) -> &Self::Output {
		if index > self.values.len() {
			panic!("attempted to index output #{index} when there are only {} outputs", self.values.len());
		}
		&self.values[index]
	}
}

impl IndexMut<usize> for SessionOutputs<'_> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		if index > self.values.len() {
			panic!("attempted to index output #{index} when there are only {} outputs", self.values.len());
		}
		&mut self.values[index]
	}
}

pub struct Keys<'x, 'r> {
	iter: core::slice::Iter<'x, &'r str>,
	effective_len: usize
}

impl<'r> Iterator for Keys<'_, 'r> {
	type Item = &'r str;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.iter.next() {
				None => return None,
				Some(&"") => continue,
				Some(x) => {
					self.effective_len -= 1;
					return Some(x);
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for Keys<'_, '_> {}
impl FusedIterator for Keys<'_, '_> {}

pub struct Values<'x, 'k> {
	value_iter: core::slice::Iter<'x, DynValue>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x> Iterator for Values<'x, '_> {
	type Item = ValueRef<'x>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.key_iter.next() {
				None => return None,
				Some(&"") => continue,
				Some(_) => {
					self.effective_len -= 1;
					return self.value_iter.next().map(DynValue::view);
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for Values<'_, '_> {}
impl FusedIterator for Values<'_, '_> {}

pub struct ValuesMut<'x, 'k> {
	value_iter: core::slice::IterMut<'x, DynValue>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x> Iterator for ValuesMut<'x, '_> {
	type Item = ValueRefMut<'x>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.key_iter.next() {
				None => return None,
				Some(&"") => continue,
				Some(_) => {
					self.effective_len -= 1;
					return self.value_iter.next().map(DynValue::view_mut);
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for ValuesMut<'_, '_> {}
impl FusedIterator for ValuesMut<'_, '_> {}

pub struct Iter<'x, 'k> {
	value_iter: core::slice::Iter<'x, DynValue>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x, 'k> Iterator for Iter<'x, 'k> {
	type Item = (&'k str, ValueRef<'x>);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.key_iter.next() {
				None => return None,
				Some(&"") => continue,
				Some(key) => {
					self.effective_len -= 1;
					return self.value_iter.next().map(|v| (*key, v.view()));
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for Iter<'_, '_> {}
impl FusedIterator for Iter<'_, '_> {}

pub struct IterMut<'x, 'k> {
	value_iter: core::slice::IterMut<'x, DynValue>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x, 'k> Iterator for IterMut<'x, 'k> {
	type Item = (&'k str, ValueRefMut<'x>);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.key_iter.next() {
				None => return None,
				Some(&"") => continue,
				Some(key) => {
					self.effective_len -= 1;
					return self.value_iter.next().map(|v| (*key, v.view_mut()));
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for IterMut<'_, '_> {}
impl FusedIterator for IterMut<'_, '_> {}

pub struct IntoIter<'r> {
	keys: smallvec::IntoIter<&'r str, { STACK_SESSION_OUTPUTS }>,
	values: smallvec::IntoIter<DynValue, { STACK_SESSION_OUTPUTS }>,
	effective_len: usize
}

impl<'r> Iterator for IntoIter<'r> {
	type Item = (&'r str, DynValue);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			match self.keys.next() {
				None => return None,
				Some("") => continue,
				Some(key) => {
					self.effective_len -= 1;
					return self.values.next().map(|v| (key, v));
				}
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}
