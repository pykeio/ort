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
/// # let env = ort_test_util::test_env().clone();
/// let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
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
	keys: SmallVec<[&'r str; STACK_SESSION_OUTPUTS]>,
	values: SmallVec<[Option<DynValue>; STACK_SESSION_OUTPUTS]>,
	effective_len: usize
}

unsafe impl Send for SessionOutputs<'_> {}

impl<'r> SessionOutputs<'r> {
	pub(crate) fn new(output_names: SmallVec<[&'r str; STACK_SESSION_OUTPUTS]>, output_values: SmallVec<[DynValue; STACK_SESSION_OUTPUTS]>) -> Self {
		debug_assert_eq!(output_names.len(), output_values.len());
		Self {
			effective_len: output_names.len(),
			keys: output_names,
			values: output_values.into_iter().map(Some).collect()
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
				return self.values[i].as_ref();
			}
		}
		None
	}

	pub fn get_mut(&mut self, key: impl AsRef<str>) -> Option<&mut DynValue> {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for (i, k) in self.keys.iter().enumerate() {
			if &key == k {
				return self.values[i].as_mut();
			}
		}
		None
	}

	pub fn remove(&mut self, key: impl AsRef<str>) -> Option<DynValue> {
		let key = key.as_ref();
		assert!(!key.is_empty(), "output name cannot be empty");
		for (i, k) in self.keys.iter_mut().enumerate() {
			if &key == k {
				let value = self.values[i].take()?;
				*k = "";
				self.effective_len -= 1;
				return Some(value);
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
		match self.values.get(index) {
			Some(Some(value)) => value,
			Some(None) => panic!("output #{index} was removed"),
			None => panic!("attempted to index output #{index} when there are only {} outputs", self.values.len())
		}
	}
}

impl IndexMut<usize> for SessionOutputs<'_> {
	fn index_mut(&mut self, index: usize) -> &mut Self::Output {
		let num_outputs = self.values.len();
		match self.values.get_mut(index) {
			Some(Some(value)) => value,
			Some(None) => panic!("output #{index} was removed"),
			None => panic!("attempted to index output #{index} when there are only {num_outputs} outputs")
		}
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
	value_iter: core::slice::Iter<'x, Option<DynValue>>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x> Iterator for Values<'x, '_> {
	type Item = ValueRef<'x>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			self.key_iter.next()?;
			if let Some(value) = self.value_iter.next()? {
				self.effective_len -= 1;
				return Some(value.view());
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
	value_iter: core::slice::IterMut<'x, Option<DynValue>>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x> Iterator for ValuesMut<'x, '_> {
	type Item = ValueRefMut<'x>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			self.key_iter.next()?;
			if let Some(value) = self.value_iter.next()? {
				self.effective_len -= 1;
				return Some(value.view_mut());
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
	value_iter: core::slice::Iter<'x, Option<DynValue>>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x, 'k> Iterator for Iter<'x, 'k> {
	type Item = (&'k str, ValueRef<'x>);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let key = self.key_iter.next()?;
			if let Some(value) = self.value_iter.next()? {
				self.effective_len -= 1;
				return Some((*key, value.view()));
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
	value_iter: core::slice::IterMut<'x, Option<DynValue>>,
	key_iter: core::slice::Iter<'x, &'k str>,
	effective_len: usize
}

impl<'x, 'k> Iterator for IterMut<'x, 'k> {
	type Item = (&'k str, ValueRefMut<'x>);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let key = self.key_iter.next()?;
			if let Some(value) = self.value_iter.next()? {
				self.effective_len -= 1;
				return Some((*key, value.view_mut()));
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
	keys: smallvec::IntoIter<[&'r str; STACK_SESSION_OUTPUTS]>,
	values: smallvec::IntoIter<[Option<DynValue>; STACK_SESSION_OUTPUTS]>,
	effective_len: usize
}

impl<'r> Iterator for IntoIter<'r> {
	type Item = (&'r str, DynValue);

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let key = self.keys.next()?;
			if let Some(value) = self.values.next()? {
				self.effective_len -= 1;
				return Some((key, value));
			}
		}
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		(self.effective_len, Some(self.effective_len))
	}
}

impl ExactSizeIterator for IntoIter<'_> {}
impl FusedIterator for IntoIter<'_> {}

#[cfg(test)]
mod tests {
	use alloc::sync::Arc;
	use core::panic::AssertUnwindSafe;
	use std::panic::catch_unwind;

	use smallvec::smallvec;

	use super::SessionOutputs;
	use crate::{AsPointer, value::Tensor};

	#[test]
	fn test_iter_after_remove() -> crate::Result<()> {
		let tensor = || Tensor::from_array(([1_usize], vec![0.0_f32])).map(|t| t.into_dyn());
		let mut outputs = SessionOutputs::new(smallvec!["a", "b", "c"], smallvec![tensor()?, tensor()?, tensor()?]);
		outputs.remove("b");

		// The last item of every iterator must be output `c`, not the removed `b`.
		let c = outputs["c"].ptr();
		assert_eq!(outputs.iter().last().map(|(k, v)| (k, v.ptr())), Some(("c", c)));
		assert_eq!(outputs.values().last().map(|v| v.ptr()), Some(c));
		assert_eq!(outputs.iter_mut().last().map(|(k, v)| (k, v.ptr())), Some(("c", c)));
		assert_eq!(outputs.values_mut().last().map(|v| v.ptr()), Some(c));
		assert_eq!(outputs.into_iter().last().map(|(k, v)| (k, v.ptr())), Some(("c", c)));
		Ok(())
	}

	#[test]
	fn test_remove_actually_removes() -> crate::Result<()> {
		let tensor = || Tensor::from_array(([1_usize], vec![0.0_f32])).map(|t| t.into_dyn());
		let mut outputs = SessionOutputs::new(smallvec!["a", "b"], smallvec![tensor()?, tensor()?]);
		let mut b = outputs.remove("b").expect("");
		assert_eq!(outputs.len(), 1);

		assert_eq!(Arc::strong_count(&b.inner), 1);
		b.try_extract_tensor_mut::<f32>()?.1[0] = 1.0;

		assert!(outputs.remove("b").is_none());
		assert!(outputs.get("b").is_none());
		assert!(outputs.get_mut("b").is_none());
		assert!(!outputs.contains_key("b"));
		assert!(catch_unwind(AssertUnwindSafe(|| outputs[1].ptr())).is_err());
		assert!(catch_unwind(AssertUnwindSafe(|| outputs["b"].ptr())).is_err());

		let _ = &mut outputs[0];
		assert_eq!(outputs.iter().count(), 1);
		Ok(())
	}
}
