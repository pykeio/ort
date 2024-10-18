use std::{
	collections::BTreeMap,
	ffi::c_void,
	ops::{Deref, DerefMut, Index}
};

use crate::{memory::Allocator, value::DynValue};

/// The outputs returned by a [`crate::Session`] inference call.
///
/// This type allows session outputs to be retrieved by index or by name.
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![input]?)?;
///
/// // get the first output
/// let output = &outputs[0];
/// // get an output by name
/// let output = &outputs["Identity:0"];
/// # 	Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SessionOutputs<'r, 's> {
	map: BTreeMap<&'r str, DynValue>,
	idxs: Vec<&'r str>,
	backing_ptr: Option<(&'s Allocator, *mut c_void)>
}

unsafe impl Send for SessionOutputs<'_, '_> {}

impl<'r, 's> SessionOutputs<'r, 's> {
	pub(crate) fn new(output_names: impl Iterator<Item = &'r str> + Clone, output_values: impl IntoIterator<Item = DynValue>) -> Self {
		let map = output_names.clone().zip(output_values).collect();
		Self {
			map,
			idxs: output_names.collect(),
			backing_ptr: None
		}
	}

	pub(crate) fn new_backed(
		output_names: impl Iterator<Item = &'r str> + Clone,
		output_values: impl IntoIterator<Item = DynValue>,
		allocator: &'s Allocator,
		backing_ptr: *mut c_void
	) -> Self {
		let map = output_names.clone().zip(output_values).collect();
		Self {
			map,
			idxs: output_names.collect(),
			backing_ptr: Some((allocator, backing_ptr))
		}
	}

	pub(crate) fn new_empty() -> Self {
		Self {
			map: BTreeMap::new(),
			idxs: Vec::new(),
			backing_ptr: None
		}
	}
}

impl Drop for SessionOutputs<'_, '_> {
	fn drop(&mut self) {
		if let Some((allocator, ptr)) = self.backing_ptr {
			unsafe { allocator.free(ptr) };
		}
	}
}

impl<'r> Deref for SessionOutputs<'r, '_> {
	type Target = BTreeMap<&'r str, DynValue>;

	fn deref(&self) -> &Self::Target {
		&self.map
	}
}

impl DerefMut for SessionOutputs<'_, '_> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.map
	}
}

impl Index<&str> for SessionOutputs<'_, '_> {
	type Output = DynValue;
	fn index(&self, index: &str) -> &Self::Output {
		self.map.get(index).expect("no entry found for key")
	}
}

impl Index<String> for SessionOutputs<'_, '_> {
	type Output = DynValue;
	fn index(&self, index: String) -> &Self::Output {
		self.map.get(index.as_str()).expect("no entry found for key")
	}
}

impl Index<usize> for SessionOutputs<'_, '_> {
	type Output = DynValue;
	fn index(&self, index: usize) -> &Self::Output {
		self.map.get(&self.idxs[index]).expect("no entry found for key")
	}
}
