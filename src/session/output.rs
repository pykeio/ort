use std::{
	collections::HashMap,
	ffi::c_void,
	ops::{Deref, DerefMut, Index}
};

use crate::{Allocator, Value};

/// The outputs returned by a [`crate::Session`] inference call.
///
/// This type allows session outputs to be retrieved by index or by name.
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
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
pub struct SessionOutputs<'s> {
	map: HashMap<&'s str, Value>,
	idxs: Vec<&'s str>,
	backing_ptr: Option<(&'s Allocator, *mut c_void)>
}

unsafe impl<'s> Send for SessionOutputs<'s> {}

impl<'s> SessionOutputs<'s> {
	pub(crate) fn new(output_names: impl Iterator<Item = &'s str> + Clone, output_values: impl IntoIterator<Item = Value>) -> Self {
		let map = output_names.clone().zip(output_values).collect();
		Self {
			map,
			idxs: output_names.collect(),
			backing_ptr: None
		}
	}

	pub(crate) fn new_backed(
		output_names: impl Iterator<Item = &'s str> + Clone,
		output_values: impl IntoIterator<Item = Value>,
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
			map: HashMap::new(),
			idxs: Vec::new(),
			backing_ptr: None
		}
	}
}

impl<'s> Drop for SessionOutputs<'s> {
	fn drop(&mut self) {
		if let Some((allocator, ptr)) = self.backing_ptr {
			unsafe { allocator.free(ptr) };
		}
	}
}

impl<'s> Deref for SessionOutputs<'s> {
	type Target = HashMap<&'s str, Value>;

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
	type Output = Value;
	fn index(&self, index: &str) -> &Self::Output {
		self.map.get(index).expect("no entry found for key")
	}
}

impl<'s> Index<String> for SessionOutputs<'s> {
	type Output = Value;
	fn index(&self, index: String) -> &Self::Output {
		self.map.get(index.as_str()).expect("no entry found for key")
	}
}

impl<'s> Index<usize> for SessionOutputs<'s> {
	type Output = Value;
	fn index(&self, index: usize) -> &Self::Output {
		self.map.get(&self.idxs[index]).expect("no entry found for key")
	}
}
