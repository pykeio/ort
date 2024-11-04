use std::{
	path::Path,
	ptr::{self, NonNull},
	sync::Arc
};

use crate::{Allocator, Result, ortsys, util};

#[derive(Debug)]
pub(crate) struct AdapterInner {
	pub(crate) ptr: NonNull<ort_sys::OrtLoraAdapter>
}

impl Drop for AdapterInner {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseLoraAdapter(self.ptr.as_ptr())];
	}
}

#[derive(Debug, Clone)]
pub struct Adapter {
	pub(crate) inner: Arc<AdapterInner>
}

impl Adapter {
	pub fn from_file(path: impl AsRef<Path>, allocator: Option<&Allocator>) -> Result<Self> {
		let path = util::path_to_os_char(path);
		let allocator_ptr = allocator.map(|c| c.ptr()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapter(path.as_ptr(), allocator_ptr, &mut ptr)?];
		Ok(Adapter {
			inner: Arc::new(AdapterInner {
				ptr: unsafe { NonNull::new_unchecked(ptr) }
			})
		})
	}

	pub fn from_memory(bytes: &[u8], allocator: Option<&Allocator>) -> Result<Self> {
		let allocator_ptr = allocator.map(|c| c.ptr()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapterFromArray(bytes.as_ptr().cast(), bytes.len(), allocator_ptr, &mut ptr)?];
		Ok(Adapter {
			inner: Arc::new(AdapterInner {
				ptr: unsafe { NonNull::new_unchecked(ptr) }
			})
		})
	}

	pub fn ptr(&self) -> *mut ort_sys::OrtLoraAdapter {
		self.inner.ptr.as_ptr()
	}
}

#[cfg(test)]
mod tests {
	use std::fs;

	use super::Adapter;
	use crate::{RunOptions, Session, Tensor};

	#[test]
	fn test_lora() -> crate::Result<()> {
		let model = Session::builder()?.commit_from_file("tests/data/lora_model.onnx")?;
		let lora = Adapter::from_file("tests/data/adapter.orl", None)?;

		let mut run_options = RunOptions::new()?;
		run_options.add_adapter(&lora)?;

		let output: Tensor<f32> = model
			.run_with_options(crate::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?]?, &run_options)?
			.remove("output")
			.expect("")
			.downcast()?;
		let (_, output) = output.extract_raw_tensor();
		assert_eq!(output[0], 154.0);
		assert_eq!(output[1], 176.0);
		assert_eq!(output[2], 198.0);
		assert_eq!(output[3], 220.0);

		Ok(())
	}

	#[test]
	fn test_lora_from_memory() -> crate::Result<()> {
		let model = Session::builder()?.commit_from_file("tests/data/lora_model.onnx")?;

		let lora_bytes = fs::read("tests/data/adapter.orl").expect("");
		let lora = Adapter::from_memory(&lora_bytes, None)?;
		drop(lora_bytes);

		let mut run_options = RunOptions::new()?;
		run_options.add_adapter(&lora)?;

		let output: Tensor<f32> = model
			.run_with_options(crate::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?]?, &run_options)?
			.remove("output")
			.expect("")
			.downcast()?;
		let (_, output) = output.extract_raw_tensor();
		assert_eq!(output[0], 154.0);
		assert_eq!(output[1], 176.0);
		assert_eq!(output[2], 198.0);
		assert_eq!(output[3], 220.0);

		Ok(())
	}
}
