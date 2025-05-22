//! An input adapter, allowing for loading many static inputs from disk at once.

use alloc::sync::Arc;
use core::ptr::{self, NonNull};
#[cfg(feature = "std")]
use std::path::Path;

use crate::{AsPointer, Result, memory::Allocator, ortsys};

#[derive(Debug)]
pub(crate) struct AdapterInner {
	ptr: NonNull<ort_sys::OrtLoraAdapter>
}

impl AsPointer for AdapterInner {
	type Sys = ort_sys::OrtLoraAdapter;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for AdapterInner {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseLoraAdapter(self.ptr.as_ptr())];
	}
}

/// An input adapter, allowing for loading many static inputs from disk at once.
///
/// [`Adapter`] essentially acts as a collection of predefined inputs allocated on a specific device that can easily be
/// swapped out between session runs via [`RunOptions::add_adapter`]. With slight modifications to the session
/// graph, [`Adapter`]s can be used as low-rank adapters (LoRAs) or as containers of style embeddings.
///
/// # Example
/// ```
/// # use ort::{adapter::Adapter, session::{run_options::RunOptions, Session}, value::Tensor};
/// # fn main() -> ort::Result<()> {
/// let mut model = Session::builder()?.commit_from_file("tests/data/lora_model.onnx")?;
/// let lora = Adapter::from_file("tests/data/adapter.orl", None)?;
///
/// let mut run_options = RunOptions::new()?;
/// run_options.add_adapter(&lora)?;
///
/// let outputs =
/// 	model.run_with_options(ort::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?], &run_options)?;
/// # Ok(())
/// # }
/// ```
///
/// [`RunOptions::add_adapter`]: crate::session::run_options::RunOptions::add_adapter
#[derive(Debug, Clone)]
pub struct Adapter {
	pub(crate) inner: Arc<AdapterInner>
}

impl Adapter {
	/// Loads an [`Adapter`] from a file.
	///
	/// An optional [`Allocator`] can be provided to specify the device on which the inputs should be allocated on.
	/// Note that providing a CPU allocator will return an error; only device allocators are expected.
	///
	/// ```
	/// # use ort::{
	/// # 	adapter::Adapter,
	/// # 	execution_providers::CUDAExecutionProvider,
	/// # 	memory::DeviceType,
	/// # 	session::{run_options::RunOptions, Session},
	/// # 	value::Tensor
	/// # };
	/// # fn main() -> ort::Result<()> {
	/// let mut model = Session::builder()?
	/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])?
	/// 	.commit_from_file("tests/data/lora_model.onnx")?;
	///
	/// let allocator = model.allocator();
	/// let lora = Adapter::from_file(
	/// 	"tests/data/adapter.orl",
	/// 	if allocator.memory_info().device_type() == DeviceType::CPU { None } else { Some(&allocator) }
	/// )?;
	///
	/// let mut run_options = RunOptions::new()?;
	/// run_options.add_adapter(&lora)?;
	///
	/// let outputs =
	/// 	model.run_with_options(ort::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?], &run_options)?;
	/// # Ok(())
	/// # }
	/// ```
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn from_file(path: impl AsRef<Path>, allocator: Option<&Allocator>) -> Result<Self> {
		let path = crate::util::path_to_os_char(path);
		let allocator_ptr = allocator.map(|c| c.ptr().cast_mut()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapter(path.as_ptr(), allocator_ptr, &mut ptr)?; nonNull(ptr)];
		Ok(Adapter {
			inner: Arc::new(AdapterInner { ptr })
		})
	}

	/// Loads an [`Adapter`] from memory. The adapter's values will be **copied**, either to the CPU or the given
	/// allocator if one is provided.
	///
	/// An [`Allocator`] can be provided to specify the device on which the inputs should be allocated on.
	/// Note that providing a CPU allocator will return an error; only device allocators are expected.
	///
	/// ```
	/// # use ort::{
	/// # 	adapter::Adapter,
	/// # 	execution_providers::CUDAExecutionProvider,
	/// # 	memory::DeviceType,
	/// # 	session::{run_options::RunOptions, Session},
	/// # 	value::Tensor
	/// # };
	/// # fn main() -> ort::Result<()> {
	/// let mut model = Session::builder()?
	/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])?
	/// 	.commit_from_file("tests/data/lora_model.onnx")?;
	///
	/// let bytes = std::fs::read("tests/data/adapter.orl").unwrap();
	/// let allocator = model.allocator();
	/// let lora = Adapter::from_memory(
	/// 	&bytes,
	/// 	if allocator.memory_info().device_type() == DeviceType::CPU { None } else { Some(&allocator) }
	/// )?;
	///
	/// let mut run_options = RunOptions::new()?;
	/// run_options.add_adapter(&lora)?;
	///
	/// let outputs =
	/// 	model.run_with_options(ort::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?], &run_options)?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn from_memory(bytes: &[u8], allocator: Option<&Allocator>) -> Result<Self> {
		let allocator_ptr = allocator.map(|c| c.ptr().cast_mut()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapterFromArray(bytes.as_ptr().cast(), bytes.len(), allocator_ptr, &mut ptr)?; nonNull(ptr)];
		Ok(Adapter {
			inner: Arc::new(AdapterInner { ptr })
		})
	}
}

impl AsPointer for Adapter {
	type Sys = ort_sys::OrtLoraAdapter;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.ptr()
	}
}

#[cfg(test)]
mod tests {
	use super::Adapter;
	use crate::{
		session::{RunOptions, Session},
		value::Tensor
	};

	#[test]
	#[cfg(feature = "std")]
	fn test_lora() -> crate::Result<()> {
		let model = std::fs::read("tests/data/lora_model.onnx").expect("");
		let mut session = Session::builder()?.commit_from_memory(&model)?;
		let lora = Adapter::from_file("tests/data/adapter.orl", None)?;

		let mut run_options = RunOptions::new()?;
		run_options.add_adapter(&lora)?;

		let output: Tensor<f32> = session
			.run_with_options(crate::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?], &run_options)?
			.remove("output")
			.expect("")
			.downcast()?;
		let (_, output) = output.extract_tensor();
		assert_eq!(output[0], 154.0);
		assert_eq!(output[1], 176.0);
		assert_eq!(output[2], 198.0);
		assert_eq!(output[3], 220.0);

		Ok(())
	}

	#[test]
	fn test_lora_from_memory() -> crate::Result<()> {
		let model = std::fs::read("tests/data/lora_model.onnx").expect("");
		let mut session = Session::builder()?.commit_from_memory(&model)?;

		let lora_bytes = std::fs::read("tests/data/adapter.orl").expect("");
		let lora = Adapter::from_memory(&lora_bytes, None)?;
		drop(lora_bytes);

		let mut run_options = RunOptions::new()?;
		run_options.add_adapter(&lora)?;

		let output: Tensor<f32> = session
			.run_with_options(crate::inputs![Tensor::<f32>::from_array(([4, 4], vec![1.0; 16]))?], &run_options)?
			.remove("output")
			.expect("")
			.downcast()?;
		let (_, output) = output.extract_tensor();
		assert_eq!(output[0], 154.0);
		assert_eq!(output[1], 176.0);
		assert_eq!(output[2], 198.0);
		assert_eq!(output[3], 220.0);

		Ok(())
	}
}
