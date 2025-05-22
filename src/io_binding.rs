//! Enables binding of session inputs and/or outputs to pre-allocated memory.

use alloc::{string::String, sync::Arc};
use core::{
	fmt::Debug,
	ptr::{self, NonNull}
};

use crate::{
	AsPointer,
	error::Result,
	memory::MemoryInfo,
	ortsys,
	session::{Session, SharedSessionInner},
	util::{MiniMap, with_cstr},
	value::{DynValue, Value, ValueInner, ValueTypeMarker}
};

/// Enables binding of session inputs and/or outputs to pre-allocated memory.
///
/// [`IoBinding`] minimizes copies between a device (like a GPU) and the host (CPU) by allowing the user to bind a
/// certain input/output to a pre-allocated value on a specific device.
///
/// [`IoBinding`] is most suitable for:
/// - An ensemble of models in which the output from one model is the input of another and does not need to pass through
///   the CPU to perform additional processing.
/// - Situations where the output should stay on a device (e.g. to perform additional processing with CUDA).
/// - Diffusion models, for instance, that accept an unchanging embedding for conditioning.
///
/// [`IoBinding`] will not provide any meaningful benefit for:
/// - Models where every input changes with each invocation, such as a causal language model or object recognition
///   model.
/// - Pipelines that go straight from CPU -> GPU -> CPU.
///
/// # Example
/// A diffusion model which takes a text condition input.
///
/// ```no_run
/// # use ort::{
/// # 	execution_providers::CUDAExecutionProvider,
/// # 	io_binding::IoBinding,
/// # 	memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType},
/// # 	session::Session,
/// # 	value::Tensor
/// # };
/// # fn main() -> ort::Result<()> {
/// let mut text_encoder = Session::builder()?
/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])?
/// 	.commit_from_file("text_encoder.onnx")?;
/// let mut unet = Session::builder()?
/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])?
/// 	.commit_from_file("unet.onnx")?;
///
/// let text_condition = text_encoder
/// 	.run(ort::inputs![Tensor::<i64>::from_array((
/// 		vec![27],
/// 		vec![
/// 			23763, 15460, 473, 68, 312, 265, 17463, 4098, 304, 1077, 283, 198, 7676, 5976, 272, 285, 3609, 435,
/// 			21680, 321, 265, 300, 1689, 64, 285, 4763, 64
/// 		]
/// 	))?])?
/// 	.remove("output0")
/// 	.unwrap();
///
/// let input_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
/// )?;
/// let mut latents = Tensor::<f32>::new(&input_allocator, [1_usize, 4, 64, 64])?;
///
/// let mut io_binding = unet.create_binding()?;
/// io_binding.bind_input("condition", &text_condition)?;
///
/// let output_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUOutput)?
/// )?;
/// io_binding.bind_output("noise_pred", Tensor::<f32>::new(&output_allocator, [1_usize, 4, 64, 64])?)?;
///
/// for _ in 0..20 {
/// 	io_binding.bind_input("latents", &latents)?;
/// 	let noise_pred = unet.run_binding(&io_binding)?.remove("noise_pred").unwrap();
///
/// 	let mut latents = latents.extract_array_mut();
/// 	latents += &noise_pred.try_extract_array::<f32>()?;
/// }
/// # Ok(())
/// # }
/// ```
///
/// [`IoBinding`] may provide a decent speedup in this example since the `condition` tensor is unchanging between runs.
/// If we were to use normal session inference, the `condition` tensor would be needlessly copied with each invocation
/// of `unet.run()`, and this copying can come with significant latency & overhead. With [`IoBinding`], the `condition`
/// tensor is only copied to the device once instead of 20 times.
#[derive(Debug)]
pub struct IoBinding {
	ptr: NonNull<ort_sys::OrtIoBinding>,
	held_inputs: MiniMap<String, Arc<ValueInner>>,
	pub(crate) output_values: MiniMap<String, Option<DynValue>>,
	_session: Arc<SharedSessionInner>
}

impl IoBinding {
	pub(crate) fn new(session: &Session) -> Result<Self> {
		let mut ptr: *mut ort_sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.ptr().cast_mut(), &mut ptr)?; nonNull(ptr)];
		Ok(Self {
			ptr,
			held_inputs: MiniMap::new(),
			output_values: MiniMap::new(),
			_session: session.inner()
		})
	}

	/// Bind a [`Value`] to a session input.
	///
	/// Upon invocation, the value's data will be queued to be copied to the device the session is allocated on. The
	/// copied data will be used as an input (specified by `name`) in all future invocations of [`Session::run_binding`]
	/// until the input is overridden (by calling [`IoBinding::bind_input`] again) or until all inputs are cleared (via
	/// [`IoBinding::clear_inputs`] or [`IoBinding::clear`]).
	///
	/// The data is only copied upon invocation of this function. Any changes to the given value afterwards will not
	/// affect the data seen by the session until the value is re-bound.
	pub fn bind_input<T: ValueTypeMarker + ?Sized, S: Into<String>>(&mut self, name: S, ort_value: &Value<T>) -> Result<()> {
		let name: String = name.into();
		let ptr = self.ptr_mut();
		with_cstr(name.as_bytes(), &|name| {
			ortsys![unsafe BindInput(ptr, name.as_ptr(), ort_value.ptr())?];
			Ok(())
		})?;
		self.held_inputs.insert(name, Arc::clone(&ort_value.inner));
		Ok(())
	}

	/// Bind a session output to a pre-allocated [`Value`].
	///
	/// This allows for the pre-allocation and reuse of memory in the session output (see [`Tensor::new`]). Any
	/// subsequent runs via [`Session::run_binding`] will reuse the same tensor to store the output instead of creating
	/// a new one each time.
	///
	/// The output will be accessible in the value returned by [`Session::run_binding`], under the name specified by
	/// `name`.
	///
	/// [`Tensor::new`]: crate::value::Tensor::new
	pub fn bind_output<T: ValueTypeMarker + ?Sized, S: Into<String>>(&mut self, name: S, mut ort_value: Value<T>) -> Result<()> {
		let name: String = name.into();
		unsafe { self.bind_output_mut(name.as_bytes(), &mut ort_value) }?;
		self.output_values.insert(name, Some(ort_value.into_dyn()));
		Ok(())
	}

	pub(crate) unsafe fn bind_output_mut<T: ValueTypeMarker + ?Sized, S: AsRef<[u8]>>(&mut self, name: S, ort_value: &mut Value<T>) -> Result<()> {
		let ptr = self.ptr_mut();
		with_cstr(name.as_ref(), &|name| {
			ortsys![unsafe BindOutput(ptr, name.as_ptr(), ort_value.ptr())?];
			Ok(())
		})?;
		Ok(())
	}

	/// Bind a session output to a device which is specified by `mem_info`.
	pub fn bind_output_to_device<S: Into<String>>(&mut self, name: S, mem_info: &MemoryInfo) -> Result<()> {
		let name: String = name.into();
		let ptr = self.ptr_mut();
		with_cstr(name.as_bytes(), &|name| {
			ortsys![unsafe BindOutputToDevice(ptr, name.as_ptr(), mem_info.ptr())?];
			Ok(())
		})?;
		self.output_values.insert(name, None);
		Ok(())
	}

	/// Clears all bound inputs specified by [`IoBinding::bind_input`].
	pub fn clear_inputs(&mut self) {
		ortsys![unsafe ClearBoundInputs(self.ptr_mut())];
		drop(self.held_inputs.drain());
	}
	/// Clears all bound outputs specified by [`IoBinding::bind_output`] or [`IoBinding::bind_output_to_device`].
	pub fn clear_outputs(&mut self) {
		ortsys![unsafe ClearBoundOutputs(self.ptr_mut())];
		drop(self.output_values.drain());
	}
	/// Clears both the bound inputs & outputs; equivalent to [`IoBinding::clear_inputs`] followed by
	/// [`IoBinding::clear_outputs`].
	pub fn clear(&mut self) {
		self.clear_inputs();
		self.clear_outputs();
	}

	// technically the synchronize methods do not have to be &mut like the other methods, as they do not mutate any state
	// on the C side - all they do is iterate through all nodes' EPs and call their device synchronize function
	// (cudaDeviceSynchronize/hipDeviceSynchronize), which I assume would be thread-safe

	/// Synchronize all bound inputs, ensuring any pending asynchronous transfers are completed.
	pub fn synchronize_inputs(&self) -> Result<()> {
		ortsys![unsafe SynchronizeBoundInputs(self.ptr().cast_mut())?];
		Ok(())
	}
	/// Synchronize all bound outputs, ensuring any pending asynchronous transfers are completed.
	pub fn synchronize_outputs(&self) -> Result<()> {
		ortsys![unsafe SynchronizeBoundOutputs(self.ptr().cast_mut())?];
		Ok(())
	}
	/// Synchronizes both inputs & outputs; equivalent to [`IoBinding::synchronize_inputs`] followed by
	/// [`IoBinding::synchronize_outputs`].
	pub fn synchronize(&self) -> Result<()> {
		self.synchronize_inputs()?;
		self.synchronize_outputs()?;
		Ok(())
	}
}

unsafe impl Send for IoBinding {}

impl AsPointer for IoBinding {
	type Sys = ort_sys::OrtIoBinding;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for IoBinding {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseIoBinding(self.ptr_mut())];
	}
}

#[cfg(test)]
mod tests {
	use core::cmp::Ordering;

	use image::{ImageBuffer, Luma, Pixel};
	#[cfg(feature = "ndarray")]
	use ndarray::{Array2, Array4, Axis};

	#[cfg(feature = "ndarray")]
	use crate::tensor::ArrayExtensions;
	use crate::{
		Result,
		memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
		session::Session,
		value::{Tensor, TensorValueTypeMarker, Value}
	};

	#[cfg(feature = "ndarray")]
	fn get_image() -> Array4<f32> {
		let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open("tests/data/mnist_5.jpg").expect("failed to load image").to_luma8();
		ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();
			(channels[c] as f32) / 255.0
		})
	}

	#[cfg(feature = "ndarray")]
	fn extract_probabilities<T: TensorValueTypeMarker>(output: &Value<T>) -> Result<Vec<(usize, f32)>> {
		let mut probabilities: Vec<(usize, f32)> = output
			.try_extract_array()?
			.softmax(Axis(1))
			.iter()
			.copied()
			.enumerate()
			.collect::<Vec<_>>();
		probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
		Ok(probabilities)
	}

	// not terribly useful since CI is CPU-only, but it at least ensures the API won't segfault or something silly
	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_input_bound() -> Result<()> {
		let mut session = Session::builder()?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;
		binding.bind_output_to_device(&session.outputs[0].name, &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		let outputs = session.run_binding(&binding)?;
		let probabilities = extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_input_output_bound() -> Result<()> {
		let mut session = Session::builder()?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;

		let output = Array2::from_shape_simple_fn((1, 10), || 0.0_f32);
		binding.bind_output(&session.outputs[0].name, Tensor::from_array(output)?)?;

		let outputs = session.run_binding(&binding)?;
		let probabilities = extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_send_iobinding() -> Result<()> {
		let mut session = Session::builder()?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		let output = Array2::from_shape_simple_fn((1, 10), || 0.0_f32);
		binding.bind_output(&session.outputs[0].name, Tensor::from_array(output)?)?;

		let probabilities = std::thread::spawn(move || {
			binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;
			let outputs = session.run_binding(&binding)?;
			let probabilities = extract_probabilities(&outputs[0])?;
			Ok::<Vec<(usize, f32)>, crate::Error>(probabilities)
		})
		.join()
		.expect("")?;

		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_clear_bounds() -> Result<()> {
		let mut session = Session::builder()?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;

		let output = Array2::from_shape_simple_fn((1, 10), || 0.0_f32);
		binding.bind_output(&session.outputs[0].name, Tensor::from_array(output)?)?;

		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_outputs();
		binding.bind_output_to_device(&session.outputs[0].name, &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_inputs();
		assert!(session.run_binding(&binding).is_err());

		Ok(())
	}
}
