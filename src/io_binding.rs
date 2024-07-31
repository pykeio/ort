use std::{
	collections::HashMap,
	ffi::CString,
	fmt::Debug,
	marker::PhantomData,
	ptr::{self, NonNull},
	sync::Arc
};

use crate::{
	error::{Error, Result},
	memory::MemoryInfo,
	ortsys,
	session::{output::SessionOutputs, NoSelectedOutputs, RunOptions, Session},
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
/// # use ort::{Allocator, AllocatorType, AllocationDevice, CUDAExecutionProvider, MemoryInfo, MemoryType, Session, Tensor, IoBinding};
/// # fn main() -> ort::Result<()> {
/// let text_encoder = Session::builder()?
/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])?
/// 	.commit_from_file("text_encoder.onnx")?;
/// let unet = Session::builder()?
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
/// 	))?]?)?
/// 	.remove("output0")
/// 	.unwrap();
///
/// let input_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDAPinned, 0, AllocatorType::Device, MemoryType::CPUInput)?
/// )?;
/// let mut latents = Tensor::<f32>::new(&input_allocator, [1, 4, 64, 64])?;
///
/// let mut io_binding = unet.create_binding()?;
/// io_binding.bind_input("condition", &text_condition)?;
///
/// let output_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDAPinned, 0, AllocatorType::Device, MemoryType::CPUOutput)?
/// )?;
/// io_binding.bind_output("noise_pred", Tensor::<f32>::new(&output_allocator, [1, 4, 64, 64])?)?;
///
/// for _ in 0..20 {
/// 	io_binding.bind_input("latents", &latents)?;
/// 	let noise_pred = io_binding.run()?.remove("noise_pred").unwrap();
///
/// 	let mut latents = latents.extract_tensor_mut();
/// 	latents += &noise_pred.try_extract_tensor::<f32>()?;
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
pub struct IoBinding<'s> {
	pub(crate) ptr: NonNull<ort_sys::OrtIoBinding>,
	session: &'s Session,
	held_inputs: HashMap<String, Arc<ValueInner>>,
	output_names: Vec<String>,
	output_values: HashMap<String, DynValue>
}

impl<'s> IoBinding<'s> {
	pub(crate) fn new(session: &'s Session) -> Result<Self> {
		let mut ptr: *mut ort_sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.inner.session_ptr.as_ptr(), &mut ptr) -> Error::CreateIoBinding; nonNull(ptr)];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(ptr) },
			session,
			held_inputs: HashMap::new(),
			output_names: Vec::new(),
			output_values: HashMap::new()
		})
	}

	/// Bind a [`Value`] to a session input.
	///
	/// Upon invocation, the value's data will be copied to the device the session is allocated on. The copied data will
	/// be used as an input (specified by `name`) in all future invocations of [`IoBinding::run`] until the input is
	/// overridden (by calling [`IoBinding::bind_input`] again) or until all inputs are cleared (via
	/// [`IoBinding::clear_inputs`] or [`IoBinding::clear`]).
	///
	/// The data is only copied **once**, immediately upon invocation of this function. Any changes to the given
	/// value afterwards will not affect the data seen by the session until the value is re-bound. Subsequent re-binds
	/// will still copy data, hence why [`IoBinding`] is really only suitable when one or more inputs do not change
	/// between runs.
	pub fn bind_input<T: ValueTypeMarker, S: AsRef<str>>(&mut self, name: S, ort_value: &Value<T>) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindInput(self.ptr.as_ptr(), cname.as_ptr(), ort_value.ptr()) -> Error::BindInput];
		self.held_inputs.insert(name.to_string(), Arc::clone(&ort_value.inner));
		Ok(())
	}

	/// Bind a session output to a pre-allocated [`Value`].
	///
	/// This allows for the pre-allocation and reuse of memory in the session output (see [`crate::Tensor::new`]). Any
	/// subsequent runs via [`IoBinding::run`] will reuse the same tensor to store the output instead of creating a new
	/// one each time.
	///
	/// The output will be accessible in the value returned by [`IoBinding::run`], under the name specified by `name`.
	pub fn bind_output<T: ValueTypeMarker, S: AsRef<str>>(&mut self, name: S, ort_value: Value<T>) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutput(self.ptr.as_ptr(), cname.as_ptr(), ort_value.ptr()) -> Error::BindOutput];
		self.output_names.push(name.to_string());
		// Clear the old bound output if we have any.
		drop(self.output_values.remove(name));
		self.output_values.insert(name.to_string(), ort_value.into_dyn());
		Ok(())
	}

	/// Bind a session output to a device which is specified by `mem_info`.
	pub fn bind_output_to_device<S: AsRef<str>>(&mut self, name: S, mem_info: &MemoryInfo) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutputToDevice(self.ptr.as_ptr(), cname.as_ptr(), mem_info.ptr.as_ptr()) -> Error::BindOutput];
		self.output_names.push(name.to_string());
		Ok(())
	}

	/// Clears all bound inputs specified by [`IoBinding::bind_input`].
	pub fn clear_inputs(&mut self) {
		ortsys![unsafe ClearBoundInputs(self.ptr.as_ptr())];
		drop(self.held_inputs.drain());
	}
	/// Clears all bound outputs specified by [`IoBinding::bind_output`] or [`IoBinding::bind_output_to_device`].
	pub fn clear_outputs(&mut self) {
		ortsys![unsafe ClearBoundOutputs(self.ptr.as_ptr())];
		drop(self.output_names.drain(..));
		drop(self.output_values.drain());
	}
	/// Clears both the bound inputs & outputs; equivalent to [`IoBinding::clear_inputs`] followed by
	/// [`IoBinding::clear_outputs`].
	pub fn clear(&mut self) {
		self.clear_inputs();
		self.clear_outputs();
	}

	/// Performs inference on the session using the bound inputs specified by [`IoBinding::bind_input`].
	pub fn run(&mut self) -> Result<SessionOutputs<'_, '_>> {
		self.run_inner(None)
	}

	/// Performs inference on the session using the bound inputs specified by [`IoBinding::bind_input`].
	pub fn run_with_options(&mut self, run_options: &RunOptions<NoSelectedOutputs>) -> Result<SessionOutputs<'_, 's>> {
		self.run_inner(Some(run_options))
	}

	fn run_inner(&mut self, run_options: Option<&RunOptions<NoSelectedOutputs>>) -> Result<SessionOutputs<'_, 's>> {
		let run_options_ptr = if let Some(run_options) = run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};
		ortsys![unsafe RunWithBinding(self.session.inner.session_ptr.as_ptr(), run_options_ptr, self.ptr.as_ptr()) -> Error::SessionRunWithIoBinding];

		let owned_ptrs: HashMap<*mut ort_sys::OrtValue, &Arc<ValueInner>> = self.output_values.values().map(|c| (c.ptr(), &c.inner)).collect();
		let mut count = self.output_names.len() as ort_sys::size_t;
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			let allocator = self.session.allocator();
			ortsys![unsafe GetBoundOutputValues(self.ptr.as_ptr(), allocator.ptr.as_ptr(), &mut output_values_ptr, &mut count) -> Error::GetBoundOutputs; nonNull(output_values_ptr)];

			let output_values = unsafe { std::slice::from_raw_parts(output_values_ptr, count as _).to_vec() }
				.into_iter()
				.map(|v| unsafe {
					if let Some(inner) = owned_ptrs.get(&v) {
						DynValue {
							inner: Arc::clone(*inner),
							_markers: PhantomData
						}
					} else {
						DynValue::from_ptr(
							NonNull::new(v).expect("OrtValue ptrs returned by GetBoundOutputValues should not be null"),
							Some(Arc::clone(&self.session.inner))
						)
					}
				});

			// output values will be freed when the `Value`s in `SessionOutputs` drop

			Ok(SessionOutputs::new_backed(self.output_names.iter().map(String::as_str), output_values, allocator, output_values_ptr.cast()))
		} else {
			Ok(SessionOutputs::new_empty())
		}
	}
}

impl<'s> Drop for IoBinding<'s> {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseIoBinding(self.ptr.as_ptr())];
	}
}

#[cfg(test)]
mod tests {
	use std::cmp::Ordering;

	use image::{ImageBuffer, Luma, Pixel};
	use ndarray::{Array2, Array4, Axis};

	use crate::{AllocationDevice, AllocatorType, ArrayExtensions, MemoryInfo, MemoryType, Result, Session, Tensor, TensorValueTypeMarker, Value};

	fn get_image() -> Array4<f32> {
		let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open("tests/data/mnist_5.jpg").expect("failed to load image").to_luma8();
		ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();
			(channels[c] as f32) / 255.0
		})
	}

	fn extract_probabilities<T: TensorValueTypeMarker>(output: &Value<T>) -> Result<Vec<(usize, f32)>> {
		let mut probabilities: Vec<(usize, f32)> = output
			.try_extract_tensor()?
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
	fn test_mnist_input_bound() -> Result<()> {
		let session = Session::builder()?.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;
		binding.bind_output_to_device(&session.outputs[0].name, &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		let outputs = binding.run()?;
		let probabilities = extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	fn test_mnist_input_output_bound() -> Result<()> {
		let session = Session::builder()?.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;

		let output = Array2::from_shape_simple_fn((1, 10), || 0.0_f32);
		binding.bind_output(&session.outputs[0].name, Tensor::from_array(output)?)?;

		let outputs = binding.run()?;
		let probabilities = extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	fn test_mnist_clears_bound() -> Result<()> {
		let session = Session::builder()?.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/mnist.onnx")?;

		let array = get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(&session.inputs[0].name, &Tensor::from_array(array)?)?;

		let output = Array2::from_shape_simple_fn((1, 10), || 0.0_f32);
		binding.bind_output(&session.outputs[0].name, Tensor::from_array(output)?)?;

		{
			let outputs = binding.run()?;
			let probabilities = extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_outputs();
		binding.bind_output_to_device(&session.outputs[0].name, &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		{
			let outputs = binding.run()?;
			let probabilities = extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_inputs();
		assert!(binding.run().is_err());

		Ok(())
	}
}
