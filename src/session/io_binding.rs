use alloc::{string::String, sync::Arc};
use core::{
	cell::Cell,
	fmt::Debug,
	marker::PhantomData,
	mem,
	ptr::{self, NonNull}
};

use crate::{
	AsPointer, Error, ErrorCode,
	error::Result,
	memory::MemoryInfo,
	ortsys,
	session::{Session, SharedSessionInner},
	util::{MiniMap, with_cstr},
	value::{DynValueRef, DynValueRefMut, Value, ValueInner, ValueRef, ValueRefMut, ValueTypeMarker}
};

/// Enables binding of session inputs and/or outputs to pre-allocated memory.
///
/// [`IoBinding`] minimizes copies between a device (like a GPU) and the host (CPU) by allowing you to bind a
/// certain input/output to a pre-allocated value on a specific device.
///
/// [`IoBinding`] is most suitable for:
/// - An ensemble of models in which the output from one model is the input to another and does not need to pass through
///   the CPU to perform additional processing.
/// - Situations where an output should stay on a device (e.g. to perform additional hardware-accelerated processing).
/// - Models that accept an input that does not change for multiple subsequent runs (like the conditional embedding for
///   a diffusion model).
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
/// # 	ep,
/// # 	memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType},
/// # 	session::{Session, IoBinding},
/// # 	value::Tensor
/// # };
/// # fn main() -> ort::Result<()> {
/// # let env = ort::test_util::test_env().clone();
/// let mut text_encoder = Session::builder(&env)?
/// 	.with_execution_providers([
/// 		#[cfg(feature = "cuda")]
/// 		ep::CUDA::default().build()
/// 	])?
/// 	.commit_from_file("text_encoder.onnx")?;
/// let mut unet = Session::builder(&env)?
/// 	.with_execution_providers([
/// 		#[cfg(feature = "cuda")]
/// 		ep::CUDA::default().build()
/// 	])?
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
/// let mut io_binding = unet.create_binding()?;
/// io_binding.bind_input("condition", text_condition)?;
///
/// let input_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
/// )?;
/// io_binding.bind_input("latents", Tensor::<f32>::new(&input_allocator, [1_usize, 4, 64, 64])?)?;
///
/// let output_allocator = Allocator::new(
/// 	&unet,
/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUOutput)?
/// )?;
/// io_binding.bind_output("noise_pred", Tensor::<f32>::new(&output_allocator, [1_usize, 4, 64, 64])?)?;
///
/// for _ in 0..20 {
/// 	let noise_pred = unet.run_binding(&io_binding)?.remove("noise_pred").unwrap();
///
/// 	// changes made to the bound input will be seen by the next run
/// 	let mut latents = io_binding.input_mut("latents")?;
/// 	let mut latents = latents.try_extract_array_mut::<f32>()?;
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
	held_inputs: MiniMap<String, BoundInput>,
	pub(crate) output_values: MiniMap<String, Option<Arc<ValueInner>>>,
	maybe_pending_copies: Cell<bool>,
	_session: Arc<SharedSessionInner>
}

#[derive(Debug)]
struct BoundInput {
	inner: Arc<ValueInner>,
	mutable: bool,
	dirty: Cell<bool>
}

/// A value which can be bound to a session input via [`IoBinding::bind_input`].
#[diagnostic::on_unimplemented(
	message = "`{Self}` cannot be bound as an input",
	note = "only owned values or (unsafe) 'static views can be bound"
)]
pub trait BindableInput {
	#[doc(hidden)]
	#[allow(private_interfaces)]
	fn into_bound_input(self) -> (Arc<ValueInner>, bool);

	private_trait!();
}

impl<Type: ValueTypeMarker + ?Sized> BindableInput for Value<Type> {
	#[allow(private_interfaces)]
	fn into_bound_input(self) -> (Arc<ValueInner>, bool) {
		(self.inner, true)
	}

	private_impl!();
}

impl<Type: ValueTypeMarker + ?Sized> BindableInput for ValueRef<'static, Type> {
	#[allow(private_interfaces)]
	fn into_bound_input(self) -> (Arc<ValueInner>, bool) {
		(self.into_inner().inner, false)
	}

	private_impl!();
}

impl<Type: ValueTypeMarker + ?Sized> BindableInput for ValueRefMut<'static, Type> {
	#[allow(private_interfaces)]
	fn into_bound_input(self) -> (Arc<ValueInner>, bool) {
		(self.inner.inner, true)
	}

	private_impl!();
}

/// A value which can be bound to a session output via [`IoBinding::bind_output`].
#[diagnostic::on_unimplemented(
	message = "`{Self}` cannot be bound as an output",
	note = "only owned values or (unsafe) 'static mutable views can be bound"
)]
pub trait BindableOutput {
	#[doc(hidden)]
	#[allow(private_interfaces)]
	fn into_bound_output(self) -> Arc<ValueInner>;

	private_trait!();
}

impl<Type: ValueTypeMarker + ?Sized> BindableOutput for Value<Type> {
	#[allow(private_interfaces)]
	fn into_bound_output(self) -> Arc<ValueInner> {
		self.inner
	}

	private_impl!();
}

impl<Type: ValueTypeMarker + ?Sized> BindableOutput for ValueRefMut<'static, Type> {
	#[allow(private_interfaces)]
	fn into_bound_output(self) -> Arc<ValueInner> {
		self.inner.inner
	}

	private_impl!();
}

impl IoBinding {
	pub(crate) fn new(session: &Session) -> Result<Self> {
		let mut ptr: *mut ort_sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.ptr().cast_mut(), &mut ptr)?; nonNull(ptr)];
		crate::logging::create!(IoBinding, ptr);
		Ok(Self {
			ptr,
			held_inputs: MiniMap::new(),
			output_values: MiniMap::new(),
			maybe_pending_copies: Cell::new(false),
			_session: session.inner()
		})
	}

	/// Bind a [`Value`] to a session input.
	///
	/// The value given here will be passed as the input specified by `name` in all future invocations of
	/// [`Session::run_binding`] until this input is overridden (by calling [`IoBinding::bind_input`] again) or until
	/// all inputs are cleared (via [`IoBinding::clear_inputs`] or [`IoBinding::clear`]). The binding takes ownership
	/// of the value; it can be accessed later with [`IoBinding::input`] or [`IoBinding::input_mut`].
	///
	/// Only owned values and `'static` views can be bound. To bind a raw device buffer that you manage yourself without
	/// copying it, create a [`TensorRefMut`](crate::value::TensorRefMut) with
	/// [`TensorRefMut::from_raw`](crate::value::TensorRefMut::from_raw) and a `'static` lifetime; you must then ensure
	/// the buffer stays valid for as long as the binding uses it.
	///
	/// If the value already resides on the device the session wants the input on, it is used directly without a copy.
	/// Otherwise (as is often the case for CPU inputs to CUDA sessions), the data will be queued to be asynchronously
	/// copied to that device. That copy is only made when the input is bound, so changes made to the value through
	/// [`IoBinding::input_mut`] cause it to be bound (and thus copied) again before the next run.
	pub fn bind_input<S: Into<String>>(&mut self, name: S, value: impl BindableInput) -> Result<()> {
		let name: String = name.into();
		let (inner, mutable) = value.into_bound_input();
		unsafe { self.bind_input_inner(name.as_bytes(), &inner) }?;
		self.maybe_pending_copies.set(true);

		if let Some(prev) = self.held_inputs.insert(
			name,
			BoundInput {
				inner,
				mutable,
				dirty: Cell::new(false)
			}
		) && !self.ensure_inputs_synchronized()
		{
			crate::warn!("bound input failed to synchronize; its previous value will be leaked!");
			mem::forget(prev);
		}
		Ok(())
	}

	pub(crate) unsafe fn bind_input_inner<S: AsRef<[u8]>>(&self, name: S, value: &ValueInner) -> Result<()> {
		let ptr = self.ptr.as_ptr();
		with_cstr(name.as_ref(), &|name| {
			ortsys![unsafe BindInput(ptr, name.as_ptr(), value.ptr())?];
			Ok(())
		})?;
		Ok(())
	}

	/// Returns a view of the value bound to the input `name`, if any.
	pub fn input(&self, name: &str) -> Option<DynValueRef<'_>> {
		self.held_inputs.get(name).map(|input| {
			ValueRef::new(Value {
				inner: Arc::clone(&input.inner),
				_markers: PhantomData
			})
		})
	}

	/// Returns a mutable view of the value bound to the input `name`.
	///
	/// Changes made to the value through this view will be seen by the next call to [`Session::run_binding`]. If ONNX
	/// Runtime had to copy the input to another device, it will be bound (and thus copied) again before the next
	/// run.
	pub fn input_mut(&mut self, name: &str) -> Result<DynValueRefMut<'_>> {
		let Some(input) = self.held_inputs.get(name) else {
			return Err(Error::new_with_code(ErrorCode::InvalidArgument, alloc::format!("no value is bound to input '{name}'")));
		};
		if !input.mutable {
			return Err(Error::new_with_code(ErrorCode::InvalidArgument, alloc::format!("input '{name}' was bound from an immutable view")));
		}

		// make sure we're not in the middle of a copy
		if self.maybe_pending_copies.get() {
			self.synchronize_inputs()?;
		}

		input.dirty.set(true);
		Ok(ValueRefMut::new(Value {
			inner: Arc::clone(&input.inner),
			_markers: PhantomData
		}))
	}

	pub(crate) fn rebind_dirty_inputs(&self) -> Result<()> {
		for (name, input) in self.held_inputs.iter() {
			if input.dirty.get() {
				unsafe { self.bind_input_inner(name.as_bytes(), &input.inner) }?;
				self.maybe_pending_copies.set(true);
				input.dirty.set(false);
			}
		}
		Ok(())
	}

	fn ensure_inputs_synchronized(&self) -> bool {
		if !self.maybe_pending_copies.get() {
			return true;
		}
		self.synchronize_inputs().is_ok()
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
	pub fn bind_output<S: Into<String>>(&mut self, name: S, value: impl BindableOutput) -> Result<()> {
		let inner = value.into_bound_output();
		if inner.is_backed() {
			// this is purely because i am too lazy to modify `run_binding` to have preallocated outputs share the
			// original value's backing
			return Err(Error::new_with_code(ErrorCode::InvalidArgument, "only tensors created with Tensor::new can be bound to outputs"));
		}

		let name: String = name.into();
		unsafe { self.bind_output_inner(name.as_bytes(), &inner) }?;
		self.output_values.insert(name, Some(inner));
		Ok(())
	}

	pub(crate) unsafe fn bind_output_inner<S: AsRef<[u8]>>(&mut self, name: S, value: &ValueInner) -> Result<()> {
		let ptr = self.ptr_mut();
		with_cstr(name.as_ref(), &|name| {
			ortsys![unsafe BindOutput(ptr, name.as_ptr(), value.ptr())?];
			Ok(())
		})?;
		Ok(())
	}

	/// Bind a session output to a device which is specified by `mem_info`. Used when the shape of the output is
	/// unknown.
	///
	/// Note that the same output buffer will be reused across runs, so the output must be
	/// [`.clone()`](crate::value::Tensor::clone)d to avoid race conditions.
	///
	/// If the shape of the output is known, prefer [`IoBinding::bind_output`] instead.
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
		let synchronized = self.ensure_inputs_synchronized();
		ortsys![unsafe ClearBoundInputs(self.ptr_mut())];
		for (name, input) in self.held_inputs.drain() {
			if !synchronized {
				crate::warn!("failed to synchronize inputs before clear; input '{name}'s previous value will be leaked!");
				mem::forget(input);
			}
		}
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

	// technically the synchronize methods do not have to be &mut like the other methods, as they do not mutate any
	// state on the C side - all they do is iterate through all nodes' EPs and call their device synchronize
	// function (cudaDeviceSynchronize/hipDeviceSynchronize), which I assume would be thread-safe

	/// Synchronize all bound inputs, ensuring any pending asynchronous transfers are completed.
	pub fn synchronize_inputs(&self) -> Result<()> {
		ortsys![unsafe SynchronizeBoundInputs(self.ptr().cast_mut())?];
		self.maybe_pending_copies.set(false);
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
		crate::logging::drop!(IoBinding, self.ptr());
	}
}

#[cfg(test)]
mod tests {
	#[cfg(feature = "ndarray")]
	use crate::test_util::{mnist, test_env};
	use crate::{
		Result,
		memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
		session::Session,
		value::Tensor
	};

	// not terribly useful since CI is CPU-only, but it at least ensures the API won't segfault or something silly
	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_input_bound() -> Result<()> {
		let mut session = Session::builder(test_env())?.commit_from_url(mnist::MODEL_URL)?;

		let array = mnist::get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(session.inputs()[0].name(), Tensor::from_array(array)?)?;
		binding
			.bind_output_to_device(session.outputs()[0].name(), &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		let outputs = session.run_binding(&binding)?;
		let probabilities = mnist::extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_input_output_bound() -> Result<()> {
		let mut session = Session::builder(test_env())?.commit_from_url(mnist::MODEL_URL)?;

		let array = mnist::get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(session.inputs()[0].name(), Tensor::from_array(array)?)?;

		binding.bind_output(session.outputs()[0].name(), Tensor::<f32>::new(&Allocator::default(), [1i64, 10])?)?;

		let outputs = session.run_binding(&binding)?;
		let probabilities = mnist::extract_probabilities(&outputs[0])?;
		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_send_iobinding() -> Result<()> {
		let mut session = Session::builder(test_env())?.commit_from_url(mnist::MODEL_URL)?;

		let array = mnist::get_image();

		let mut binding = session.create_binding()?;
		binding.bind_output(session.outputs()[0].name(), Tensor::<f32>::new(&Allocator::default(), [1i64, 10])?)?;

		let probabilities = std::thread::spawn(move || {
			binding.bind_input(session.inputs()[0].name(), Tensor::from_array(array)?)?;
			let outputs = session.run_binding(&binding)?;
			let probabilities = mnist::extract_probabilities(&outputs[0])?;
			Ok::<Vec<(usize, f32)>, crate::Error>(probabilities)
		})
		.join()
		.expect("")?;

		assert_eq!(probabilities[0].0, 5);

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_clear_binds() -> Result<()> {
		let mut session = Session::builder(test_env())?.commit_from_url(mnist::MODEL_URL)?;

		let array = mnist::get_image();

		let mut binding = session.create_binding()?;
		binding.bind_input(session.inputs()[0].name(), Tensor::from_array(array)?)?;

		binding.bind_output(session.outputs()[0].name(), Tensor::<f32>::new(&Allocator::default(), [1i64, 10])?)?;

		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = mnist::extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_outputs();
		binding
			.bind_output_to_device(session.outputs()[0].name(), &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = mnist::extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		binding.clear_inputs();
		assert!(session.run_binding(&binding).is_err());

		Ok(())
	}

	#[test]
	#[cfg(all(feature = "ndarray", feature = "fetch-models"))]
	fn test_mnist_input_mut() -> Result<()> {
		use crate::value::TensorRef;

		let mut session = Session::builder(test_env())?.commit_from_url(mnist::MODEL_URL)?;
		let input_name = session.inputs()[0].name().to_string();

		let mut binding = session.create_binding()?;
		binding.bind_input(&input_name, Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 1, 28, 28)))?)?;
		binding
			.bind_output_to_device(session.outputs()[0].name(), &MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::CPUOutput)?)?;

		binding
			.input_mut(&input_name)?
			.try_extract_array_mut::<f32>()?
			.assign(&mnist::get_image());
		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = mnist::extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}
		assert_eq!(binding.input(&input_name).expect("").try_extract_array::<f32>()?, mnist::get_image().into_dyn());

		let data: &'static [f32] = Box::leak(mnist::get_image().into_raw_vec_and_offset().0.into_boxed_slice());
		binding.bind_input(&input_name, TensorRef::from_array_view(([1_usize, 1, 28, 28], data))?)?;
		assert!(binding.input_mut(&input_name).is_err());
		{
			let outputs = session.run_binding(&binding)?;
			let probabilities = mnist::extract_probabilities(&outputs[0])?;
			assert_eq!(probabilities[0].0, 5);
		}

		Ok(())
	}
}
