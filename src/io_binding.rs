use std::{
	ffi::CString,
	fmt::Debug,
	ptr::{self, NonNull},
	sync::Arc
};

use crate::{
	memory::MemoryInfo,
	ortsys,
	session::{output::SessionOutputs, RunOptions},
	value::{Value, ValueRefMut},
	Error, Result, Session, ValueTypeMarker
};

/// Enables binding of session inputs and/or outputs to pre-allocated memory.
///
/// Note that this arrangement is designed to minimize data copies, and to that effect, your memory allocations must
/// match what is expected by the model, whether you run on CPU or GPU. Data will still be copied if the
/// pre-allocated memory location does not match the one expected by the model. However, copies with `IoBinding`s are
/// only done once, at the time of the binding, not at run time. This means, that if your input data required a copy,
/// your further input modifications would not be seen by ONNX Runtime unless you rebind it, even if it is the same
/// buffer. If your scenario requires that the data is copied, `IoBinding` may not be the best match for your use case.
/// The fact that data copy is not made during runtime may also have performance implications.
#[derive(Debug)]
pub struct IoBinding<'s> {
	pub(crate) ptr: NonNull<ort_sys::OrtIoBinding>,
	session: &'s Session,
	output_names: Vec<String>
}

impl<'s> IoBinding<'s> {
	pub(crate) fn new(session: &'s Session) -> Result<Self> {
		let mut ptr: *mut ort_sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.inner.session_ptr.as_ptr(), &mut ptr) -> Error::CreateIoBinding; nonNull(ptr)];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(ptr) },
			session,
			output_names: Vec::new()
		})
	}

	/// Bind a [`Value`] to a session input.
	pub fn bind_input<'i: 's, T: ValueTypeMarker, S: AsRef<str>>(&mut self, name: S, ort_value: &'i mut Value<T>) -> Result<ValueRefMut<'i, T>> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindInput(self.ptr.as_ptr(), cname.as_ptr(), ort_value.ptr()) -> Error::BindInput];
		Ok(ort_value.view_mut())
	}

	/// Bind a session output to a pre-allocated [`Value`].
	pub fn bind_output<'o: 's, T: ValueTypeMarker, S: AsRef<str>>(&mut self, name: S, ort_value: &'o mut Value<T>) -> Result<ValueRefMut<'o, T>> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutput(self.ptr.as_ptr(), cname.as_ptr(), ort_value.ptr()) -> Error::BindOutput];
		self.output_names.push(name.to_string());
		Ok(ort_value.view_mut())
	}

	/// Bind a session output to a device which is specified by `mem_info`.
	pub fn bind_output_to_device<S: AsRef<str>>(&mut self, name: S, mem_info: &MemoryInfo) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutputToDevice(self.ptr.as_ptr(), cname.as_ptr(), mem_info.ptr.as_ptr()) -> Error::BindOutput];
		self.output_names.push(name.to_string());
		Ok(())
	}

	pub fn run<'i: 's>(&'i self) -> Result<SessionOutputs<'s>> {
		self.run_inner(None)
	}

	pub fn run_with_options<'i: 's>(&'i self, run_options: Arc<RunOptions>) -> Result<SessionOutputs<'s>> {
		self.run_inner(Some(run_options))
	}

	fn run_inner<'i: 's>(&'i self, run_options: Option<Arc<RunOptions>>) -> Result<SessionOutputs<'s>> {
		let run_options_ptr = if let Some(run_options) = run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};
		ortsys![unsafe RunWithBinding(self.session.inner.session_ptr.as_ptr(), run_options_ptr, self.ptr.as_ptr()) -> Error::SessionRunWithIoBinding];

		let mut count = self.output_names.len() as ort_sys::size_t;
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			let allocator = self.session.allocator();
			ortsys![unsafe GetBoundOutputValues(self.ptr.as_ptr(), allocator.ptr.as_ptr(), &mut output_values_ptr, &mut count) -> Error::GetBoundOutputs; nonNull(output_values_ptr)];

			let output_values = unsafe { std::slice::from_raw_parts(output_values_ptr, count as _).to_vec() }
				.into_iter()
				.map(|v| unsafe {
					Value::from_ptr(
						NonNull::new(v).expect("OrtValue ptrs returned by GetBoundOutputValues should not be null"),
						Some(Arc::clone(&self.session.inner))
					)
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
