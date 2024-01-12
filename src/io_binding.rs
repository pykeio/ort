use std::{ffi::CString, fmt::Debug, ptr, sync::Arc};

use crate::{
	memory::MemoryInfo,
	ortsys,
	session::{output::SessionOutputs, RunOptions},
	value::Value,
	Error, Result, Session
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
	pub(crate) ptr: *mut ort_sys::OrtIoBinding,
	session: &'s Session,
	input_values: Vec<Value>,
	output_names: Vec<String>
}

impl<'s> IoBinding<'s> {
	pub(crate) fn new(session: &'s Session) -> Result<Self> {
		let mut ptr: *mut ort_sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.inner.session_ptr, &mut ptr) -> Error::CreateIoBinding; nonNull(ptr)];
		Ok(Self {
			ptr,
			session,
			input_values: Vec::new(),
			output_names: Vec::new()
		})
	}

	/// Bind a [`Value`] to a session input.
	pub fn bind_input<S: AsRef<str>>(&mut self, name: S, ort_value: Value) -> Result<&mut Value> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindInput(self.ptr, cname.as_ptr(), ort_value.ptr()) -> Error::BindInput];
		self.input_values.push(ort_value);
		Ok(self.input_values.last_mut().unwrap())
	}

	/// Bind a session output to a pre-allocated [`Value`].
	pub fn bind_output<'o: 's, S: AsRef<str>>(&mut self, name: S, ort_value: &'o mut Value) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutput(self.ptr, cname.as_ptr(), ort_value.ptr()) -> Error::BindOutput];
		self.output_names.push(name.to_string());
		Ok(())
	}

	/// Bind a session output to a device which is specified by `mem_info`.
	pub fn bind_output_to_device<S: AsRef<str>>(&mut self, name: S, mem_info: MemoryInfo) -> Result<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutputToDevice(self.ptr, cname.as_ptr(), mem_info.ptr) -> Error::BindOutput];
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
			run_options.run_options_ptr
		} else {
			std::ptr::null_mut()
		};
		ortsys![unsafe RunWithBinding(self.session.inner.session_ptr, run_options_ptr, self.ptr) -> Error::SessionRunWithIoBinding];

		let mut count = self.output_names.len() as ort_sys::size_t;
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			let allocator = self.session.allocator();
			ortsys![unsafe GetBoundOutputValues(self.ptr, allocator.ptr, &mut output_values_ptr, &mut count) -> Error::GetBoundOutputs; nonNull(output_values_ptr)];

			let output_values = unsafe { std::slice::from_raw_parts(output_values_ptr, count as _).to_vec() }
				.into_iter()
				.map(|v| unsafe { Value::from_raw(v, Arc::clone(&self.session.inner)) });

			// output values will be freed when the `Value`s in `SessionOutputs` drop

			Ok(SessionOutputs::new_backed(self.output_names.iter().map(|c| c.as_str()), output_values, allocator.ptr, output_values_ptr as *mut _))
		} else {
			Ok(SessionOutputs::new_empty())
		}
	}
}

impl<'s> Drop for IoBinding<'s> {
	fn drop(&mut self) {
		if !self.ptr.is_null() {
			ortsys![unsafe ReleaseIoBinding(self.ptr)];
		}
		self.ptr = ptr::null_mut();
	}
}
