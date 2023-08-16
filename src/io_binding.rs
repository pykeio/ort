use std::{
	ffi::{c_char, c_void, CString},
	fmt::Debug,
	mem::ManuallyDrop,
	ptr,
	sync::Arc
};

use crate::{memory::MemoryInfo, ortsys, session::output::SessionOutputs, sys, value::Value, OrtError, OrtResult, Session};

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
	pub(crate) ptr: *mut sys::OrtIoBinding,
	session: &'s Session
}

impl<'s> IoBinding<'s> {
	pub(crate) fn new(session: &'s Session) -> OrtResult<Self> {
		let mut ptr: *mut sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.inner.session_ptr, &mut ptr) -> OrtError::CreateIoBinding; nonNull(ptr)];
		Ok(Self { ptr, session })
	}

	/// Bind a [`Value`] to a session input.
	pub fn bind_input<'a, 'b: 'a, S: AsRef<str> + Clone + Debug>(&'a mut self, name: S, ort_value: Value<'b>) -> OrtResult<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindInput(self.ptr, cname.as_ptr(), ort_value.ptr()) -> OrtError::CreateIoBinding];
		Ok(())
	}

	/// Bind a session output to a pre-allocated [`Value`].
	pub fn bind_output<'a, 'b: 'a, S: AsRef<str> + Clone + Debug>(&'a mut self, name: S, ort_value: Value<'b>) -> OrtResult<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutput(self.ptr, cname.as_ptr(), ort_value.ptr()) -> OrtError::CreateIoBinding];
		Ok(())
	}

	/// Bind a session output to a device which is specified by `mem_info`.
	pub fn bind_output_to_device<S: AsRef<str> + Clone + Debug>(&mut self, name: S, mem_info: MemoryInfo) -> OrtResult<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutputToDevice(self.ptr, cname.as_ptr(), mem_info.ptr) -> OrtError::CreateIoBinding];
		Ok(())
	}

	pub fn run(&self) -> OrtResult<SessionOutputs> {
		let run_options_ptr: *const sys::OrtRunOptions = std::ptr::null();
		ortsys![unsafe RunWithBinding(self.session.inner.session_ptr, run_options_ptr, self.ptr) -> OrtError::SessionRun];
		self.outputs()
	}

	pub fn outputs(&self) -> OrtResult<SessionOutputs> {
		let mut names_ptr: *mut c_char = ptr::null_mut();
		let mut lengths = Vec::new();
		let mut lengths_ptr = lengths.as_mut_ptr();
		let mut count = 0;

		ortsys![
			unsafe GetBoundOutputNames(
				self.ptr,
				self.session.allocator().ptr,
				&mut names_ptr,
				&mut lengths_ptr,
				&mut count
			) -> OrtError::CreateIoBinding;
			nonNull(names_ptr)
		];
		if count > 0 {
			let lengths = unsafe { std::slice::from_raw_parts(lengths_ptr, count).to_vec() };
			let output_names = unsafe { ManuallyDrop::new(String::from_raw_parts(names_ptr as *mut u8, lengths.iter().sum(), lengths.iter().sum())) };
			let mut output_names_chars = output_names.chars();

			let output_names = lengths
				.into_iter()
				.map(|length| output_names_chars.by_ref().take(length).collect::<String>())
				.collect::<Vec<_>>();

			ortsys![unsafe AllocatorFree(self.session.allocator().ptr, names_ptr as *mut c_void) -> OrtError::CreateIoBinding];

			let mut output_values_ptr: *mut *mut sys::OrtValue = vec![ptr::null_mut(); count].as_mut_ptr();
			ortsys![unsafe GetBoundOutputValues(self.ptr, self.session.allocator().ptr, &mut output_values_ptr, &mut count) -> OrtError::CreateIoBinding; nonNull(output_values_ptr)];

			let output_values_ptr = unsafe { std::slice::from_raw_parts(output_values_ptr, count).to_vec() }
				.into_iter()
				.map(|v| unsafe { Value::from_raw(v, Arc::clone(&self.session.inner)) });

			Ok(SessionOutputs::new(output_names, output_values_ptr))
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
