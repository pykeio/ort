use std::{
	collections::HashMap,
	ffi::{c_char, c_void, CString},
	fmt::Debug,
	mem::ManuallyDrop,
	ptr,
	sync::Arc
};

use crate::{
	memory::MemoryInfo,
	ortsys,
	sys::{self, size_t},
	value::Value,
	OrtError, OrtResult, Session
};

#[derive(Debug)]
pub struct IoBinding<'s> {
	pub(crate) ptr: *mut sys::OrtIoBinding,
	session: &'s Session
}

impl<'s> IoBinding<'s> {
	pub(crate) fn new(session: &'s Session) -> OrtResult<Self> {
		let mut ptr: *mut sys::OrtIoBinding = ptr::null_mut();
		ortsys![unsafe CreateIoBinding(session.session_ptr.inner, &mut ptr) -> OrtError::CreateIoBinding; nonNull(ptr)];
		Ok(Self { ptr, session })
	}

	pub fn bind_input<'a, 'b: 'a, S: AsRef<str> + Clone + Debug>(&'a mut self, name: S, ort_value: Value<'b>) -> OrtResult<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindInput(self.ptr, cname.as_ptr(), ort_value.ptr()) -> OrtError::CreateIoBinding];
		Ok(())
	}

	pub fn bind_output<S: AsRef<str> + Clone + Debug>(&mut self, name: S, mem_info: MemoryInfo) -> OrtResult<()> {
		let name = name.as_ref();
		let cname = CString::new(name)?;
		ortsys![unsafe BindOutputToDevice(self.ptr, cname.as_ptr(), mem_info.ptr) -> OrtError::CreateIoBinding];
		Ok(())
	}

	pub fn outputs(&self) -> OrtResult<HashMap<String, Value<'static>>> {
		let mut names_ptr: *mut c_char = ptr::null_mut();
		let mut lengths = Vec::new();
		let mut lengths_ptr = lengths.as_mut_ptr();
		let mut count = 0;

		ortsys![
			unsafe GetBoundOutputNames(
				self.ptr,
				self.session.allocator(),
				&mut names_ptr,
				&mut lengths_ptr,
				&mut count
			) -> OrtError::CreateIoBinding;
			nonNull(names_ptr)
		];
		if count > 0 {
			let lengths = unsafe { std::slice::from_raw_parts(lengths_ptr, count as _).to_vec() };
			let output_names = unsafe {
				ManuallyDrop::new(String::from_raw_parts(names_ptr as *mut u8, lengths.iter().sum::<size_t>() as _, lengths.iter().sum::<size_t>() as _))
			};
			let mut output_names_chars = output_names.chars();

			let output_names = lengths
				.into_iter()
				.map(|length| output_names_chars.by_ref().take(length as _).collect::<String>())
				.collect::<Vec<_>>();

			ortsys![unsafe AllocatorFree(self.session.allocator(), names_ptr as *mut c_void) -> OrtError::CreateIoBinding];

			let mut output_values_ptr: *mut *mut sys::OrtValue = vec![ptr::null_mut(); count as _].as_mut_ptr();
			ortsys![unsafe GetBoundOutputValues(self.ptr, self.session.allocator(), &mut output_values_ptr, &mut count) -> OrtError::CreateIoBinding; nonNull(output_values_ptr)];

			let output_values_ptr = unsafe { std::slice::from_raw_parts(output_values_ptr, count as _).to_vec() }
				.into_iter()
				.map(|v| Value::from_raw(v, Arc::clone(&self.session.session_ptr)));

			Ok(output_names.into_iter().zip(output_values_ptr).collect::<HashMap<_, _>>())
		} else {
			Ok(HashMap::new())
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
