use std::{fmt::Debug, ptr};

use crate::{ortsys, sys, OrtError, OrtResult, Session};

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

	//#[allow(clippy::not_unsafe_ptr_arg_deref)]
	// pub fn bind_input<S: Into<String> + Clone + Debug>(&mut self, name: S, ort_value: &Value) -> OrtResult<()> {}
}
