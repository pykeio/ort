#![allow(deprecated)]

use super::ExecutionProvider;
use crate::{
	AsPointer,
	error::{Error, Result},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct VSINPU;

super::impl_ep!(VSINPU);

impl ExecutionProvider for VSINPU {
	fn name(&self) -> &'static str {
		"VSINPUExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_VSINPU(options: *mut ort_sys::OrtSessionOptions) -> ort_sys::OrtStatusPtr);
		unsafe { Error::result_from_status(OrtSessionOptionsAppendExecutionProvider_VSINPU(session_builder.ptr_mut())) }
	}
}
