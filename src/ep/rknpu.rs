use super::ExecutionProvider;
use crate::{
	AsPointer,
	error::{Error, Result},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct RKNPU {}

super::impl_ep!(RKNPU);

impl ExecutionProvider for RKNPU {
	fn name(&self) -> &'static str {
		"RknpuExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_RKNPU(options: *mut ort_sys::OrtSessionOptions) -> ort_sys::OrtStatusPtr);
		unsafe { Error::result_from_status(OrtSessionOptionsAppendExecutionProvider_RKNPU(session_builder.ptr_mut())) }
	}
}
