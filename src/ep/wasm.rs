use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct WASM {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; WASM);

impl ExecutionProvider for WASM {
	fn name(&self) -> &'static str {
		"WASMExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_arch = "wasm32")
	}

	#[allow(unused)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		let ffi_options = self.options.to_ffi();
		ortsys![unsafe SessionOptionsAppendExecutionProvider(
			session_builder.ptr_mut(),
			c"WASM".as_ptr().cast::<core::ffi::c_char>(),
			ffi_options.key_ptrs(),
			ffi_options.value_ptrs(),
			ffi_options.len(),
		)?];
		Ok(())
	}
}
