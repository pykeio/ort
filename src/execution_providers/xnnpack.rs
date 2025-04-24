use alloc::string::ToString;
use core::num::NonZeroUsize;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct XNNPACKExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; XNNPACKExecutionProvider);

impl XNNPACKExecutionProvider {
	#[must_use]
	pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
		self.options.set("intra_op_num_threads", num_threads.to_string());
		self
	}
}

impl ExecutionProvider for XNNPACKExecutionProvider {
	fn as_str(&self) -> &'static str {
		"XnnpackExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_arch = "aarch64", all(target_arch = "arm", any(target_os = "linux", target_os = "android")), target_arch = "x86_64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "xnnpack"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"XNNPACK".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
