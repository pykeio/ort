use alloc::{format, string::ToString};
use core::num::NonZeroUsize;

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct XNNPACKExecutionProvider {
	options: ExecutionProviderOptions
}

impl XNNPACKExecutionProvider {
	#[must_use]
	pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
		self.options.set("intra_op_num_threads", num_threads.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for XNNPACKExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<XNNPACKExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: XNNPACKExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "xnnpack"))]
		{
			use crate::AsPointer;

			let ffi_options = self.options.to_ffi();
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"XNNPACK".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
