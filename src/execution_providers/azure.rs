use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct AzureExecutionProvider {
	options: ExecutionProviderOptions
}

impl AzureExecutionProvider {
	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for AzureExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<AzureExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: AzureExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for AzureExecutionProvider {
	fn as_str(&self) -> &'static str {
		"AzureExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_os = "linux", target_os = "windows", target_os = "android"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "azure"))]
		{
			use crate::AsPointer;

			let ffi_options = self.options.to_ffi();
			crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"AZURE".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
