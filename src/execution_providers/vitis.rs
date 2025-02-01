use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct VitisAIExecutionProvider {
	options: ExecutionProviderOptions
}

impl VitisAIExecutionProvider {
	pub fn with_config_file(mut self, config_file: impl ToString) -> Self {
		self.options.set("config_file", config_file.to_string());
		self
	}

	pub fn with_cache_dir(mut self, cache_dir: impl ToString) -> Self {
		self.options.set("cache_dir", cache_dir.to_string());
		self
	}

	pub fn with_cache_key(mut self, cache_key: impl ToString) -> Self {
		self.options.set("cache_key", cache_key.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for VitisAIExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<VitisAIExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: VitisAIExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for VitisAIExecutionProvider {
	fn as_str(&self) -> &'static str {
		"VitisAIExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(all(target_os = "linux", target_arch = "x86_64"), all(target_os = "windows", target_arch = "x86_64")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "vitis"))]
		{
			use crate::AsPointer;

			let ffi_options = self.options.to_ffi();
			crate::ortsys![
				unsafe SessionOptionsAppendExecutionProvider_VitisAI(
					session_builder.ptr_mut(),
					ffi_options.key_ptrs(),
					ffi_options.value_ptrs(),
					ffi_options.len()
				)?
			];
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
