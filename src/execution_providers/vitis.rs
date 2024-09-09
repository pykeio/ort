use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct VitisAIExecutionProvider {
	config_file: Option<String>,
	cache_dir: Option<String>,
	cache_key: Option<String>
}

impl VitisAIExecutionProvider {
	pub fn with_config_file(mut self, config_file: impl ToString) -> Self {
		self.config_file = Some(config_file.to_string());
		self
	}

	pub fn with_cache_dir(mut self, cache_dir: impl ToString) -> Self {
		self.cache_dir = Some(cache_dir.to_string());
		self
	}

	pub fn with_cache_key(mut self, cache_key: impl ToString) -> Self {
		self.cache_key = Some(cache_key.to_string());
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "vitis"))]
		{
			let (key_ptrs, value_ptrs, len, keys, values) = super::map_keys! {
				config_file = self.config_file.clone(),
				cacheDir = self.cache_dir.clone(),
				cacheKey = self.cache_key.clone()
			};

			let status = crate::ortsys![
				unsafe SessionOptionsAppendExecutionProvider_VitisAI(
					session_builder.session_options_ptr.as_ptr(),
					key_ptrs.as_ptr(),
					value_ptrs.as_ptr(),
					len as _
				)
			];
			std::mem::drop((keys, values));
			return crate::error::status_to_result(status);
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
