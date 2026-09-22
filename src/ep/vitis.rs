use alloc::string::ToString;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Default, Clone)]
pub struct Vitis(ExecutionProviderOptions);

super::impl_ep!(arbitrary; Vitis);

impl Vitis {
	super::define_options! {
		pub fn with_config_file(mut self, config_file: impl ToString) -> Self = "config_file";

		pub fn with_cache_dir(mut self, cache_dir: impl ToString) -> Self = "cache_dir";

		pub fn with_cache_key(mut self, cache_key: impl ToString) -> Self = "cache_key";
	}
}

impl SimpleExecutionProvider for Vitis {
	const CANONICAL_NAME: &'static str = "VitisAIExecutionProvider";
	const SHORT_NAME: &'static str = "VitisAI";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
