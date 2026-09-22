use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Default, Clone)]
pub struct WASM(ExecutionProviderOptions);

super::impl_ep!(arbitrary; WASM);

impl SimpleExecutionProvider for WASM {
	const CANONICAL_NAME: &'static str = "WASMExecutionProvider";
	const SHORT_NAME: &'static str = "WASM";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
