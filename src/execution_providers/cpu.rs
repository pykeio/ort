use crate::{
	AsPointer,
	error::Result,
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	ortsys,
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct CPUExecutionProvider {
	use_arena: bool
}

impl CPUExecutionProvider {
	#[must_use]
	pub fn with_arena_allocator(mut self, enable: bool) -> Self {
		self.use_arena = enable;
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<CPUExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: CPUExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for CPUExecutionProvider {
	fn as_str(&self) -> &'static str {
		"CPUExecutionProvider"
	}

	/// The CPU execution provider is always available.
	fn is_available(&self) -> Result<bool> {
		Ok(true)
	}

	fn supported_by_platform(&self) -> bool {
		true
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		if self.use_arena {
			ortsys![unsafe EnableCpuMemArena(session_builder.ptr_mut())?];
		} else {
			ortsys![unsafe DisableCpuMemArena(session_builder.ptr_mut())?];
		}
		Ok(())
	}
}
