use std::num::NonZeroUsize;

use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct XNNPACKExecutionProvider {
	intra_op_num_threads: Option<NonZeroUsize>
}

impl XNNPACKExecutionProvider {
	#[must_use]
	pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
		self.intra_op_num_threads = Some(num_threads);
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<XNNPACKExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: XNNPACKExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for XNNPACKExecutionProvider {
	fn as_str(&self) -> &'static str {
		"XNNPACKExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_arch = "aarch64", all(target_arch = "arm", any(target_os = "linux", target_os = "android")), target_arch = "x86_64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "xnnpack"))]
		{
			let (key_ptrs, value_ptrs, len, _keys, _values) = super::map_keys! {
				intra_op_num_threads = self.intra_op_num_threads.as_ref()
			};
			let ep_name = std::ffi::CString::new("XNNPACK").unwrap_or_else(|_| unreachable!());
			return crate::error::status_to_result(crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.session_options_ptr.as_ptr(),
				ep_name.as_ptr(),
				key_ptrs.as_ptr(),
				value_ptrs.as_ptr(),
				len as _,
			)])
			.map_err(Error::ExecutionProvider);
		}

		Err(Error::ExecutionProviderNotRegistered(self.as_str()))
	}
}
