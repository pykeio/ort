use std::num::NonZeroUsize;

use super::ExecutionProvider;
use crate::{Error, ExecutionProviderDispatch, Result, SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct XNNPACKExecutionProvider {
	intra_op_num_threads: Option<NonZeroUsize>
}

impl XNNPACKExecutionProvider {
	pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
		self.intra_op_num_threads = Some(num_threads);
		self
	}

	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl From<XNNPACKExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: XNNPACKExecutionProvider) -> Self {
		ExecutionProviderDispatch::XNNPACK(value)
	}
}

impl ExecutionProvider for XNNPACKExecutionProvider {
	fn as_str(&self) -> &'static str {
		"XNNPACKExecutionProvider"
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "xnnpack"))]
		{
			let (key_ptrs, value_ptrs, len, _keys, _values) = super::map_keys! {
				intra_op_num_threads = self.intra_op_num_threads.as_ref()
			};
			let ep_name = std::ffi::CString::new("XNNPACK").unwrap();
			return crate::error::status_to_result(crate::ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.session_options_ptr,
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
