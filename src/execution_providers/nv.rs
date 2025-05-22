use alloc::string::ToString;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct NVExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; NVExecutionProvider);

impl NVExecutionProvider {
	pub fn with_device_id(mut self, device_id: u32) -> Self {
		self.options.set("ep.nvtensorrtrtxexecutionprovider.device_id", device_id.to_string());
		self
	}

	pub fn with_cuda_graph(mut self, enable: bool) -> Self {
		self.options
			.set("ep.nvtensorrtrtxexecutionprovider.nv_cuda_graph_enable", if enable { "1" } else { "0" });
		self
	}
}

impl ExecutionProvider for NVExecutionProvider {
	fn name(&self) -> &'static str {
		"NvTensorRTRTXExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_os = "windows", target_arch = "x86_64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "nv"))]
		{
			use crate::{AsPointer, ortsys};

			let _ = crate::environment::get_environment();

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"NvTensorRtRtx".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
