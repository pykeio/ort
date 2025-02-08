use alloc::{format, string::ToString};

use super::{ArbitrarilyConfigurableExecutionProvider, ExecutionProviderOptions};
use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::builder::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct OneDNNExecutionProvider {
	options: ExecutionProviderOptions
}

impl OneDNNExecutionProvider {
	#[must_use]
	pub fn with_use_arena(mut self, enable: bool) -> Self {
		self.options.set("use_arena", if enable { "1" } else { "0" });
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
	}
}

impl ArbitrarilyConfigurableExecutionProvider for OneDNNExecutionProvider {
	fn with_arbitrary_config(mut self, key: impl ToString, value: impl ToString) -> Self {
		self.options.set(key.to_string(), value.to_string());
		self
	}
}

impl From<OneDNNExecutionProvider> for ExecutionProviderDispatch {
	fn from(value: OneDNNExecutionProvider) -> Self {
		ExecutionProviderDispatch::new(value)
	}
}

impl ExecutionProvider for OneDNNExecutionProvider {
	fn as_str(&self) -> &'static str {
		"DnnlExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(all(target_arch = "x86_64", any(target_os = "windows", target_os = "linux")))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "onednn"))]
		{
			use crate::AsPointer;

			let mut dnnl_options: *mut ort_sys::OrtDnnlProviderOptions = core::ptr::null_mut();
			crate::ortsys![unsafe CreateDnnlProviderOptions(&mut dnnl_options)?];
			let ffi_options = self.options.to_ffi();

			let res = crate::ortsys![unsafe UpdateDnnlProviderOptions(
				dnnl_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)];
			if let Err(e) = unsafe { crate::error::status_to_result(res) } {
				crate::ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_Dnnl(session_builder.ptr_mut(), dnnl_options)];
			crate::ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
			return unsafe { crate::error::status_to_result(status) };
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
