use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct OneDNNExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; OneDNNExecutionProvider);

impl OneDNNExecutionProvider {
	#[must_use]
	pub fn with_use_arena(mut self, enable: bool) -> Self {
		self.options.set("use_arena", if enable { "1" } else { "0" });
		self
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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "onednn"))]
		{
			use core::ptr;

			use crate::{AsPointer, ortsys, util};

			let mut dnnl_options: *mut ort_sys::OrtDnnlProviderOptions = ptr::null_mut();
			ortsys![unsafe CreateDnnlProviderOptions(&mut dnnl_options)?];
			let _guard = util::run_on_drop(|| {
				ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
			});

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe UpdateDnnlProviderOptions(
				dnnl_options,
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len()
			)?];

			ortsys![unsafe SessionOptionsAppendExecutionProvider_Dnnl(session_builder.ptr_mut(), dnnl_options)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
