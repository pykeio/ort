use crate::{
	error::{Error, Result},
	execution_providers::{ExecutionProvider, ExecutionProviderDispatch},
	session::SessionBuilder
};

#[derive(Debug, Default, Clone)]
pub struct OneDNNExecutionProvider {
	use_arena: Option<bool>
}

impl OneDNNExecutionProvider {
	#[must_use]
	pub fn with_use_arena(mut self, enable: bool) -> Self {
		self.use_arena = Some(enable);
		self
	}

	#[must_use]
	pub fn build(self) -> ExecutionProviderDispatch {
		self.into()
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
	fn register(&self, session_builder: &SessionBuilder) -> Result<()> {
		#[cfg(any(feature = "load-dynamic", feature = "onednn"))]
		{
			let mut dnnl_options: *mut ort_sys::OrtDnnlProviderOptions = std::ptr::null_mut();
			crate::ortsys![unsafe CreateDnnlProviderOptions(&mut dnnl_options)?];
			let (key_ptrs, value_ptrs, len, keys, values) = super::map_keys! {
				use_arena = self.use_arena.map(<bool as Into<i32>>::into)
			};
			if let Err(e) =
				crate::error::status_to_result(crate::ortsys![unsafe UpdateDnnlProviderOptions(dnnl_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), len as _)])
			{
				crate::ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
				std::mem::drop((keys, values));
				return Err(e);
			}

			let status = crate::ortsys![unsafe SessionOptionsAppendExecutionProvider_Dnnl(session_builder.session_options_ptr.as_ptr(), dnnl_options)];
			crate::ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
			std::mem::drop((keys, values));
			return Ok(());
		}

		Err(Error::new(format!("`{}` was not registered because its corresponding Cargo feature is not enabled.", self.as_str())))
	}
}
