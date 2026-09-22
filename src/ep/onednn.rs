use core::ptr;

use super::{ExecutionProvider, ExecutionProviderOptions};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder, util};

/// [oneDNN/DNNL execution provider](https://onnxruntime.ai/docs/execution-providers/oneDNN-ExecutionProvider.html) for
/// Intel CPUs & iGPUs.
#[derive(Debug, Default, Clone)]
#[doc(alias = "DNNL")]
pub struct OneDNN(ExecutionProviderOptions);

super::impl_ep!(arbitrary; OneDNN);

impl OneDNN {
	super::define_options! {
		/// Enable/disable the usage of the arena allocator.
		///
		/// ```
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::OneDNN::default().with_arena_allocator(true).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_arena_allocator(mut self, enable: bool) -> Self = "use_arena";
	}
}

impl ExecutionProvider for OneDNN {
	fn name(&self) -> &'static str {
		"DnnlExecutionProvider"
	}

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		let mut dnnl_options: *mut ort_sys::OrtDnnlProviderOptions = ptr::null_mut();
		ortsys![unsafe CreateDnnlProviderOptions(&mut dnnl_options)?];
		let _guard = util::run_on_drop(|| {
			ortsys![unsafe ReleaseDnnlProviderOptions(dnnl_options)];
		});

		let ffi_options = self.0.to_ffi();
		ortsys![unsafe UpdateDnnlProviderOptions(
			dnnl_options,
			ffi_options.key_ptrs(),
			ffi_options.value_ptrs(),
			ffi_options.len()
		)?];

		ortsys![unsafe SessionOptionsAppendExecutionProvider_Dnnl(session_builder.ptr_mut(), dnnl_options)?];

		Ok(())
	}
}
