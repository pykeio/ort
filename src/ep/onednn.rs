use super::{ExecutionProvider, ExecutionProviderOptions};
use crate::{AsPointer, error::Result, ortsys, session::builder::SessionBuilder};

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
	const NAME: &'static str = "DnnlExecutionProvider";

	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()> {
		super::register_with_options_struct!(
			self.0,
			session_builder,
			ort_sys::OrtDnnlProviderOptions,
			CreateDnnlProviderOptions,
			UpdateDnnlProviderOptions,
			ReleaseDnnlProviderOptions,
			SessionOptionsAppendExecutionProvider_Dnnl
		)
	}
}
