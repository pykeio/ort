use core::num::NonZeroUsize;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

/// [XNNPACK execution provider](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html) for
/// ARM, x86, and WASM platforms.
///
/// # Threading
/// XNNPACK uses its own threadpool separate from the [`Session`](crate::session::Session)'s intra-op threadpool. If
/// most of your model's compute lies in nodes supported by XNNPACK (i.e. `Conv`, `Gemm`, `MatMul`), it's best to
/// disable the session intra-op threadpool to reduce contention:
/// ```no_run
/// # use core::num::NonZeroUsize;
/// # use ort::{ep, session::Session};
/// # fn main() -> ort::Result<()> {
/// # let env = ort::test_util::test_env().clone();
/// let session = Session::builder(&env)?
/// 	.with_intra_op_spinning(false)?
/// 	.with_intra_threads(1)?
/// 	.with_execution_providers([ep::XNNPACK::default()
/// 		.with_intra_op_num_threads(NonZeroUsize::new(4).unwrap())
/// 		.build()])?
/// 	.commit_from_file("model.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct XNNPACK(ExecutionProviderOptions);

super::impl_ep!(arbitrary; XNNPACK);

impl XNNPACK {
	super::define_options! {
		/// Configures the number of threads to use for XNNPACK's internal intra-op threadpool.
		///
		/// ```
		/// # use core::num::NonZeroUsize;
		/// # use ort::{ep, session::Session};
		/// # fn main() -> ort::Result<()> {
		/// let ep = ep::XNNPACK::default().with_intra_op_num_threads(NonZeroUsize::new(4).unwrap()).build();
		/// # Ok(())
		/// # }
		/// ```
		pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self = "intra_op_num_threads";
	}
}

impl SimpleExecutionProvider for XNNPACK {
	const SHORT_NAME: &'static str = "XNNPACK";
	const CANONICAL_NAME: &'static str = "XnnpackExecutionProvider";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
