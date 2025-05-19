use alloc::string::ToString;
use core::num::NonZeroUsize;

use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [XNNPACK execution provider](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html) for
/// ARM, x86, and WASM platforms.
///
/// # Threading
/// XNNPACK uses its own threadpool separate from the [`Session`](crate::session::Session)'s intra-op threadpool. If
/// most of your model's compute lies in nodes supported by XNNPACK (i.e. `Conv`, `Gemm`, `MatMul`), it's best to
/// disable the session intra-op threadpool to reduce contention:
/// ```no_run
/// # use core::num::NonZeroUsize;
/// # use ort::{execution_providers::xnnpack::XNNPACKExecutionProvider, session::Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_intra_op_spinning(false)?
/// 	.with_intra_threads(1)?
/// 	.with_execution_providers([XNNPACKExecutionProvider::default()
/// 		.with_intra_op_num_threads(NonZeroUsize::new(4).unwrap())
/// 		.build()])?
/// 	.commit_from_file("model.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct XNNPACKExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; XNNPACKExecutionProvider);

impl XNNPACKExecutionProvider {
	/// Configures the number of threads to use for XNNPACK's internal intra-op threadpool.
	///
	/// ```
	/// # use core::num::NonZeroUsize;
	/// # use ort::{execution_providers::xnnpack::XNNPACKExecutionProvider, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = XNNPACKExecutionProvider::default()
	/// 	.with_intra_op_num_threads(NonZeroUsize::new(4).unwrap())
	/// 	.build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_intra_op_num_threads(mut self, num_threads: NonZeroUsize) -> Self {
		self.options.set("intra_op_num_threads", num_threads.to_string());
		self
	}
}

impl ExecutionProvider for XNNPACKExecutionProvider {
	fn name(&self) -> &'static str {
		"XnnpackExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_arch = "aarch64", all(target_arch = "arm", any(target_os = "linux", target_os = "android")), target_arch = "x86_64"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "xnnpack"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"XNNPACK".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
