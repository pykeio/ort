use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

#[derive(Debug, Default, Clone)]
pub struct NNAPIExecutionProvider {
	use_fp16: bool,
	use_nchw: bool,
	disable_cpu: bool,
	cpu_only: bool
}

super::impl_ep!(NNAPIExecutionProvider);

impl NNAPIExecutionProvider {
	/// Use fp16 relaxation in NNAPI EP. This may improve performance but can also reduce accuracy due to the lower
	/// precision.
	#[must_use]
	pub fn with_fp16(mut self, enable: bool) -> Self {
		self.use_fp16 = enable;
		self
	}

	/// Use the NCHW layout in NNAPI EP. This is only available for Android API level 29 and higher. Please note that
	/// for now, NNAPI might have worse performance using NCHW compared to using NHWC.
	#[must_use]
	pub fn with_nchw(mut self, enable: bool) -> Self {
		self.use_nchw = enable;
		self
	}

	/// Prevents NNAPI from using CPU devices. NNAPI is more efficient using GPU or NPU for execution, and NNAPI
	/// might fall back to its own CPU implementation for operations not supported by the GPU/NPU. However, the
	/// CPU implementation of NNAPI might be less efficient than the optimized versions of operators provided by
	/// ORT's default MLAS execution provider. It might be better to disable the NNAPI CPU fallback and instead
	/// use MLAS kernels. This option is only available after Android API level 29.
	#[must_use]
	pub fn with_disable_cpu(mut self, enable: bool) -> Self {
		self.disable_cpu = enable;
		self
	}

	/// Using CPU only in NNAPI EP, this may decrease the perf but will provide reference output value without precision
	/// loss, which is useful for validation. This option is only available for Android API level 29 and higher, and
	/// will be ignored for Android API level 28 and lower.
	#[must_use]
	pub fn with_cpu_only(mut self, enable: bool) -> Self {
		self.cpu_only = enable;
		self
	}
}

impl ExecutionProvider for NNAPIExecutionProvider {
	fn name(&self) -> &'static str {
		"NnapiExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_os = "android")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "nnapi"))]
		{
			use crate::AsPointer;

			super::define_ep_register!(OrtSessionOptionsAppendExecutionProvider_Nnapi(options: *mut ort_sys::OrtSessionOptions, flags: u32) -> ort_sys::OrtStatusPtr);
			let mut flags = 0;
			if self.use_fp16 {
				flags |= 0x001;
			}
			if self.use_nchw {
				flags |= 0x002;
			}
			if self.disable_cpu {
				flags |= 0x004;
			}
			if self.cpu_only {
				flags |= 0x008;
			}
			return Ok(unsafe { crate::error::status_to_result(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_builder.ptr_mut(), flags)) }?);
		}

		Err(RegisterError::MissingFeature)
	}
}
