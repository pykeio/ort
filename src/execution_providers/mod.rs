//! [`ExecutionProvider`]s provide hardware acceleration to [`Session`](crate::session::Session)s.
//!
//! Sessions can be configured with execution providers via [`SessionBuilder::with_execution_providers`]:
//!
//! ```no_run
//! use ort::{execution_providers::CUDAExecutionProvider, session::Session};
//!
//! fn main() -> ort::Result<()> {
//! 	let session = Session::builder()?
//! 		.with_execution_providers([CUDAExecutionProvider::default().build()])?
//! 		.commit_from_file("model.onnx")?;
//!
//! 	Ok(())
//! }
//! ```

use alloc::{ffi::CString, string::ToString, sync::Arc, vec::Vec};
use core::{
	ffi::c_char,
	fmt::{self, Debug},
	ptr
};

use crate::{char_p_to_string, error::Result, ortsys, session::builder::SessionBuilder, util::MiniMap};

pub mod cpu;
pub use self::cpu::CPUExecutionProvider;
pub mod cuda;
pub use self::cuda::CUDAExecutionProvider;
pub mod tensorrt;
pub use self::tensorrt::TensorRTExecutionProvider;
pub mod onednn;
pub use self::onednn::OneDNNExecutionProvider;
pub mod acl;
pub use self::acl::ACLExecutionProvider;
pub mod openvino;
pub use self::openvino::OpenVINOExecutionProvider;
pub mod coreml;
pub use self::coreml::CoreMLExecutionProvider;
pub mod rocm;
pub use self::rocm::ROCmExecutionProvider;
pub mod cann;
pub use self::cann::CANNExecutionProvider;
pub mod directml;
pub use self::directml::DirectMLExecutionProvider;
pub mod tvm;
pub use self::tvm::TVMExecutionProvider;
pub mod nnapi;
pub use self::nnapi::NNAPIExecutionProvider;
pub mod qnn;
pub use self::qnn::QNNExecutionProvider;
pub mod xnnpack;
pub use self::xnnpack::XNNPACKExecutionProvider;
pub mod armnn;
pub use self::armnn::ArmNNExecutionProvider;
pub mod migraphx;
pub use self::migraphx::MIGraphXExecutionProvider;
pub mod vitis;
pub use self::vitis::VitisAIExecutionProvider;
pub mod rknpu;
pub use self::rknpu::RKNPUExecutionProvider;
pub mod webgpu;
pub use self::webgpu::WebGPUExecutionProvider;
pub mod azure;
pub use self::azure::AzureExecutionProvider;
pub mod nv;
pub use self::nv::NVExecutionProvider;

/// ONNX Runtime works with different hardware acceleration libraries through its extensible **Execution Providers**
/// (EP) framework to optimally execute the ONNX models on the hardware platform. This interface enables flexibility for
/// the AP application developer to deploy their ONNX models in different environments in the cloud and the edge and
/// optimize the execution by taking advantage of the compute capabilities of the platform.
///
/// ![](https://www.onnxruntime.ai/images/ONNX_Runtime_EP1.png)
pub trait ExecutionProvider: Send + Sync {
	/// Returns the identifier of this execution provider used internally by ONNX Runtime.
	///
	/// This is the same as what's used in ONNX Runtime's Python API to register this execution provider, i.e.
	/// [`TVMExecutionProvider`]'s identifier is `TvmExecutionProvider`.
	fn name(&self) -> &'static str;

	/// Returns whether this execution provider is supported on this platform.
	///
	/// For example, the CoreML execution provider implements this as:
	/// ```ignore
	/// impl ExecutionProvider for CoreMLExecutionProvider {
	/// 	fn supported_by_platform() -> bool {
	/// 		cfg!(target_vendor = "apple")
	/// 	}
	/// }
	/// ```
	fn supported_by_platform(&self) -> bool {
		true
	}

	/// Returns `Ok(true)` if ONNX Runtime was *compiled with support* for this execution provider, and `Ok(false)`
	/// otherwise.
	///
	/// An `Err` may be returned if a serious internal error occurs, in which case your application should probably
	/// just abort.
	///
	/// **Note that this does not always mean the execution provider is *usable* for a specific session.** A model may
	/// use operators not supported by an execution provider, or the EP may encounter an error while attempting to load
	/// dependencies during session creation. In most cases (i.e. showing the user an error message if CUDA could not be
	/// enabled), you'll instead want to manually register this EP via [`ExecutionProvider::register`] and detect
	/// and handle any errors returned by that function.
	fn is_available(&self) -> Result<bool> {
		let mut providers: *mut *mut c_char = ptr::null_mut();
		let mut num_providers = 0;
		ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers)?];
		if providers.is_null() {
			return Ok(false);
		}

		for i in 0..num_providers {
			let avail = match char_p_to_string(unsafe { *providers.offset(i as isize) }) {
				Ok(avail) => avail,
				Err(e) => {
					let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
					return Err(e);
				}
			};
			if self.name() == avail {
				let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
				return Ok(true);
			}
		}

		let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
		Ok(false)
	}

	/// Attempts to register this execution provider on the given session.
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError>;
}

/// Trait used for execution providers that can have arbitrary configuration keys applied.
///
/// Most execution providers have a small set of configuration options which don't change between ONNX Runtime releases;
/// others, like the CUDA execution provider, often have options added that go undocumented and thus unimplemented by
/// `ort`. This allows you to configure these options regardless.
pub trait ArbitrarilyConfigurableExecutionProvider {
	fn with_arbitrary_config(self, key: impl ToString, value: impl ToString) -> Self;
}

/// The strategy for extending the device memory arena.
#[derive(Debug, Default, Clone)]
pub enum ArenaExtendStrategy {
	/// (Default) Subsequent extensions extend by larger amounts (multiplied by powers of two)
	#[default]
	NextPowerOfTwo,
	/// Memory extends by the requested amount.
	SameAsRequested
}

/// Dynamic execution provider container, used to provide a list of multiple types of execution providers when
/// configuring execution providers for a [`SessionBuilder`] or
/// [`EnvironmentBuilder`](crate::environment::EnvironmentBuilder).
///
/// See [`ExecutionProvider`] for more info on execution providers.
#[derive(Clone)]
#[allow(missing_docs)]
#[non_exhaustive]
pub struct ExecutionProviderDispatch {
	pub(crate) inner: Arc<dyn ExecutionProvider>,
	error_on_failure: bool
}

impl ExecutionProviderDispatch {
	pub(crate) fn new<E: ExecutionProvider + 'static>(ep: E) -> Self {
		ExecutionProviderDispatch {
			inner: Arc::new(ep) as _,
			error_on_failure: false
		}
	}

	/// Configures this execution provider to silently log an error if registration of the EP fails.
	/// This is the default behavior; it can be overridden with [`ExecutionProviderDispatch::error_on_failure`].
	pub fn fail_silently(mut self) -> Self {
		self.error_on_failure = false;
		self
	}

	/// Configures this execution provider to return an error upon EP registration if registration of this EP fails.
	/// The default behavior is to silently fail and fall back to the next execution provider, or the CPU provider if no
	/// registrations succeed.
	pub fn error_on_failure(mut self) -> Self {
		self.error_on_failure = true;
		self
	}
}

impl Debug for ExecutionProviderDispatch {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct(self.inner.name())
			.field("error_on_failure", &self.error_on_failure)
			.finish()
	}
}

/// Sets the current GPU device of the active EP to the device specified by `device_id`.
///
/// This only works for [`CUDAExecutionProvider`] & [`ROCmExecutionProvider`].
pub fn set_gpu_device(device_id: i32) -> Result<()> {
	ortsys![unsafe SetCurrentGpuDeviceId(device_id)?];
	Ok(())
}

/// Returns the ID of the GPU device being used by the active EP.
///
/// This only works for [`CUDAExecutionProvider`] & [`ROCmExecutionProvider`].
pub fn get_gpu_device() -> Result<i32> {
	let mut out = 0;
	ortsys![unsafe GetCurrentGpuDeviceId(&mut out)?];
	Ok(out)
}

#[derive(Default, Debug, Clone)]
pub(crate) struct ExecutionProviderOptions(MiniMap<CString, CString>);

impl ExecutionProviderOptions {
	pub fn set(&mut self, key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) {
		self.0
			.insert(CString::new(key).expect("unexpected nul in key string"), CString::new(value).expect("unexpected nul in value string"));
	}

	#[allow(unused)]
	pub fn to_ffi(&self) -> ExecutionProviderOptionsFFI {
		let (key_ptrs, value_ptrs) = self.0.iter().map(|(k, v)| (k.as_ptr(), v.as_ptr())).unzip();
		ExecutionProviderOptionsFFI { key_ptrs, value_ptrs }
	}
}

#[allow(unused)]
pub(crate) struct ExecutionProviderOptionsFFI {
	key_ptrs: Vec<*const c_char>,
	value_ptrs: Vec<*const c_char>
}

#[allow(unused)]
impl ExecutionProviderOptionsFFI {
	pub fn key_ptrs(&self) -> *const *const c_char {
		self.key_ptrs.as_ptr()
	}

	pub fn value_ptrs(&self) -> *const *const c_char {
		self.value_ptrs.as_ptr()
	}

	pub fn len(&self) -> usize {
		self.key_ptrs.len()
	}
}

#[derive(Debug)]
pub enum RegisterError {
	Error(crate::Error),
	MissingFeature
}

impl From<crate::Error> for RegisterError {
	fn from(value: crate::Error) -> Self {
		Self::Error(value)
	}
}

impl From<RegisterError> for crate::Error {
	fn from(value: RegisterError) -> Self {
		match value {
			RegisterError::Error(e) => e,
			RegisterError::MissingFeature => {
				crate::Error::new("The execution provider could not be registered because its corresponding Cargo feature is not enabled.")
			}
		}
	}
}

impl fmt::Display for RegisterError {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Error(e) => fmt::Display::fmt(e, f),
			Self::MissingFeature => f.write_str("The execution provider could not be registered because its corresponding Cargo feature is not enabled.")
		}
	}
}

#[cfg(feature = "std")]
impl std::error::Error for RegisterError {}

#[allow(unused)]
macro_rules! define_ep_register {
	($symbol:ident($($id:ident: $type:ty),*) -> $rt:ty) => {
		#[cfg(feature = "load-dynamic")]
		#[allow(non_snake_case)]
		let $symbol = unsafe {
			let dylib = $crate::lib_handle();
			let symbol: ::core::result::Result<
				::libloading::Symbol<unsafe extern "C" fn($($id: $type),*) -> $rt>,
				::libloading::Error
			> = dylib.get(stringify!($symbol).as_bytes());
			match symbol {
				Ok(symbol) => symbol.into_raw(),
				Err(e) => {
					return ::core::result::Result::Err($crate::Error::new(::alloc::format!("Error attempting to load symbol `{}` from dynamic library: {}", stringify!($symbol), e)))?;
				}
			}
		};
		#[cfg(not(feature = "load-dynamic"))]
		unsafe extern "C" {
			fn $symbol($($id: $type),*) -> $rt;
		}
	};
}
#[allow(unused)]
pub(crate) use define_ep_register;

macro_rules! impl_ep {
	(arbitrary; $symbol:ident) => {
		$crate::execution_providers::impl_ep!($symbol);

		impl $crate::execution_providers::ArbitrarilyConfigurableExecutionProvider for $symbol {
			fn with_arbitrary_config(mut self, key: impl ::alloc::string::ToString, value: impl ::alloc::string::ToString) -> Self {
				self.options.set(key.to_string(), value.to_string());
				self
			}
		}
	};
	($symbol:ident) => {
		impl $symbol {
			#[must_use]
			pub fn build(self) -> $crate::execution_providers::ExecutionProviderDispatch {
				self.into()
			}
		}

		impl From<$symbol> for $crate::execution_providers::ExecutionProviderDispatch {
			fn from(value: $symbol) -> Self {
				$crate::execution_providers::ExecutionProviderDispatch::new(value)
			}
		}
	};
}
pub(crate) use impl_ep;

pub(crate) fn apply_execution_providers(session_builder: &mut SessionBuilder, eps: &[ExecutionProviderDispatch], source: &'static str) -> Result<()> {
	fn register_inner(session_builder: &mut SessionBuilder, ep: &ExecutionProviderDispatch, #[allow(unused)] source: &'static str) -> Result<bool> {
		if let Err(e) = ep.inner.register(session_builder) {
			if ep.error_on_failure {
				return Err(e)?;
			}

			if matches!(e, RegisterError::MissingFeature) {
				if ep.inner.supported_by_platform() {
					crate::warn!(%source, "{e}");
				} else {
					crate::debug!(%source, "{e} (note: additionally, `{}` may not be supported on this platform)", ep.inner.name());
				}
			} else {
				crate::error!(%source, "An error occurred when attempting to register `{}`: {e}", ep.inner.name());
			}
			Ok(false)
		} else {
			crate::info!(%source, "Successfully registered `{}`", ep.inner.name());
			Ok(true)
		}
	}

	let mut fallback_to_cpu = !eps.is_empty();
	for ep in eps {
		if register_inner(session_builder, ep, source)? {
			fallback_to_cpu = false;
		}
	}
	if fallback_to_cpu {
		crate::warn!("No execution providers from {source} registered successfully; may fall back to CPU.");
	}
	Ok(())
}
