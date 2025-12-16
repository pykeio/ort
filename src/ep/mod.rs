//! [`ExecutionProvider`]s provide hardware acceleration to [`Session`](crate::session::Session)s.
//!
//! Sessions can be configured with execution providers via [`SessionBuilder::with_execution_providers`]:
//!
//! ```no_run
//! use ort::{ep, session::Session};
//!
//! fn main() -> ort::Result<()> {
//! 	let session = Session::builder()?
//! 		.with_execution_providers([ep::CUDA::default().build()])?
//! 		.commit_from_file("model.onnx")?;
//!
//! 	Ok(())
//! }
//! ```

use alloc::{ffi::CString, string::ToString, sync::Arc, vec::Vec};
use core::{
	any::Any,
	ffi::c_char,
	fmt::{self, Debug},
	ptr
};

use crate::{
	error::Result,
	ortsys,
	session::builder::SessionBuilder,
	util::{MiniMap, char_p_to_string, run_on_drop}
};

pub mod cpu;
pub use self::cpu::CPU;
pub mod cuda;
pub use self::cuda::CUDA;
pub mod tensorrt;
pub use self::tensorrt::TensorRT;
pub mod onednn;
pub use self::onednn::OneDNN;
pub mod acl;
pub use self::acl::ACL;
pub mod openvino;
pub use self::openvino::OpenVINO;
pub mod coreml;
pub use self::coreml::CoreML;
pub mod rocm;
pub use self::rocm::ROCm;
pub mod cann;
pub use self::cann::CANN;
pub mod directml;
pub use self::directml::DirectML;
pub mod tvm;
pub use self::tvm::TVM;
pub mod nnapi;
pub use self::nnapi::NNAPI;
pub mod qnn;
pub use self::qnn::QNN;
pub mod xnnpack;
pub use self::xnnpack::XNNPACK;
pub mod armnn;
pub use self::armnn::ArmNN;
pub mod migraphx;
pub use self::migraphx::MIGraphX;
pub mod vitis;
pub use self::vitis::Vitis;
pub mod rknpu;
pub use self::rknpu::RKNPU;
pub mod webgpu;
pub use self::webgpu::WebGPU;
pub mod azure;
pub use self::azure::Azure;
pub mod nvrtx;
pub use self::nvrtx::NVRTX;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub mod webnn;
#[cfg(target_arch = "wasm32")]
pub use self::{wasm::WASM, webnn::WebNN};

pub trait ExecutionProvider: Any + Send + Sync {
	/// Returns the identifier of this execution provider used internally by ONNX Runtime.
	///
	/// This is the same as what's used in ONNX Runtime's Python API to register this execution provider, i.e.
	/// [`TVM`]'s identifier is `TvmExecutionProvider`.
	fn name(&self) -> &'static str;

	/// Returns whether this execution provider is supported on this platform.
	///
	/// For example, the CoreML execution provider implements this as:
	/// ```ignore
	/// impl ExecutionProvider for CoreML {
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
		is_ep_available(self.name())
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

	/// Attempt to downcast this execution provider to a concrete type `E`.
	pub fn downcast_ref<E: ExecutionProvider>(&self) -> Option<&E> {
		<dyn Any>::downcast_ref(&*self.inner)
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

impl core::error::Error for RegisterError {}

#[allow(unused)]
macro_rules! define_ep_register {
	($symbol:ident($($id:ident: $type:ty),*) -> $rt:ty) => {
		#[cfg(feature = "load-dynamic")]
		#[allow(non_snake_case)]
		let $symbol = unsafe {
			let dylib = $crate::G_ORT_LIB.get().expect("dylib not yet initialized");
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
		$crate::ep::impl_ep!($symbol);

		impl $crate::ep::ArbitrarilyConfigurableExecutionProvider for $symbol {
			fn with_arbitrary_config(mut self, key: impl ::alloc::string::ToString, value: impl ::alloc::string::ToString) -> Self {
				self.options.set(key.to_string(), value.to_string());
				self
			}
		}
	};
	($symbol:ident) => {
		impl $symbol {
			#[must_use]
			pub fn build(self) -> $crate::ep::ExecutionProviderDispatch {
				self.into()
			}
		}

		impl From<$symbol> for $crate::ep::ExecutionProviderDispatch {
			fn from(value: $symbol) -> Self {
				$crate::ep::ExecutionProviderDispatch::new(value)
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

fn is_ep_available(name: &str) -> Result<bool> {
	let mut providers: *mut *mut c_char = ptr::null_mut();
	let mut num_providers = 0;
	ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers)?];
	if providers.is_null() {
		return Ok(false);
	}

	let _guard = run_on_drop(|| ortsys![unsafe ReleaseAvailableProviders(providers, num_providers).expect("infallible")]);

	for i in 0..num_providers {
		let avail = match char_p_to_string(unsafe { *providers.offset(i as isize) }) {
			Ok(avail) => avail,
			Err(e) => {
				return Err(e);
			}
		};
		if name == avail {
			return Ok(true);
		}
	}

	Ok(false)
}

#[deprecated = "import `ort::ep::ACL` instead"]
#[doc(hidden)]
pub use self::acl::ACL as ACLExecutionProvider;
#[deprecated = "import `ort::ep::ArmNN` instead"]
#[doc(hidden)]
pub use self::armnn::ArmNN as ArmNNExecutionProvider;
#[deprecated = "import `ort::ep::Azure` instead"]
#[doc(hidden)]
pub use self::azure::Azure as AzureExecutionProvider;
#[deprecated = "import `ort::ep::CANN` instead"]
#[doc(hidden)]
pub use self::cann::CANN as CANNExecutionProvider;
#[deprecated = "import `ort::ep::CoreML` instead"]
#[doc(hidden)]
pub use self::coreml::CoreML as CoreMLExecutionProvider;
#[deprecated = "import `ort::ep::CPU` instead"]
#[doc(hidden)]
pub use self::cpu::CPU as CPUExecutionProvider;
#[deprecated = "import `ort::ep::CUDA` instead"]
#[doc(hidden)]
pub use self::cuda::CUDA as CUDAExecutionProvider;
#[deprecated = "import `ort::ep::DirectML` instead"]
#[doc(hidden)]
pub use self::directml::DirectML as DirectMLExecutionProvider;
#[deprecated = "import `ort::ep::MIGraphX` instead"]
#[doc(hidden)]
pub use self::migraphx::MIGraphX as MIGraphXExecutionProvider;
#[deprecated = "import `ort::ep::NNAPI` instead"]
#[doc(hidden)]
pub use self::nnapi::NNAPI as NNAPIExecutionProvider;
#[deprecated = "import `ort::ep::NVRTX` instead"]
#[doc(hidden)]
pub use self::nvrtx::NVRTX as NVRTXExecutionProvider;
#[deprecated = "import `ort::ep::OneDNN` instead"]
#[doc(hidden)]
pub use self::onednn::OneDNN as OneDNNExecutionProvider;
#[deprecated = "import `ort::ep::OpenVINO` instead"]
#[doc(hidden)]
pub use self::openvino::OpenVINO as OpenVINOExecutionProvider;
#[deprecated = "import `ort::ep::QNN` instead"]
#[doc(hidden)]
pub use self::qnn::QNN as QNNExecutionProvider;
#[deprecated = "import `ort::ep::RKNPU` instead"]
#[doc(hidden)]
pub use self::rknpu::RKNPU as RKNPUExecutionProvider;
#[deprecated = "import `ort::ep::ROCm` instead"]
#[doc(hidden)]
pub use self::rocm::ROCm as ROCmExecutionProvider;
#[deprecated = "import `ort::ep::TensorRT` instead"]
#[doc(hidden)]
pub use self::tensorrt::TensorRT as TensorRTExecutionProvider;
#[deprecated = "import `ort::ep::TVM` instead"]
#[doc(hidden)]
pub use self::tvm::TVM as TVMExecutionProvider;
#[deprecated = "import `ort::ep::Vitis` instead"]
#[doc(hidden)]
pub use self::vitis::Vitis as VitisAIExecutionProvider;
#[deprecated = "import `ort::ep::WASM` instead"]
#[doc(hidden)]
#[cfg(target_arch = "wasm32")]
pub use self::wasm::WASM as WASMExecutionProvider;
#[deprecated = "import `ort::ep::WebGPU` instead"]
#[doc(hidden)]
pub use self::webgpu::WebGPU as WebGPUExecutionProvider;
#[deprecated = "import `ort::ep::WebNN` instead"]
#[doc(hidden)]
#[cfg(target_arch = "wasm32")]
pub use self::webnn::WebNN as WebNNExecutionProvider;
#[deprecated = "import `ort::ep::XNNPACK` instead"]
#[doc(hidden)]
pub use self::xnnpack::XNNPACK as XNNPACKExecutionProvider;
