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
#[cfg(feature = "api-22")]
use alloc::{string::String, sync::Weak};
use core::{
	any::Any,
	ffi::c_char,
	fmt::{self, Debug},
	ptr
};

#[cfg(feature = "api-22")]
use crate::environment::Environment;
use crate::{
	error::Result,
	ortsys,
	session::builder::SessionBuilder,
	util::{MiniMap, char_p_to_string, run_on_drop}
};

pub trait ExecutionProvider: Any + Send + Sync {
	/// Returns the identifier of this execution provider used internally by ONNX Runtime.
	///
	/// This is the same as what's used in ONNX Runtime's Python API to register this execution provider, i.e.
	/// [`TVM`]'s identifier is `TvmExecutionProvider`.
	fn name(&self) -> &'static str;

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
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<()>;
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
/// This only works for [`CUDA`] & [`ROCm`].
pub fn set_gpu_device(device_id: i32) -> Result<()> {
	ortsys![unsafe SetCurrentGpuDeviceId(device_id)?];
	Ok(())
}

/// Returns the ID of the GPU device being used by the active EP.
///
/// This only works for [`CUDA`] & [`ROCm`].
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

#[allow(unused)]
macro_rules! define_ep_register {
	($symbol:ident($($id:ident: $type:ty),*) -> $rt:ty) => {
		#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
		#[allow(non_snake_case)]
		let $symbol = unsafe {
			let dylib = $crate::load_dynamic::G_ORT_LIB.get().expect("dylib not yet initialized");
			let symbol: ::core::result::Result<
				::libloading::Symbol<unsafe extern "C" fn($($id: $type),*) -> $rt>,
				::libloading::Error
			> = dylib.get(stringify!($symbol).as_bytes());
			match symbol {
				Ok(symbol) => symbol.into_raw(),
				Err(e) => {
					return ::core::result::Result::Err($crate::Error::new(::alloc::format!("Error attempting to load symbol `{}` from dynamic library: {}", stringify!($symbol), e)));
				}
			}
		};
		#[cfg(not(all(feature = "load-dynamic", not(target_arch = "wasm32"))))]
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

			crate::error!(%source, "An error occurred when attempting to register `{}`: {e}", ep.inner.name());
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

/// Handle to a loaded execution provider library, obtained from [`Environment::register_ep_library`].
#[cfg(feature = "api-22")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
pub struct ExecutionProviderLibrary {
	name: String,
	env: Weak<Environment>
}

#[cfg(feature = "api-22")]
impl ExecutionProviderLibrary {
	pub(crate) fn new(name: impl Into<String>, env: &Arc<Environment>) -> Self {
		Self {
			name: name.into(),
			env: Arc::downgrade(env)
		}
	}

	/// Unregister the EP library from the environment.
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn unregister(self) -> Result<()> {
		if let Some(env) = self.env.upgrade() {
			use crate::AsPointer;
			crate::util::with_cstr(self.name.as_bytes(), &|name| {
				ortsys![unsafe UnregisterExecutionProviderLibrary(env.ptr().cast_mut(), name.as_ptr())?];
				Ok(())
			})?;
		}
		Ok(())
	}
}

pub trait ExecutionProviderResource {
	type Type;

	const VERSION: u32;

	fn id(&self) -> u32;

	#[doc(hidden)]
	fn convert(value: *const core::ffi::c_void) -> Self::Type;
}

pub mod cpu;
pub use self::cpu::CPU;

// I hope you like attributes!
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub mod cuda;
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub use self::cuda::CUDA;
#[cfg(feature = "tensorrt")]
#[cfg_attr(docsrs, doc(cfg(feature = "tensorrt")))]
pub mod tensorrt;
#[cfg(feature = "tensorrt")]
#[cfg_attr(docsrs, doc(cfg(feature = "tensorrt")))]
pub use self::tensorrt::TensorRT;
#[cfg(feature = "onednn")]
#[cfg_attr(docsrs, doc(cfg(feature = "onednn")))]
#[doc(alias = "dnnl")]
pub mod onednn;
#[cfg(feature = "onednn")]
#[cfg_attr(docsrs, doc(cfg(feature = "onednn")))]
#[doc(alias = "DNNL")]
pub use self::onednn::OneDNN;
#[cfg(feature = "acl")]
#[cfg_attr(docsrs, doc(cfg(feature = "acl")))]
pub mod acl;
#[cfg(feature = "acl")]
#[cfg_attr(docsrs, doc(cfg(feature = "acl")))]
pub use self::acl::ACL;
#[cfg(feature = "openvino")]
#[cfg_attr(docsrs, doc(cfg(feature = "openvino")))]
pub mod openvino;
#[cfg(feature = "openvino")]
#[cfg_attr(docsrs, doc(cfg(feature = "openvino")))]
pub use self::openvino::OpenVINO;
#[cfg(feature = "coreml")]
#[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
pub mod coreml;
#[cfg(feature = "coreml")]
#[cfg_attr(docsrs, doc(cfg(feature = "coreml")))]
pub use self::coreml::CoreML;
#[cfg(feature = "rocm")]
#[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
pub mod rocm;
#[cfg(feature = "rocm")]
#[cfg_attr(docsrs, doc(cfg(feature = "rocm")))]
pub use self::rocm::ROCm;
#[cfg(feature = "cann")]
#[cfg_attr(docsrs, doc(cfg(feature = "cann")))]
pub mod cann;
#[cfg(feature = "cann")]
#[cfg_attr(docsrs, doc(cfg(feature = "cann")))]
pub use self::cann::CANN;
#[cfg(feature = "directml")]
#[cfg_attr(docsrs, doc(cfg(feature = "directml")))]
pub mod directml;
#[cfg(feature = "directml")]
#[cfg_attr(docsrs, doc(cfg(feature = "directml")))]
pub use self::directml::DirectML;
#[cfg(feature = "tvm")]
#[cfg_attr(docsrs, doc(cfg(feature = "tvm")))]
pub mod tvm;
#[cfg(feature = "tvm")]
#[cfg_attr(docsrs, doc(cfg(feature = "tvm")))]
pub use self::tvm::TVM;
#[cfg(feature = "nnapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "nnapi")))]
pub mod nnapi;
#[cfg(feature = "nnapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "nnapi")))]
pub use self::nnapi::NNAPI;
#[cfg(feature = "qnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "qnn")))]
pub mod qnn;
#[cfg(feature = "qnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "qnn")))]
pub use self::qnn::QNN;
#[cfg(feature = "xnnpack")]
#[cfg_attr(docsrs, doc(cfg(feature = "xnnpack")))]
pub mod xnnpack;
#[cfg(feature = "xnnpack")]
#[cfg_attr(docsrs, doc(cfg(feature = "xnnpack")))]
pub use self::xnnpack::XNNPACK;
#[cfg(feature = "armnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "armnn")))]
pub mod armnn;
#[cfg(feature = "armnn")]
#[cfg_attr(docsrs, doc(cfg(feature = "armnn")))]
#[allow(deprecated)]
pub use self::armnn::ArmNN;
#[cfg(feature = "migraphx")]
#[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
pub mod migraphx;
#[cfg(feature = "migraphx")]
#[cfg_attr(docsrs, doc(cfg(feature = "migraphx")))]
pub use self::migraphx::MIGraphX;
#[cfg(all(feature = "api-18", feature = "vitis"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "api-18", feature = "vitis"))))]
pub mod vitis;
#[cfg(all(feature = "api-18", feature = "vitis"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "api-18", feature = "vitis"))))]
pub use self::vitis::Vitis;
#[cfg(feature = "rknpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "rknpu")))]
pub mod rknpu;
#[cfg(feature = "rknpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "rknpu")))]
pub use self::rknpu::RKNPU;
#[cfg(any(target_arch = "wasm32", feature = "webgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(target_arch = "wasm32", feature = "webgpu"))))]
pub mod webgpu;
#[cfg(any(target_arch = "wasm32", feature = "webgpu"))]
#[cfg_attr(docsrs, doc(cfg(any(target_arch = "wasm32", feature = "webgpu"))))]
pub use self::webgpu::WebGPU;
#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub mod azure;
#[cfg(feature = "azure")]
#[cfg_attr(docsrs, doc(cfg(feature = "azure")))]
pub use self::azure::Azure;
#[cfg(feature = "vsinpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "vsinpu")))]
pub mod vsinpu;
#[cfg(feature = "vsinpu")]
#[cfg_attr(docsrs, doc(cfg(feature = "vsinpu")))]
pub use self::vsinpu::VSINPU;
#[cfg(feature = "nvrtx")]
#[cfg_attr(docsrs, doc(cfg(feature = "nvrtx")))]
pub mod nvrtx;
#[cfg(feature = "nvrtx")]
#[cfg_attr(docsrs, doc(cfg(feature = "nvrtx")))]
pub use self::nvrtx::NVRTX;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(target_arch = "wasm32")]
pub mod webnn;
#[cfg(target_arch = "wasm32")]
pub use self::{wasm::WASM, webnn::WebNN};
