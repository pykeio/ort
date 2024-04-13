use std::{fmt::Debug, os::raw::c_char};

use crate::{char_p_to_string, ortsys, Error, Result, SessionBuilder};

mod cpu;
pub use self::cpu::CPUExecutionProvider;
mod cuda;
pub use self::cuda::{CUDAExecutionProvider, CUDAExecutionProviderCuDNNConvAlgoSearch};
mod tensorrt;
pub use self::tensorrt::TensorRTExecutionProvider;
mod onednn;
pub use self::onednn::OneDNNExecutionProvider;
mod acl;
pub use self::acl::ACLExecutionProvider;
mod openvino;
pub use self::openvino::OpenVINOExecutionProvider;
mod coreml;
pub use self::coreml::CoreMLExecutionProvider;
mod rocm;
pub use self::rocm::ROCmExecutionProvider;
mod cann;
pub use self::cann::{CANNExecutionProvider, CANNExecutionProviderImplementationMode, CANNExecutionProviderPrecisionMode};
mod directml;
pub use self::directml::DirectMLExecutionProvider;
mod tvm;
pub use self::tvm::{TVMExecutionProvider, TVMExecutorType, TVMTuningType};
mod nnapi;
pub use self::nnapi::NNAPIExecutionProvider;
mod qnn;
pub use self::qnn::{QNNExecutionProvider, QNNExecutionProviderPerformanceMode};
mod xnnpack;
pub use self::xnnpack::XNNPACKExecutionProvider;
mod armnn;
pub use self::armnn::ArmNNExecutionProvider;

/// ONNX Runtime works with different hardware acceleration libraries through its extensible **Execution Providers**
/// (EP) framework to optimally execute the ONNX models on the hardware platform. This interface enables flexibility for
/// the AP application developer to deploy their ONNX models in different environments in the cloud and the edge and
/// optimize the execution by taking advantage of the compute capabilities of the platform.
///
/// ![](https://www.onnxruntime.ai/images/ONNX_Runtime_EP1.png)
pub trait ExecutionProvider {
	/// Returns the identifier of this execution provider used internally by ONNX Runtime.
	///
	/// This is the same as what's used in ONNX Runtime's Python API to register this execution provider, i.e.
	/// [`TVMExecutionProvider`]'s identifier is `TvmExecutionProvider`.
	fn as_str(&self) -> &'static str;

	/// Returns whether this execution provider is supported on this platform.
	///
	/// For example, the CoreML execution provider implements this as:
	/// ```ignore
	/// impl ExecutionProvider for CoreMLExecutionProvider {
	/// 	fn supported_by_platform() -> bool {
	/// 		cfg!(any(target_os = "macos", target_os = "ios"))
	/// 	}
	/// }
	/// ```
	fn supported_by_platform(&self) -> bool {
		true
	}

	/// Returns `Ok(true)` if ONNX Runtime was compiled with support for this execution provider, and `Ok(false)`
	/// otherwise.
	///
	/// An `Err` may be returned if a serious internal error occurs, in which case your application should probably
	/// just abort.
	///
	/// Note that this does not always mean the execution provider is *usable* for a specific model. A model may use
	/// operators not supported by an execution provider, or the EP may encounter an error while attempting to load a
	/// dynamic library during registration. In most cases (i.e. showing the user an error message if CUDA could not be
	/// enabled), you'll instead want to detect and handle errors from [`ExecutionProvider::register`].
	fn is_available(&self) -> Result<bool> {
		let mut providers: *mut *mut c_char = std::ptr::null_mut();
		let mut num_providers = 0;
		ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers) -> Error::GetAvailableProviders];
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
			if self.as_str() == avail {
				let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
				return Ok(true);
			}
		}

		let _ = ortsys![unsafe ReleaseAvailableProviders(providers, num_providers)];
		Ok(false)
	}

	/// Attempts to register this execution provider on the given session.
	fn register(&self, session_builder: &SessionBuilder) -> Result<()>;
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

/// Execution provider container. See [the ONNX Runtime docs](https://onnxruntime.ai/docs/execution-providers/) for more
/// info on execution providers. Execution providers are actually registered via the functions [`crate::SessionBuilder`]
/// (per-session) or [`EnvironmentBuilder`](crate::environment::EnvironmentBuilder) (default for all sessions in an
/// environment).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
#[non_exhaustive]
pub enum ExecutionProviderDispatch {
	CPU(CPUExecutionProvider),
	CUDA(CUDAExecutionProvider),
	TensorRT(TensorRTExecutionProvider),
	OpenVINO(OpenVINOExecutionProvider),
	ACL(ACLExecutionProvider),
	OneDNN(OneDNNExecutionProvider),
	CoreML(CoreMLExecutionProvider),
	DirectML(DirectMLExecutionProvider),
	ROCm(ROCmExecutionProvider),
	NNAPI(NNAPIExecutionProvider),
	QNN(QNNExecutionProvider),
	TVM(TVMExecutionProvider),
	CANN(CANNExecutionProvider),
	XNNPACK(XNNPACKExecutionProvider),
	ArmNN(ArmNNExecutionProvider)
}

macro_rules! impl_dispatch {
	($($variant:ident),*) => {
		impl ExecutionProvider for ExecutionProviderDispatch {
			fn as_str(&self) -> &'static str {
				match self {
					$(Self::$variant(inner) => inner.as_str(),)*
				}
			}

			fn is_available(&self) -> $crate::Result<bool> {
				match self {
					$(Self::$variant(inner) => inner.is_available(),)*
				}
			}

			fn register(&self, session_builder: &$crate::SessionBuilder) -> $crate::Result<()> {
				match self {
					$(Self::$variant(inner) => inner.register(session_builder),)*
				}
			}
		}
	};
}

impl_dispatch!(CPU, CUDA, TensorRT, ACL, OneDNN, OpenVINO, CoreML, CANN, ROCm, DirectML, TVM, NNAPI, QNN, XNNPACK, ArmNN);

#[allow(unused)]
macro_rules! map_keys {
	($($fn_name:ident = $ex:expr),*) => {
		{
			let mut keys = ::std::vec::Vec::<std::ffi::CString>::new();
			let mut values = ::std::vec::Vec::<std::ffi::CString>::new();
			$(
				if let Some(v) = $ex {
					keys.push(::std::ffi::CString::new(stringify!($fn_name)).unwrap());
					values.push(::std::ffi::CString::new(v.to_string().as_str()).unwrap());
				}
			)*
			assert_eq!(keys.len(), values.len()); // sanity check
			let key_ptrs: ::std::vec::Vec<*const ::std::ffi::c_char> = keys.iter().map(|k| k.as_ptr()).collect();
			let value_ptrs: ::std::vec::Vec<*const ::std::ffi::c_char> = values.iter().map(|v| v.as_ptr()).collect();
			(key_ptrs, value_ptrs, keys.len(), keys, values)
		}
	};
}
#[allow(unused)]
pub(crate) use map_keys;

#[allow(unused)]
macro_rules! get_ep_register {
	($symbol:ident($($id:ident: $type:ty),*) -> $rt:ty) => {
		#[cfg(feature = "load-dynamic")]
		#[allow(non_snake_case)]
		let $symbol = unsafe {
			let dylib = $crate::lib_handle();
			let symbol: ::std::result::Result<
				::libloading::Symbol<unsafe extern "C" fn($($id: $type),*) -> $rt>,
				::libloading::Error
			> = dylib.get(stringify!($symbol).as_bytes());
			match symbol {
				Ok(symbol) => symbol.into_raw(),
				Err(e) => {
					return ::std::result::Result::Err($crate::Error::DlLoad { symbol: stringify!($symbol), error: e.to_string() });
				}
			}
		};
	};
}
#[allow(unused)]
pub(crate) use get_ep_register;

#[tracing::instrument(skip_all)]
pub(crate) fn apply_execution_providers(session_builder: &SessionBuilder, execution_providers: impl Iterator<Item = ExecutionProviderDispatch>) {
	let execution_providers: Vec<_> = execution_providers.collect();
	let mut fallback_to_cpu = !execution_providers.is_empty();
	for ex in execution_providers {
		if let Err(e) = ex.register(session_builder) {
			if let &Error::ExecutionProviderNotRegistered(ep_name) = &e {
				if ex.supported_by_platform() {
					tracing::warn!("{e}");
				} else {
					tracing::debug!("{e} (additionally, `{ep_name}` is not supported on this platform)");
				}
			} else {
				tracing::warn!("An error occurred when attempting to register `{}`: {e}", ex.as_str());
			}
		} else {
			tracing::info!("Successfully registered `{}`", ex.as_str());
			fallback_to_cpu = false;
		}
	}
	if fallback_to_cpu {
		tracing::warn!("No execution providers registered successfully. Falling back to CPU.");
	}
}
