#![doc(html_logo_url = "https://raw.githubusercontent.com/pykeio/ort/v2/docs/icon.png")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::tabs_in_doc_comments, clippy::arc_with_non_send_sync)]
#![warn(clippy::unwrap_used)]

//! <div align=center>
//! 	<img src="https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/docs/trend-banner.png" width="350px">
//! 	<hr />
//! </div>
//!
//! `ort` is a Rust binding for [ONNX Runtime](https://onnxruntime.ai/). For information on how to get started with `ort`,
//! see <https://ort.pyke.io/introduction>.

#[cfg(all(test, not(feature = "fetch-models")))]
compile_error!("`cargo test --features fetch-models`!!1!");

pub(crate) mod environment;
pub(crate) mod error;
pub(crate) mod execution_providers;
pub(crate) mod io_binding;
pub(crate) mod memory;
pub(crate) mod metadata;
pub(crate) mod operator;
pub(crate) mod session;
pub(crate) mod tensor;
pub(crate) mod value;
#[cfg_attr(docsrs, doc(cfg(target_arch = "wasm32")))]
#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "load-dynamic")]
use std::sync::Arc;
use std::{
	ffi::CStr,
	os::raw::c_char,
	ptr::{self, NonNull},
	sync::{
		atomic::{AtomicPtr, Ordering},
		OnceLock
	}
};

pub use ort_sys as sys;

#[cfg(feature = "load-dynamic")]
pub use self::environment::init_from;
pub use self::environment::{get_environment, init, Environment, EnvironmentBuilder, EnvironmentGlobalThreadPoolOptions};
#[cfg(feature = "fetch-models")]
#[cfg_attr(docsrs, doc(cfg(feature = "fetch-models")))]
pub use self::error::FetchModelError;
pub use self::error::{Error, ErrorInternal, Result};
pub use self::execution_providers::*;
pub use self::io_binding::IoBinding;
pub use self::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
pub use self::metadata::ModelMetadata;
pub use self::operator::{
	io::{OperatorInput, OperatorOutput},
	kernel::{Kernel, KernelAttributes, KernelContext},
	InferShapeFn, Operator, OperatorDomain
};
pub use self::session::{
	GraphOptimizationLevel, InMemorySession, Input, Output, RunOptions, Session, SessionBuilder, SessionInputValue, SessionInputs, SessionOutputs,
	SharedSessionInner
};
#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
pub use self::tensor::ArrayExtensions;
pub use self::tensor::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType, Utf8Data};
pub use self::value::{
	DowncastableTarget, DynMap, DynMapRef, DynMapRefMut, DynMapValueType, DynSequence, DynSequenceRef, DynSequenceRefMut, DynSequenceValueType, DynTensor,
	DynTensorRef, DynTensorRefMut, DynTensorValueType, DynValue, DynValueTypeMarker, Map, MapRef, MapRefMut, MapValueType, MapValueTypeMarker, Sequence,
	SequenceRef, SequenceRefMut, SequenceValueType, SequenceValueTypeMarker, Tensor, TensorRef, TensorRefMut, TensorValueTypeMarker, Value, ValueRef,
	ValueRefMut, ValueType, ValueTypeMarker
};

#[cfg(not(all(target_arch = "x86", target_os = "windows")))]
macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "C" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "C" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "C" fn $($tt)*);
}
#[cfg(all(target_arch = "x86", target_os = "windows"))]
macro_rules! extern_system_fn {
	($(#[$meta:meta])* fn $($tt:tt)*) => ($(#[$meta])* extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis fn $($tt:tt)*) => ($(#[$meta])* $vis extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* unsafe fn $($tt:tt)*) => ($(#[$meta])* unsafe extern "stdcall" fn $($tt)*);
	($(#[$meta:meta])* $vis:vis unsafe fn $($tt:tt)*) => ($(#[$meta])* $vis unsafe extern "stdcall" fn $($tt)*);
}

pub(crate) use extern_system_fn;

/// The minor version of ONNX Runtime used by this version of `ort`.
pub const MINOR_VERSION: u32 = ort_sys::ORT_API_VERSION;

#[cfg(feature = "load-dynamic")]
pub(crate) static G_ORT_DYLIB_PATH: OnceLock<Arc<String>> = OnceLock::new();
#[cfg(feature = "load-dynamic")]
pub(crate) static G_ORT_LIB: OnceLock<Arc<libloading::Library>> = OnceLock::new();

#[cfg(feature = "load-dynamic")]
pub(crate) fn dylib_path() -> &'static String {
	G_ORT_DYLIB_PATH.get_or_init(|| {
		let path = match std::env::var("ORT_DYLIB_PATH") {
			Ok(s) if !s.is_empty() => s,
			#[cfg(target_os = "windows")]
			_ => "onnxruntime.dll".to_owned(),
			#[cfg(any(target_os = "linux", target_os = "android"))]
			_ => "libonnxruntime.so".to_owned(),
			#[cfg(any(target_os = "macos", target_os = "ios"))]
			_ => "libonnxruntime.dylib".to_owned()
		};
		Arc::new(path)
	})
}

#[cfg(feature = "load-dynamic")]
pub(crate) fn lib_handle() -> &'static libloading::Library {
	G_ORT_LIB.get_or_init(|| {
		// resolve path relative to executable
		let path: std::path::PathBuf = dylib_path().into();
		let absolute_path = if path.is_absolute() {
			path
		} else {
			let relative = std::env::current_exe()
				.expect("could not get current executable path")
				.parent()
				.expect("executable is root?")
				.join(&path);
			if relative.exists() { relative } else { path }
		};
		let lib = unsafe { libloading::Library::new(&absolute_path) }
			.unwrap_or_else(|e| panic!("An error occurred while attempting to load the ONNX Runtime binary at `{}`: {e}", absolute_path.display()));
		Arc::new(lib)
	})
}

pub(crate) static G_ORT_API: OnceLock<AtomicPtr<ort_sys::OrtApi>> = OnceLock::new();

/// Returns a pointer to the global [`ort_sys::OrtApi`] object.
///
/// # Panics
/// May panic if:
/// - Getting the `OrtApi` struct fails, due to `ort` loading an unsupported version of ONNX Runtime.
/// - Loading the ONNX Runtime dynamic library fails if the `load-dynamic` feature is enabled.
///
/// # Examples
/// The primary (public-facing) use case for this function is accessing APIs that do not have a corresponding safe
/// implementation in `ort`. For example, [`GetBuildInfoString`](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0a7dba37b0017c0ef3a0ab4e266a967d):
///
/// ```
/// # use std::ffi::CStr;
/// # fn main() -> ort::Result<()> {
/// let api = ort::api().as_ptr();
/// let build_info = unsafe { CStr::from_ptr((*api).GetBuildInfoString.unwrap()()) };
/// println!("{}", build_info.to_string_lossy());
/// // ORT Build Info: git-branch=HEAD, git-commit-id=4573740, build type=Release, cmake cxx flags: /DWIN32 /D_WINDOWS /EHsc /EHsc /wd26812 -DEIGEN_HAS_C99_MATH -DCPUINFO_SUPPORTED
/// # Ok(())
/// # }
/// ```
///
/// For the full list of ONNX Runtime APIs, consult the [`ort_sys::OrtApi`] struct and the [ONNX Runtime C API](https://onnxruntime.ai/docs/api/c/struct_ort_api.html).
pub fn api() -> NonNull<ort_sys::OrtApi> {
	unsafe {
		NonNull::new_unchecked(
			G_ORT_API
				.get_or_init(|| {
					#[cfg(feature = "load-dynamic")]
					{
						let dylib = lib_handle();
						let base_getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort_sys::OrtApiBase> = dylib
							.get(b"OrtGetApiBase")
							.expect("`OrtGetApiBase` must be present in ONNX Runtime dylib");
						let base: *const ort_sys::OrtApiBase = base_getter();
						assert_ne!(base, ptr::null());

						let get_version_string: extern_system_fn! { unsafe fn () -> *const c_char } =
							(*base).GetVersionString.expect("`GetVersionString` must be present in `OrtApiBase`");
						let version_string = get_version_string();
						let version_string = CStr::from_ptr(version_string).to_string_lossy();
						tracing::info!("Loaded ONNX Runtime dylib with version '{version_string}'");

						let lib_minor_version = version_string.split('.').nth(1).map_or(0, |x| x.parse::<u32>().unwrap_or(0));
						match lib_minor_version.cmp(&MINOR_VERSION) {
							std::cmp::Ordering::Less => panic!(
								"ort 2.0 is not compatible with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.{MINOR_VERSION}.x', but got '{version_string}'",
								dylib_path()
							),
							std::cmp::Ordering::Greater => tracing::warn!(
								"ort 2.0 may have compatibility issues with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.{MINOR_VERSION}.x', but got '{version_string}'",
								dylib_path()
							),
							std::cmp::Ordering::Equal => {}
						};
						let get_api: extern_system_fn! { unsafe fn(u32) -> *const ort_sys::OrtApi } =
							(*base).GetApi.expect("`GetApi` must be present in `OrtApiBase`");
						let api: *const ort_sys::OrtApi = get_api(ort_sys::ORT_API_VERSION);
						assert!(!api.is_null());
						AtomicPtr::new(api.cast_mut())
					}
					#[cfg(not(feature = "load-dynamic"))]
					{
						let base: *const ort_sys::OrtApiBase = ort_sys::OrtGetApiBase();
						assert_ne!(base, ptr::null());
						let get_api: extern_system_fn! { unsafe fn(u32) -> *const ort_sys::OrtApi } =
							(*base).GetApi.expect("`GetApi` must be present in `OrtApiBase`");
						let api: *const ort_sys::OrtApi = get_api(ort_sys::ORT_API_VERSION);
						assert!(!api.is_null());
						AtomicPtr::new(api.cast_mut())
					}
				})
				.load(Ordering::Relaxed)
		)
	}
}

macro_rules! ortsys {
	($method:ident) => {
		$crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))
	};
	(unsafe $method:ident) => {
		unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null"))) }
	};
	($method:ident($($n:expr),+ $(,)?)) => {
		$crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)
	};
	(unsafe $method:ident($($n:expr),+ $(,)?)) => {
		unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }
	};
	($method:ident($($n:expr),+ $(,)?).expect($e:expr)) => {
		$crate::error::status_to_result($crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).expect($e)
	};
	(unsafe $method:ident($($n:expr),+ $(,)?).expect($e:expr)) => {
		$crate::error::status_to_result(unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).expect($e)
	};
	($method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+);
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		let _x = unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) };
		$($crate::error::assert_non_null_pointer($check, stringify!($method)).unwrap();)+
		_x
	}};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result($crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).map_err($err)?;
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result(unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).map_err($err)?;
	};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::error::status_to_result($crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		$crate::error::status_to_result(unsafe { $crate::api().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	}};
}

pub(crate) use ortsys;

pub(crate) fn char_p_to_string(raw: *const c_char) -> Result<String> {
	let c_string = unsafe { CStr::from_ptr(raw.cast_mut()).to_owned() };
	match c_string.into_string() {
		Ok(string) => Ok(string),
		Err(e) => Err(ErrorInternal::IntoStringError(e))
	}
	.map_err(Error::FfiStringConversion)
}

pub(crate) struct PrivateTraitMarker;

macro_rules! private_trait {
	() => {
		#[doc(hidden)]
		#[allow(private_interfaces)]
		fn _private() -> crate::PrivateTraitMarker;
	};
}
macro_rules! private_impl {
	() => {
		#[allow(private_interfaces)]
		fn _private() -> crate::PrivateTraitMarker {
			crate::PrivateTraitMarker
		}
	};
}
pub(crate) use private_impl;
pub(crate) use private_trait;

#[cfg(test)]
mod test {
	use std::ffi::CString;

	use super::*;

	#[test]
	fn test_char_p_to_string() {
		let s = CString::new("foo").unwrap_or_else(|_| unreachable!());
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).expect("failed to convert string"));
	}
}
