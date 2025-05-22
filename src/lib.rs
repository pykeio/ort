#![doc(html_logo_url = "https://ort.pyke.io/assets/icon.png")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::tabs_in_doc_comments, clippy::arc_with_non_send_sync)]
#![allow(clippy::macro_metavars_in_unsafe)]
#![warn(clippy::unwrap_used)]
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]

//! <div align=center>
//! 	<img src="https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/docs/trend-banner.png" width="350px">
//! 	<hr />
//! </div>
//!
//! `ort` is a Rust binding for [ONNX Runtime](https://onnxruntime.ai/). For information on how to get started with `ort`,
//! see <https://ort.pyke.io/introduction>.

extern crate alloc;
extern crate core;

#[doc(hidden)]
pub mod __private {
	pub extern crate alloc;
	pub extern crate core;
}
#[macro_use]
pub(crate) mod private;

pub mod adapter;
pub mod compiler;
pub mod editor;
pub mod environment;
pub mod error;
pub mod execution_providers;
pub mod io_binding;
pub mod logging;
pub mod memory;
pub mod metadata;
pub mod operator;
pub mod session;
pub mod tensor;
#[cfg(feature = "training")]
#[cfg_attr(docsrs, doc(cfg(feature = "training")))]
pub mod training;
pub mod util;
pub mod value;
pub mod api {
	#[cfg(feature = "training")]
	pub use super::training::training_api as training;
	pub use super::{api as ort, compiler::compile_api as compile, editor::editor_api as editor};
}

#[cfg(feature = "load-dynamic")]
use alloc::sync::Arc;
use alloc::{borrow::ToOwned, boxed::Box, string::String};
use core::{
	ffi::{CStr, c_char},
	ptr::NonNull,
	str
};

pub use ort_sys as sys;

#[cfg(feature = "load-dynamic")]
pub use self::environment::init_from;
pub(crate) use self::logging::{debug, error, info, trace, warning as warn};
use self::util::OnceLock;
pub use self::{
	environment::init,
	error::{Error, ErrorCode, Result}
};

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

/// Returns information about the build of ONNX Runtime used, including version, Git commit, and compile flags.
///
/// ```
/// println!("{}", ort::info());
/// // ORT Build Info: git-branch=rel-1.19.0, git-commit-id=26250ae, build type=Release, cmake cxx flags: /DWIN32 /D_WINDOWS /EHsc /Zc:__cplusplus /EHsc /wd26812 -DEIGEN_HAS_C99_MATH -DCPUINFO_SUPPORTED
/// ```
pub fn info() -> &'static str {
	let str = unsafe { ortsys![GetBuildInfoString]() };
	unsafe { CStr::from_ptr(str) }.to_str().expect("invalid build info string")
}

struct ApiPointer(NonNull<ort_sys::OrtApi>);
unsafe impl Send for ApiPointer {}
unsafe impl Sync for ApiPointer {}

static G_ORT_API: OnceLock<ApiPointer> = OnceLock::new();

/// Returns a reference to the global [`ort_sys::OrtApi`] object.
///
/// # Panics
/// May panic if:
/// - The `alternative-backend` feature is enabled and [`set_api`] was not yet called.
/// - Getting the `OrtApi` struct fails, due to `ort` loading an unsupported version of ONNX Runtime.
/// - Loading the ONNX Runtime dynamic library fails if the `load-dynamic` feature is enabled.
pub fn api() -> &'static ort_sys::OrtApi {
	#[cfg(feature = "alternative-backend")]
	let ptr = G_ORT_API
		.get()
		.expect(
			"attempted to use `ort` APIs before initializing a backend\nwhen the `alternative-backend` feature is enabled, `ort::set_api` must be called to configure the `OrtApi` used by the library"
		)
		.0;
	#[cfg(not(feature = "alternative-backend"))]
	let ptr = G_ORT_API
		.get_or_init(|| {
			#[cfg(feature = "load-dynamic")]
			unsafe {
				use core::cmp::Ordering;

				let dylib = lib_handle();
				let base_getter: libloading::Symbol<unsafe extern "C" fn() -> *const ort_sys::OrtApiBase> = dylib
					.get(b"OrtGetApiBase")
					.expect("`OrtGetApiBase` must be present in ONNX Runtime dylib");
				let base: *const ort_sys::OrtApiBase = base_getter();
				assert!(!base.is_null());

				let version_string = ((*base).GetVersionString)();
				let version_string = CStr::from_ptr(version_string).to_string_lossy();
				crate::info!("Loaded ONNX Runtime dylib with version '{version_string}'");

				let lib_minor_version = version_string.split('.').nth(1).map_or(0, |x| x.parse::<u32>().unwrap_or(0));
				match lib_minor_version.cmp(&MINOR_VERSION) {
					Ordering::Less => panic!(
						"ort {} is not compatible with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.{MINOR_VERSION}.x', but got '{version_string}'",
						env!("CARGO_PKG_VERSION"),
						dylib_path()
					),
					Ordering::Greater => crate::warn!(
						"ort {} may have compatibility issues with the ONNX Runtime binary found at `{}`; expected GetVersionString to return '1.{MINOR_VERSION}.x', but got '{version_string}'",
						env!("CARGO_PKG_VERSION"),
						dylib_path()
					),
					Ordering::Equal => {}
				};
				let api: *const ort_sys::OrtApi = ((*base).GetApi)(ort_sys::ORT_API_VERSION);
				ApiPointer(NonNull::new(api.cast_mut()).expect("Failed to initialize ORT API"))
			}
			#[cfg(not(feature = "load-dynamic"))]
			unsafe {
				let base: *const ort_sys::OrtApiBase = ort_sys::OrtGetApiBase();
				assert!(!base.is_null());
				let api: *const ort_sys::OrtApi = ((*base).GetApi)(ort_sys::ORT_API_VERSION);
				ApiPointer(NonNull::new(api.cast_mut()).expect("Failed to initialize ORT API"))
			}
		})
		.0;
	unsafe { ptr.as_ref() }
}

/// Sets the global [`ort_sys::OrtApi`] interface used by `ort` in order to use alternative backends, or a custom
/// loading scheme.
///
/// When using `alternative-backend`, this must be called before using any other `ort` API.
///
/// Returns `true` if successful (i.e. no API has been set up to this point). This function will not override the API if
/// one was already set.
pub fn set_api(api: ort_sys::OrtApi) -> bool {
	G_ORT_API.try_insert_with(|| ApiPointer(unsafe { NonNull::new_unchecked(Box::leak(Box::new(api))) }))
}

/// Trait to access raw pointers from safe types which wrap unsafe [`ort_sys`] types.
pub trait AsPointer {
	/// This safe type's corresponding [`ort_sys`] type.
	type Sys;

	/// Returns the underlying [`ort_sys`] type pointer this safe type wraps. The pointer is guaranteed to be non-null.
	fn ptr(&self) -> *const Self::Sys;

	/// Returns the underlying [`ort_sys`] type pointer this safe type wraps as a mutable pointer. The pointer is
	/// guaranteed to be non-null.
	fn ptr_mut(&mut self) -> *mut Self::Sys {
		self.ptr().cast_mut()
	}
}

#[macro_export]
macro_rules! ortsys {
	($method:ident) => {
		($crate::api().$method)
	};
	(unsafe $method:ident($($n:expr),* $(,)?)) => {
		ortsys![@ort: unsafe $method($($n),*)]
	};
	(unsafe $method:ident($($n:expr),* $(,)?).expect($e:expr)) => {
		ortsys![@ort: unsafe $method($($n),*) as Result].expect($e)
	};
	(unsafe $method:ident($($n:expr),* $(,)?).expect($e:expr); nonNull($($check:ident),+ $(,)?)$(;)?) => {
		ortsys![unsafe $method($($n),*).expect($e)];
		ortsys![@nonNull_panic; $($check),+];
	};
	(unsafe $method:ident($($n:expr),* $(,)?); nonNull($($check:ident),+ $(,)?)$(;)?) => {
		let _x = ortsys![unsafe $method($($n),*)];
		ortsys![@nonNull_panic; $($check),+];
		_x
	};
	(unsafe $method:ident($($n:expr),* $(,)?)?) => {
		ortsys![@ort: unsafe $method($($n),+) as Result]?;
	};
	(unsafe $method:ident($($n:expr),* $(,)?)?; nonNull($($check:ident),+)$(;)?) => {
		ortsys![unsafe $method($($n),*)?];
		ortsys![@nonNull?; $($check),+];
	};

	(@nonNull_panic; $($check:ident),+) => {
		$(
			let Some($check) = $crate::__private::core::ptr::NonNull::new($check as *mut _) else {
				$crate::util::cold();
				$crate::__private::core::panic!(concat!("expected `", stringify!($check), "` to not be null"));
			};
		)+
	};
	(@nonNull?; $($check:ident),+) => {
		$(
			let Some($check) = $crate::__private::core::ptr::NonNull::new($check as *mut _) else {
				$crate::util::cold();
				return Err($crate::Error::new(concat!("expected `", stringify!($check), "` to not be null")));
			};
		)+
	};

	(@ort: unsafe $method:ident($($n:expr),*)) => {
		unsafe { ($crate::api().$method)($($n),*) }
	};
	(@ort: unsafe $method:ident($($n:expr),*) as Result) => {
		unsafe { $crate::error::status_to_result(($crate::api().$method)($($n),+)) }
	};
	(@$api:ident: unsafe $method:ident($($n:expr),*)) => {
		unsafe { ($crate::api::$api().unwrap().$method)($($n),+) }
	};
	(@$api:ident: unsafe $method:ident($($n:expr),*)?) => {
		$crate::api::$api().and_then(|api| unsafe { $crate::error::status_to_result((api.$method)($($n),+)) })?
	};
	(@$api:ident: unsafe $method:ident($($n:expr),*)?; nonNull($($check:ident),+)$(;)?) => {
		$crate::api::$api().and_then(|api| unsafe { $crate::error::status_to_result((api.$method)($($n),+)) })?;
		ortsys![@nonNull?; $($check),+];
	};
	(@$api:ident: unsafe $method:ident($($n:expr),*) as Result) => {
		$crate::api::$api().and_then(|api| unsafe { $crate::error::status_to_result((api.$method)($($n),+)) })
	};
}

pub(crate) fn char_p_to_string(raw: *const c_char) -> Result<String> {
	if raw.is_null() {
		return Ok(String::new());
	}
	let c_string = unsafe { CStr::from_ptr(raw.cast_mut()).to_owned() };
	Ok(c_string.to_string_lossy().into())
}

#[cfg(test)]
mod test {
	use alloc::ffi::CString;

	use super::*;

	#[test]
	fn test_char_p_to_string() {
		let s = CString::new("foo").unwrap_or_else(|_| unreachable!());
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).expect("failed to convert string"));
	}
}
