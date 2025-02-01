#[cfg(feature = "tracing")]
use core::{
	ffi::{self, CStr},
	ptr
};

macro_rules! trace {
	($($arg:tt)+) => {{
		#[cfg(feature = "tracing")]
		tracing::trace!($($arg)+);
	}}
}
macro_rules! debug {
	($($arg:tt)+) => {{
		#[cfg(feature = "tracing")]
		tracing::debug!($($arg)+);
	}}
}
macro_rules! info {
	($($arg:tt)+) => {{
		#[cfg(feature = "tracing")]
		tracing::info!($($arg)+);
	}}
}
macro_rules! warning {
	($($arg:tt)+) => {{
		#[cfg(feature = "tracing")]
		tracing::warn!($($arg)+);
	}}
}
macro_rules! error {
	($($arg:tt)+) => {{
		#[cfg(feature = "tracing")]
		tracing::error!($($arg)+);
	}}
}
pub(crate) use debug;
pub(crate) use error;
pub(crate) use info;
pub(crate) use trace;
pub(crate) use warning;

#[cfg(not(feature = "tracing"))]
pub fn default_log_level() -> ort_sys::OrtLoggingLevel {
	#[cfg(feature = "std")]
	match std::env::var("ORT_LOG").as_deref() {
		Ok("fatal") => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL,
		Ok("error") => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
		Ok("warning") => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
		Ok("info") => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
		Ok("verbose") => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
		_ => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR
	}
	#[cfg(not(feature = "std"))]
	ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING
}

/// Callback from C that will handle ONNX logging, forwarding ONNX's logs to the `tracing` crate.
#[cfg(feature = "tracing")]
pub(crate) extern "system" fn custom_logger(
	_params: *mut ffi::c_void,
	severity: ort_sys::OrtLoggingLevel,
	_: *const ffi::c_char,
	id: *const ffi::c_char,
	code_location: *const ffi::c_char,
	message: *const ffi::c_char
) {
	assert_ne!(code_location, ptr::null());
	let code_location = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("<decode error>");
	assert_ne!(message, ptr::null());
	let message = unsafe { CStr::from_ptr(message) }.to_str().unwrap_or("<decode error>");
	assert_ne!(id, ptr::null());
	let id = unsafe { CStr::from_ptr(id) }.to_str().unwrap_or("<decode error>");

	let span = tracing::span!(tracing::Level::TRACE, "ort", id = id, location = code_location);

	match severity {
		ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => tracing::event!(parent: &span, tracing::Level::TRACE, "{message}"),
		ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => tracing::event!(parent: &span, tracing::Level::INFO, "{message}"),
		ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => tracing::event!(parent: &span, tracing::Level::WARN, "{message}"),
		ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => tracing::event!(parent: &span, tracing::Level::ERROR, "{message}"),
		ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => tracing::event!(parent: &span, tracing::Level::ERROR, "(FATAL): {message}")
	}
}
