use alloc::boxed::Box;
#[cfg(feature = "tracing")]
use core::ptr;
use core::{
	ffi::{self, CStr},
	marker::PhantomData,
	ptr::NonNull
};

use crate::{AsPointer, ortsys, util::with_cstr_ptr_array};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
	Verbose,
	Info,
	Warning,
	Error,
	Fatal
}

impl From<LogLevel> for ort_sys::OrtLoggingLevel {
	fn from(value: LogLevel) -> Self {
		match value {
			LogLevel::Verbose => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
			LogLevel::Info => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO,
			LogLevel::Warning => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
			LogLevel::Error => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR,
			LogLevel::Fatal => ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL
		}
	}
}

impl From<ort_sys::OrtLoggingLevel> for LogLevel {
	fn from(value: ort_sys::OrtLoggingLevel) -> Self {
		match value {
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE => LogLevel::Verbose,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO => LogLevel::Info,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING => LogLevel::Warning,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR => LogLevel::Error,
			ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL => LogLevel::Fatal
		}
	}
}

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
pub(crate) extern "system" fn tracing_logger(
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

/// `LoggerFunction` accepts the message's [`LogLevel`], its category, log ID, code location, and the message
/// itself.
pub type LoggerFunction = Box<dyn Fn(LogLevel, &str, &str, &str, &str) + Sync>;

pub(crate) extern "system" fn custom_logger(
	logger: *mut ffi::c_void,
	severity: ort_sys::OrtLoggingLevel,
	category: *const ffi::c_char,
	id: *const ffi::c_char,
	code_location: *const ffi::c_char,
	message: *const ffi::c_char
) {
	if category.is_null() || code_location.is_null() || message.is_null() || id.is_null() {
		return;
	}

	let category = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("<decode error>");
	let code_location = unsafe { CStr::from_ptr(code_location) }.to_str().unwrap_or("<decode error>");
	let message = unsafe { CStr::from_ptr(message) }.to_str().unwrap_or("<decode error>");
	let id = unsafe { CStr::from_ptr(id) }.to_str().unwrap_or("<decode error>");

	let logger = logger.cast::<LoggerFunction>();
	unsafe { (*logger)(LogLevel::from(severity), category, id, code_location, message) };
}

/// A reference to a session's logger, typically obtained in custom operator contexts.
///
/// Messages can be logged to a [`Logger`] via the [`log!`](crate::log) macro.
#[derive(Debug)]
pub struct Logger<'a> {
	ptr: NonNull<ort_sys::OrtLogger>,
	_p: PhantomData<&'a ()>
}

impl<'a> Logger<'a> {
	pub(crate) unsafe fn from_raw(ptr: NonNull<ort_sys::OrtLogger>) -> Self {
		Self { ptr, _p: PhantomData }
	}

	/// Returns the current [`LogLevel`] of this logger.
	pub fn level(&self) -> LogLevel {
		let mut log_level = ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
		ortsys![unsafe Logger_GetLoggingSeverityLevel(self.ptr(), &mut log_level as *mut ort_sys::OrtLoggingLevel as *mut _).expect("infallible")];
		LogLevel::from(log_level)
	}

	/// Logs a message to this logger with the given level and message.
	///
	/// For more convenient usage, see the [`log!`](crate::log) macro.
	pub fn log(&self, level: LogLevel, message: &str, file_path: &str, line: u32, func_name: &str) {
		let _ = with_cstr_ptr_array(&[message, func_name], &|arr| {
			let (message, func_name) = (arr[0], arr[1]);
			#[cfg(target_family = "windows")]
			{
				let file_path = crate::util::str_to_os_char(file_path);
				ortsys![unsafe Logger_LogMessage(self.ptr.as_ptr(), level.into(), message, file_path.as_ptr(), line as _, func_name)?];
				Ok(())
			}
			#[cfg(not(target_family = "windows"))]
			crate::util::with_cstr(file_path.as_bytes(), &|file_path| {
				ortsys![unsafe Logger_LogMessage(self.ptr.as_ptr(), level.into(), message, file_path.as_ptr(), line as _, func_name)?];
				Ok(())
			})
		});
	}
}

impl<'a> AsPointer for Logger<'a> {
	type Sys = ort_sys::OrtLogger;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

/// Logs a message to a given [`Logger`].
///
/// ```
/// # use ort::operator::kernel::{Kernel, KernelContext};
/// struct MyKernel;
///
/// impl Kernel for MyKernel {
/// 	fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
/// 		let logger = ctx.logger()?;
/// 		ort::log!(logger, Warning @ "something is off");
///
/// 		// log!() can also be used with formatting arguments:
/// 		ort::log!(logger, Info @ "value: {:?}", 42);
///
/// 		Ok(())
/// 	}
/// }
/// ```
///
/// See [`LogLevel`] for supported log levels.
#[macro_export]
macro_rules! log {
	($logger:expr, $level:ident @ $fmt:expr) => {{
		($logger).log(
			$crate::logging::LogLevel::$level,
			&$crate::__private::alloc::format!($fmt),
			$crate::__private::core::file!(),
			$crate::__private::core::line!(),
			$crate::__private::core::module_path!()
		);
	}};
	($logger:expr, $level:ident @ $fmt:expr, $($arg:tt),+) => {{
		($logger).log(
			$crate::logging::LogLevel::$level,
			&$crate::__private::alloc::format!($fmt, $($arg),+),
			$crate::__private::core::file!(),
			$crate::__private::core::line!(),
			$crate::__private::core::module_path!()
		);
	}};
}
