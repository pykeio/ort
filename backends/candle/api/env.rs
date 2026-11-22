use ort_sys::*;

use crate::Environment;

pub(super) unsafe extern "system" fn CreateEnv(_log_severity_level: OrtLoggingLevel, _logid: *const c_char, out: *mut *mut OrtEnv) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateEnvWithCustomLogger(
	_logging_function: OrtLoggingFunction,
	_logger_param: *mut c_void,
	_log_severity_level: OrtLoggingLevel,
	_logid: *const c_char,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateEnvWithGlobalThreadPools(
	_log_severity_level: OrtLoggingLevel,
	_logid: *const c_char,
	_tp_options: *const OrtThreadingOptions,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateEnvWithCustomLoggerAndGlobalThreadPools(
	_logging_function: OrtLoggingFunction,
	_logger_param: *mut c_void,
	_log_severity_level: OrtLoggingLevel,
	_logid: *const c_char,
	_tp_options: *const OrtThreadingOptions,
	out: *mut *mut OrtEnv
) -> OrtStatusPtr {
	unsafe { out.write(Environment::new_sys()) };
	OrtStatusPtr::default()
}

struct ThreadingOptions;

pub(super) unsafe extern "system" fn CreateThreadingOptions(out: *mut *mut OrtThreadingOptions) -> OrtStatusPtr {
	unsafe { out.write(Box::into_raw(Box::new(ThreadingOptions)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseThreadingOptions(input: *mut OrtThreadingOptions) {
	drop(unsafe { Box::from_raw(input.cast::<ThreadingOptions>()) });
}

pub(super) unsafe extern "system" fn ReleaseEnv(input: *mut OrtEnv) {
	drop(unsafe { Environment::consume_sys(input) });
}
