use ort_sys::*;

use crate::env::Environment;

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

pub(super) unsafe extern "system" fn EnableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	let env = unsafe { Environment::cast_from_sys_mut(env.cast_mut()) };
	env.with_telemetry = true;
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn DisableTelemetryEvents(env: *const OrtEnv) -> OrtStatusPtr {
	let env = unsafe { Environment::cast_from_sys_mut(env.cast_mut()) };
	env.with_telemetry = false;
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseEnv(input: *mut OrtEnv) {
	drop(unsafe { Environment::consume_sys(input) });
}
