use std::{ffi::CStr, sync::atomic::Ordering};

use ort_sys::*;

use crate::{
	session::RunOptions,
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateRunOptions(out: *mut *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { out.write(Box::into_raw(Box::new(RunOptions::default())).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunLogVerbosityLevel(options: *mut OrtRunOptions, log_verbosity_level: c_int) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.log_verbosity_level = log_verbosity_level;
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunLogSeverityLevel(options: *mut OrtRunOptions, log_severity_level: c_int) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.log_severity_level = log_severity_level;
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunTag(options: *mut OrtRunOptions, run_tag: *const c_char) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.tag = unsafe { CStr::from_ptr(run_tag) }.to_owned();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunLogVerbosityLevel(options: *const OrtRunOptions, log_verbosity_level: *mut c_int) -> OrtStatusPtr {
	unsafe { log_verbosity_level.write((*options.cast::<RunOptions>()).log_verbosity_level) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunLogSeverityLevel(options: *const OrtRunOptions, log_severity_level: *mut c_int) -> OrtStatusPtr {
	unsafe { log_severity_level.write((*options.cast::<RunOptions>()).log_severity_level) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunTag(options: *const OrtRunOptions, run_tag: *mut *const c_char) -> OrtStatusPtr {
	unsafe { run_tag.write((*options.cast::<RunOptions>()).tag.as_ptr()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { &*options.cast::<RunOptions>() }.terminate.store(true, Ordering::Release);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsUnsetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { &*options.cast::<RunOptions>() }.terminate.store(false, Ordering::Release);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddRunConfigEntry(options: *mut OrtRunOptions, config_key: *const c_char, config_value: *const c_char) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<RunOptions>() };
	let key = trys!(unsafe { str_from_ptr(config_key) });
	options.config.insert(key.to_string(), unsafe { CStr::from_ptr(config_value) }.to_owned());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseRunOptions(input: *mut OrtRunOptions) {
	drop(unsafe { Box::from_raw(input.cast::<RunOptions>()) });
}
