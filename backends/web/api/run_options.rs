use core::ffi::CStr;

use ort_sys::*;

use crate::{
	session::RunOptions,
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateRunOptions(out: *mut *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { out.write((Box::leak(Box::new(RunOptions::default())) as *mut RunOptions).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunLogVerbosityLevel(options: *mut OrtRunOptions, log_verbosity_level: c_int) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.js.log_verbosity_level = Some(log_verbosity_level);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunLogSeverityLevel(options: *mut OrtRunOptions, log_severity_level: c_int) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.js.log_severity_level = u8::try_from(log_severity_level).ok();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunLogVerbosityLevel(options: *const OrtRunOptions, log_verbosity_level: *mut c_int) -> OrtStatusPtr {
	unsafe { log_verbosity_level.write((*options.cast::<RunOptions>()).js.log_verbosity_level.unwrap_or(0)) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunLogSeverityLevel(options: *const OrtRunOptions, log_severity_level: *mut c_int) -> OrtStatusPtr {
	unsafe { log_severity_level.write((*options.cast::<RunOptions>()).js.log_severity_level.map_or(2, |l| l as c_int)) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetRunTag(options: *mut OrtRunOptions, run_tag: *const c_char) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<RunOptions>() };
	let tag = unsafe { CStr::from_ptr(run_tag) };
	options.js.tag = Some(tag.to_string_lossy().into_owned());
	options.tag = tag.to_owned();
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsGetRunTag(options: *const OrtRunOptions, run_tag: *mut *const c_char) -> OrtStatusPtr {
	unsafe { run_tag.write((*options.cast::<RunOptions>()).tag.as_ptr()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsSetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.js.terminate = Some(true);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn RunOptionsUnsetTerminate(options: *mut OrtRunOptions) -> OrtStatusPtr {
	unsafe { &mut *options.cast::<RunOptions>() }.js.terminate = Some(false);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddRunConfigEntry(options: *mut OrtRunOptions, config_key: *const c_char, config_value: *const c_char) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<RunOptions>() };
	let key = trys!(unsafe { str_from_ptr(config_key) });
	let value = trys!(unsafe { str_from_ptr(config_value) });
	options.js.extra.get_or_insert_default().insert(key.to_string(), value.to_string());
	options.config.insert(key.to_string(), unsafe { CStr::from_ptr(config_value) }.to_owned());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseRunOptions(input: *mut OrtRunOptions) {
	drop(unsafe { Box::<RunOptions>::from_raw(input.cast()) });
}
