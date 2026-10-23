use alloc::string::String;
use core::{
	ffi::{CStr, c_char},
	slice
};

use ort_sys::{OrtErrorCode, stub::Error};
use wasm_bindgen::{JsCast, JsValue};

pub(crate) fn value_to_string(value: &JsValue) -> String {
	js_sys::Object::<JsValue>::unchecked_from_js_ref(value).to_js_string().into()
}

// todo: replace usages of &[i32] so we dont need 2 of these
pub(crate) fn num_elements(dims: &[i32]) -> usize {
	let mut size = 1usize;
	for dim in dims {
		if *dim < 0 {
			return 0;
		}
		size *= *dim as usize;
	}
	size
}

pub(crate) fn num_elements_i64(dims: &[i64]) -> usize {
	let mut size = 1usize;
	for dim in dims {
		if *dim < 0 {
			return 0;
		}
		size *= *dim as usize;
	}
	size
}

macro_rules! trys {
	($e:expr) => {
		match $e {
			Ok(v) => v,
			Err(e) => return ort_sys::stub::Error::into_sys(e)
		}
	};
}
pub(crate) use trys;

pub(crate) unsafe fn str_from_ptr<'s>(ptr: *const c_char) -> Result<&'s str, Error> {
	if ptr.is_null() {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "string ptr should not be null"));
	}
	unsafe { CStr::from_ptr(ptr) }
		.to_str()
		.map_err(|_| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "string is not valid UTF-8"))
}

pub(crate) unsafe fn read_vec<T: Clone>(ptr: *const T, len: usize) -> Vec<T> {
	if len == 0 { Vec::new() } else { unsafe { slice::from_raw_parts(ptr, len) }.to_vec() }
}
