mod api;
mod memory;
mod session;
mod tensor;
mod util;

pub(crate) use ort_sys::stub::Error;

pub use self::api::api;

pub(crate) struct Environment {}

impl Environment {
	pub fn new_sys() -> *mut ort_sys::OrtEnv {
		(Box::leak(Box::new(Self {})) as *mut Environment).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtEnv) -> Box<Environment> {
		unsafe { Box::from_raw(ptr.cast::<Environment>()) }
	}
}
