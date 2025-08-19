use alloc::boxed::Box;

use crate::binding;

pub(crate) struct Environment {
	pub with_telemetry: bool
}

impl Environment {
	pub fn new_sys() -> *mut ort_sys::OrtEnv {
		(Box::leak(Box::new(Self { with_telemetry: true })) as *mut Environment).cast()
	}

	pub unsafe fn cast_from_sys<'e>(ptr: *const ort_sys::OrtEnv) -> &'e Environment {
		unsafe { &*ptr.cast::<Environment>() }
	}

	pub unsafe fn cast_from_sys_mut<'e>(ptr: *mut ort_sys::OrtEnv) -> &'e mut Environment {
		unsafe { &mut *ptr.cast::<Environment>() }
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtEnv) -> Box<Environment> {
		unsafe { Box::from_raw(ptr.cast::<Environment>()) }
	}

	pub fn send_telemetry_event(&self, event: TelemetryEvent) {
		if !self.with_telemetry {
			return;
		}

		let _ = match event {
			TelemetryEvent::SessionInit => binding::track_session_init()
		};
	}
}

#[derive(Debug)]
pub enum TelemetryEvent {
	SessionInit
}
