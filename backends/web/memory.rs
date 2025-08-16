use alloc::ffi::CString;
use core::{
	ffi::{CStr, c_void},
	ptr
};

use crate::binding;

#[repr(C)]
pub struct Allocator {
	_sys_api: ort_sys::OrtAllocator
}

impl Allocator {
	pub const fn new() -> Self {
		Self {
			_sys_api: ort_sys::OrtAllocator {
				version: ort_sys::ORT_API_VERSION,
				Alloc: Some(sys_allocator_alloc),
				Free: Some(sys_allocator_free),
				Info: Some(sys_allocator_info),
				Reserve: Some(sys_allocator_reserve)
			}
		}
	}
}

pub static DEFAULT_CPU_ALLOCATOR: Allocator = Allocator::new();

unsafe extern "system" fn sys_allocator_alloc(_this: *mut ort_sys::OrtAllocator, _size: usize) -> *mut c_void {
	ptr::null_mut()
}

unsafe extern "system" fn sys_allocator_free(_this: *mut ort_sys::OrtAllocator, p: *mut c_void) {
	drop(unsafe { CString::from_raw(p.cast()) });
}

unsafe extern "system" fn sys_allocator_info(this_: *const ort_sys::OrtAllocator) -> *const ort_sys::OrtMemoryInfo {
	let _allocator = unsafe { &*this_.cast::<Allocator>() };
	ptr::dangling()
}

unsafe extern "system" fn sys_allocator_reserve(_this: *const ort_sys::OrtAllocator, _size: usize) -> *mut c_void {
	ptr::null_mut()
}

#[derive(Clone, PartialEq, Eq)]
pub struct MemoryInfo {
	pub location: binding::DataLocation
}

impl MemoryInfo {
	pub fn location_exposed(&self) -> Option<&'static CStr> {
		match self.location {
			binding::DataLocation::Cpu | binding::DataLocation::CpuPinned => Some(c"Cpu"),
			binding::DataLocation::Texture => Some(c"WebGL"),
			binding::DataLocation::GpuBuffer => Some(c"WebGPU_Buffer"),
			binding::DataLocation::MlTensor => Some(c"WebNN"),
			_ => None
		}
	}

	pub fn from_location(location: &str) -> Option<Self> {
		match location {
			"Cpu" => Some(Self {
				location: binding::DataLocation::CpuPinned
			}),
			_ => None
		}
	}
}
