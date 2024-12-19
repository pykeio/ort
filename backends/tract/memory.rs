use std::{ffi::CString, ptr};

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

unsafe extern "system" fn sys_allocator_alloc(_this: *mut ort_sys::OrtAllocator, _size: usize) -> *mut ::std::os::raw::c_void {
	ptr::null_mut()
}

unsafe extern "system" fn sys_allocator_free(_this: *mut ort_sys::OrtAllocator, p: *mut ::std::os::raw::c_void) {
	drop(CString::from_raw(p.cast()));
}

unsafe extern "system" fn sys_allocator_info(this_: *const ort_sys::OrtAllocator) -> *const ort_sys::OrtMemoryInfo {
	let _allocator = unsafe { &*this_.cast::<Allocator>() };
	ptr::dangling()
}

unsafe extern "system" fn sys_allocator_reserve(_this: *const ort_sys::OrtAllocator, _size: usize) -> *mut ::std::os::raw::c_void {
	ptr::null_mut()
}
