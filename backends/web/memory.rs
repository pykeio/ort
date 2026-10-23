use alloc::alloc::{Layout, alloc, alloc_zeroed, dealloc};
use core::{
	ffi::{self, CStr, c_void},
	ptr, slice
};

use ort_sys::{OrtAllocatorType, OrtErrorCode, OrtMemType, stub::Error};

use crate::binding;

const ALIGN: usize = 16; // max align is 8 (f64/i64/u64), + 8 for usize length prefix

fn aligned_layout(size: usize) -> Option<Layout> {
	Layout::from_size_align(size.checked_add(ALIGN)?, ALIGN).ok()
}

fn alloc_host(size: usize, zeroed: bool) -> *mut u8 {
	let Some(layout) = aligned_layout(size) else {
		return ptr::null_mut();
	};
	let ptr = unsafe { if zeroed { alloc_zeroed(layout) } else { alloc(layout) } };
	if ptr.is_null() {
		return ptr::null_mut();
	}
	unsafe {
		let ptr = ptr.add(ALIGN);
		ptr.cast::<usize>().sub(1).write(size);
		ptr
	}
}

unsafe fn free_host(p: *mut u8) {
	if p.is_null() {
		return;
	}
	unsafe {
		let size = p.cast::<usize>().sub(1).read();
		dealloc(p.sub(ALIGN), aligned_layout(size).unwrap_unchecked());
	}
}

pub(crate) struct HostBuffer {
	ptr: *mut u8,
	len: usize
}

impl HostBuffer {
	pub(crate) fn new_zeroed(len: usize) -> Result<Self, Error> {
		let ptr = alloc_host(len, true);
		if ptr.is_null() {
			// probably a bad idea to format! on alloc failure but Error allocs anyway sooo
			return Err(Error::new(OrtErrorCode::ORT_RUNTIME_EXCEPTION, format!("allocation of {len} bytes failed")));
		}
		Ok(Self { ptr, len })
	}

	pub(crate) fn as_ptr(&self) -> *mut u8 {
		self.ptr
	}

	pub(crate) fn as_slice(&self) -> &[u8] {
		unsafe { slice::from_raw_parts(self.ptr, self.len) }
	}

	pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
		unsafe { slice::from_raw_parts_mut(self.ptr, self.len) }
	}
}

impl Drop for HostBuffer {
	fn drop(&mut self) {
		unsafe { free_host(self.ptr) };
	}
}

#[repr(C)]
pub(crate) struct Allocator {
	_sys_api: ort_sys::OrtAllocator,
	pub(crate) memory_info: MemoryInfo
}

impl Allocator {
	pub(crate) const fn new(memory_info: MemoryInfo) -> Self {
		Self {
			_sys_api: ort_sys::OrtAllocator {
				version: ort_sys::ORT_API_VERSION,
				Alloc: Some(sys_allocator_alloc),
				Free: Some(sys_allocator_free),
				Info: Some(sys_allocator_info),
				Reserve: Some(sys_allocator_reserve),
				..unsafe { ort_sys::OrtAllocator::default() }
			},
			memory_info
		}
	}
}

pub(crate) static DEFAULT_CPU_MEMORY_INFO: MemoryInfo = MemoryInfo::cpu();
pub(crate) static DEFAULT_CPU_ALLOCATOR: Allocator = Allocator::new(DEFAULT_CPU_MEMORY_INFO);

unsafe extern "system" fn sys_allocator_alloc(_this: *mut ort_sys::OrtAllocator, size: usize) -> *mut c_void {
	alloc_host(size, false).cast()
}

unsafe extern "system" fn sys_allocator_free(_this: *mut ort_sys::OrtAllocator, p: *mut c_void) {
	unsafe { free_host(p.cast()) };
}

unsafe extern "system" fn sys_allocator_info(this_: *const ort_sys::OrtAllocator) -> *const ort_sys::OrtMemoryInfo {
	let allocator = unsafe { &*this_.cast::<Allocator>() };
	(&allocator.memory_info as *const MemoryInfo).cast()
}

unsafe extern "system" fn sys_allocator_reserve(this_: *const ort_sys::OrtAllocator, size: usize) -> *mut c_void {
	unsafe { sys_allocator_alloc(this_.cast_mut(), size) }
}

pub(crate) unsafe fn ort_alloc(allocator: *mut ort_sys::OrtAllocator, size: usize) -> Result<*mut u8, Error> {
	if allocator.is_null() {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "allocator cannot be null"));
	}

	let Some(alloc_fn) = (unsafe { (*allocator).Alloc }) else {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "invalid allocator"));
	};

	let ptr = unsafe { alloc_fn(allocator, size) };
	if ptr.is_null() {
		return Err(Error::new(OrtErrorCode::ORT_RUNTIME_EXCEPTION, format!("allocation of {size} bytes failed")));
	}
	Ok(ptr.cast())
}

pub(crate) unsafe fn ort_alloc_cstring(allocator: *mut ort_sys::OrtAllocator, s: &str) -> Result<*mut ffi::c_char, Error> {
	let ptr = unsafe { ort_alloc(allocator, s.len() + 1)? };
	unsafe {
		ptr::copy_nonoverlapping(s.as_ptr(), ptr, s.len());
		*ptr.add(s.len()) = 0;
	};
	Ok(ptr.cast())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MemoryInfo {
	pub(crate) location: binding::DataLocation,
	pub(crate) allocator_type: OrtAllocatorType,
	pub(crate) mem_type: OrtMemType
}

impl MemoryInfo {
	pub(crate) const fn from_location(location: binding::DataLocation) -> Self {
		Self {
			location,
			allocator_type: OrtAllocatorType::OrtDeviceAllocator,
			mem_type: OrtMemType::OrtMemTypeDefault
		}
	}

	pub(crate) const fn cpu() -> Self {
		Self::from_location(binding::DataLocation::Cpu)
	}

	pub(crate) const fn is_cpu(&self) -> bool {
		matches!(self.location, binding::DataLocation::Cpu | binding::DataLocation::CpuPinned)
	}

	pub(crate) fn location_exposed(&self) -> Option<&'static CStr> {
		match self.location {
			binding::DataLocation::Cpu | binding::DataLocation::CpuPinned => Some(c"Cpu"),
			binding::DataLocation::Texture => Some(c"WebGL"),
			binding::DataLocation::GpuBuffer => Some(c"WebGPU_Buffer"),
			binding::DataLocation::MlTensor => Some(c"WebNN"),
			_ => None
		}
	}
}

pub(crate) fn location_from_device_name(name: &str) -> Option<binding::DataLocation> {
	match name {
		"Cpu" => Some(binding::DataLocation::Cpu),
		"WebGL" => Some(binding::DataLocation::Texture),
		"WebGPU_Buffer" => Some(binding::DataLocation::GpuBuffer),
		"WebNN" => Some(binding::DataLocation::MlTensor),
		_ => None
	}
}
