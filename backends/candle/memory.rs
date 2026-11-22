use std::{
	alloc::{Layout, alloc, alloc_zeroed, dealloc},
	ffi::{c_char, c_void},
	ptr
};

use candle_core::{Device, DeviceLocation};
use ort_sys::{OrtAllocatorType, OrtErrorCode, OrtMemType, OrtMemoryInfoDeviceType};

use crate::Error;

const ALIGN: usize = 64;

fn aligned_layout(size: usize) -> Option<Layout> {
	Layout::from_size_align(size.checked_add(ALIGN)?, ALIGN).ok()
}

pub(crate) fn alloc_host(size: usize, zeroed: bool) -> *mut u8 {
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

pub(crate) unsafe fn free_host(p: *mut u8) {
	if p.is_null() {
		return;
	}
	unsafe {
		let size = p.cast::<usize>().sub(1).read();
		dealloc(p.sub(ALIGN), aligned_layout(size).unwrap_unchecked());
	}
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryInfo {
	pub(crate) device: Device,
	pub(crate) allocator_type: OrtAllocatorType,
	pub(crate) memory_type: OrtMemType
}

impl MemoryInfo {
	pub(crate) const fn cpu() -> Self {
		Self {
			device: Device::Cpu,
			allocator_type: OrtAllocatorType::OrtDeviceAllocator,
			memory_type: OrtMemType::OrtMemTypeDefault
		}
	}

	pub(crate) fn from_device(device: &Device) -> Self {
		Self {
			device: device.clone(),
			allocator_type: OrtAllocatorType::OrtDeviceAllocator,
			memory_type: OrtMemType::OrtMemTypeDefault
		}
	}

	pub(crate) fn new(device_name: impl AsRef<str>, device_id: usize, allocator_type: OrtAllocatorType, memory_type: OrtMemType) -> Result<Self, Error> {
		let device = match device_name.as_ref() {
			"Cpu" | "CudaPinned" => Device::Cpu,
			"Cuda" => match memory_type {
				OrtMemType::OrtMemTypeCPUInput | OrtMemType::OrtMemTypeCPUOutput => Device::Cpu,
				OrtMemType::OrtMemTypeDefault => Device::new_cuda(device_id).map_err(|e| Error::new(OrtErrorCode::ORT_ENGINE_ERROR, e.to_string()))?
			},
			"Metal" => Device::new_metal(device_id).map_err(|e| Error::new(OrtErrorCode::ORT_ENGINE_ERROR, e.to_string()))?,
			device_name => return Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("ort-candle does not support the '{device_name}' device")))
		};
		Ok(Self { device, allocator_type, memory_type })
	}

	pub(crate) fn is_cpu(&self) -> bool {
		self.device.is_cpu()
	}

	pub(crate) fn device_type(&self) -> OrtMemoryInfoDeviceType {
		match &self.device {
			Device::Cpu => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
			Device::Cuda(_) | Device::Metal(_) => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU
		}
	}

	pub(crate) fn device_name_sys(&self) -> *const c_char {
		match &self.device {
			Device::Cpu => c"Cpu".as_ptr(),
			Device::Cuda(_) => c"Cuda".as_ptr(),
			Device::Metal(_) => c"Metal".as_ptr()
		}
	}

	pub(crate) fn device_id(&self) -> usize {
		match self.device.location() {
			DeviceLocation::Cpu => 0,
			DeviceLocation::Cuda { gpu_id } => gpu_id,
			DeviceLocation::Metal { gpu_id } => gpu_id
		}
	}
}

impl PartialEq for MemoryInfo {
	fn eq(&self, other: &Self) -> bool {
		self.device.same_device(&other.device) && self.allocator_type == other.allocator_type && self.memory_type == other.memory_type
	}
}

#[repr(C)]
pub(crate) struct Allocator {
	_sys_api: ort_sys::OrtAllocator,
	pub memory_info: MemoryInfo
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
pub(crate) static DEFAULT_CPU_ALLOCATOR: Allocator = Allocator::new(MemoryInfo::cpu());

unsafe extern "system" fn sys_allocator_alloc(this_: *mut ort_sys::OrtAllocator, size: usize) -> *mut c_void {
	let allocator = unsafe { &*this_.cast::<Allocator>() };
	if !allocator.memory_info.is_cpu() {
		return ptr::null_mut();
	}
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
	};

	let Some(alloc_fn) = (unsafe { (*allocator).Alloc }) else {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "invalid allocator"));
	};

	let ptr = unsafe { alloc_fn(allocator, size) };
	if ptr.is_null() {
		// again probably a bad idea to format! on alloc failure but Error allocs anyway
		return Err(Error::new(OrtErrorCode::ORT_RUNTIME_EXCEPTION, format!("allocation of {size} bytes failed")));
	}
	Ok(ptr.cast())
}

pub(crate) unsafe fn ort_alloc_cstring(allocator: *mut ort_sys::OrtAllocator, s: &str) -> Result<*mut c_char, Error> {
	let ptr = unsafe { ort_alloc(allocator, s.len() + 1) }?;
	unsafe {
		ptr::copy_nonoverlapping(s.as_ptr(), ptr, s.len());
		*ptr.add(s.len()) = 0;
	}
	Ok(ptr.cast())
}
