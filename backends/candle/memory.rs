use std::{any::Any, ffi::CString, ptr};

use candle_core::{Device, DeviceLocation, backend::BackendDevice};
use ort_sys::{OrtAllocatorType, OrtErrorCode, OrtMemType, OrtMemoryInfoDeviceType};

use crate::error::Error;

#[repr(transparent)]
pub struct MemoryInfo(pub Device);

impl MemoryInfo {
	pub fn new(device_name: impl AsRef<str>, device_id: usize, mem_type: OrtMemType) -> Result<Self, Error> {
		match device_name.as_ref() {
			"Cpu" | "CudaPinned" => Ok(Self(Device::Cpu)),
			"Cuda" => match mem_type {
				OrtMemType::OrtMemTypeCPUInput | OrtMemType::OrtMemTypeCPUOutput => Ok(Self(Device::Cpu)),
				OrtMemType::OrtMemTypeDefault => Device::new_cuda(device_id)
					.map(Self)
					.map_err(|e| Error::new(OrtErrorCode::ORT_ENGINE_ERROR, e.to_string()))
			},
			"Metal" => Device::new_metal(device_id)
				.map(Self)
				.map_err(|e| Error::new(OrtErrorCode::ORT_ENGINE_ERROR, e.to_string())),
			device_name => Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("ort-candle does not support the '{device_name}' device")))
		}
	}

	pub fn device(&self) -> &Device {
		&self.0
	}

	pub fn device_type(&self) -> OrtMemoryInfoDeviceType {
		match &self.0 {
			Device::Cpu => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_CPU,
			Device::Cuda(_) | Device::Metal(_) => OrtMemoryInfoDeviceType::OrtMemoryInfoDeviceType_GPU
		}
	}

	pub fn device_name(&self) -> &'static str {
		let sys_str = self.device_name_sys();
		&sys_str[..sys_str.len() - 1]
	}

	pub fn device_name_sys(&self) -> &'static str {
		match &self.0 {
			Device::Cpu => "Cpu\0",
			Device::Cuda(_) => "Cuda\0",
			Device::Metal(_) => "Metal\0"
		}
	}

	pub fn device_id(&self) -> usize {
		match self.0.location() {
			DeviceLocation::Cpu => 0,
			DeviceLocation::Cuda { gpu_id } => gpu_id,
			DeviceLocation::Metal { gpu_id } => gpu_id
		}
	}

	pub fn memory_type(&self) -> OrtMemType {
		OrtMemType::OrtMemTypeDefault
	}
}

impl PartialEq for MemoryInfo {
	fn eq(&self, other: &Self) -> bool {
		self.0.same_device(&other.0)
	}
}

#[repr(C)]
pub struct Allocator<'m> {
	_sys_api: ort_sys::OrtAllocator,
	pub memory_info: &'m MemoryInfo
}

impl<'m> Allocator<'m> {
	pub const fn new(memory_info: &'m MemoryInfo) -> Self {
		Self {
			_sys_api: ort_sys::OrtAllocator {
				version: ort_sys::ORT_API_VERSION,
				Alloc: Some(sys_allocator_alloc),
				Free: Some(sys_allocator_free),
				Info: Some(sys_allocator_info),
				Reserve: Some(sys_allocator_reserve)
			},
			memory_info
		}
	}
}

pub static DEFAULT_CPU_ALLOCATOR: Allocator = Allocator::new(&MemoryInfo(Device::Cpu));

unsafe extern "system" fn sys_allocator_alloc(_this: *mut ort_sys::OrtAllocator, _size: usize) -> *mut ::std::os::raw::c_void {
	ptr::null_mut()
}

unsafe extern "system" fn sys_allocator_free(_this: *mut ort_sys::OrtAllocator, p: *mut ::std::os::raw::c_void) {
	drop(CString::from_raw(p.cast()));
}

unsafe extern "system" fn sys_allocator_info(this_: *const ort_sys::OrtAllocator) -> *const ort_sys::OrtMemoryInfo {
	let _allocator = unsafe { &*this_.cast::<Allocator>() };
	(_allocator.memory_info as *const MemoryInfo).cast()
}

unsafe extern "system" fn sys_allocator_reserve(_this: *const ort_sys::OrtAllocator, _size: usize) -> *mut ::std::os::raw::c_void {
	ptr::null_mut()
}
