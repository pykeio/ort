use std::ffi::{c_char, c_int, CStr, CString};

use super::{error::OrtResult, ortsys, sys, AllocatorType, MemType};
use crate::OrtError;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AllocationDevice {
	CPU,
	CUDA,
	CUDAPinned,
	CANN,
	CANNPinned,
	DirectML,
	OpenVINOCPU,
	OpenVINOGPU
}

impl AllocationDevice {
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::CPU => "Cpu",
			Self::CUDA => "Cuda",
			Self::CUDAPinned => "CudaPinned",
			Self::CANN => "Cann",
			Self::CANNPinned => "CannPinned",
			Self::DirectML => "Dml",
			Self::OpenVINOCPU => "OpenVINO_CPU",
			Self::OpenVINOGPU => "OpenVINO_GPU"
		}
	}
}

impl From<&str> for AllocationDevice {
	fn from(value: &str) -> Self {
		match value {
			"Cpu" => AllocationDevice::CPU,
			"CUDA_CPU" => AllocationDevice::CPU,
			"Cuda" => AllocationDevice::CUDA,
			"CudaPinned" => AllocationDevice::CUDAPinned,
			"Cann" => AllocationDevice::CANN,
			"CannPinned" => AllocationDevice::CANNPinned,
			"Dml" => AllocationDevice::DirectML,
			"OpenVINO_CPU" => AllocationDevice::OpenVINOCPU,
			"OpenVINO_GPU" => AllocationDevice::OpenVINOGPU,
			other => unimplemented!("not implemented `{other}`")
		}
	}
}

#[derive(Debug)]
pub struct MemoryInfo {
	pub(crate) ptr: *mut sys::OrtMemoryInfo,
	pub(crate) should_release: bool
}

impl MemoryInfo {
	#[tracing::instrument]
	pub fn new_cpu(allocator: AllocatorType, memory_type: MemType) -> OrtResult<Self> {
		let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
		ortsys![
			unsafe CreateCpuMemoryInfo(allocator.into(), memory_type.into(), &mut memory_info_ptr);
			nonNull(memory_info_ptr)
		];
		Ok(Self {
			ptr: memory_info_ptr,
			should_release: true
		})
	}

	#[tracing::instrument]
	pub fn new(allocation_device: AllocationDevice, device_id: c_int, allocator_type: AllocatorType, memory_type: MemType) -> OrtResult<Self> {
		let mut memory_info_ptr: *mut sys::OrtMemoryInfo = std::ptr::null_mut();
		let allocator_name = CString::new(allocation_device.as_str()).unwrap();
		ortsys![
			unsafe CreateMemoryInfo(allocator_name.as_ptr(), allocator_type.into(), device_id, memory_type.into(), &mut memory_info_ptr);
			nonNull(memory_info_ptr)
		];
		Ok(Self {
			ptr: memory_info_ptr,
			should_release: true
		})
	}

	#[allow(clippy::not_unsafe_ptr_arg_deref)]
	pub fn allocation_device(memory_info_ptr: *const sys::OrtMemoryInfo) -> OrtResult<AllocationDevice> {
		let mut name_ptr: *const c_char = std::ptr::null_mut();
		ortsys![unsafe MemoryInfoGetName(memory_info_ptr, &mut name_ptr) -> OrtError::CreateCpuMemoryInfo; nonNull(name_ptr)];
		let name: String = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().to_string();
		Ok(AllocationDevice::from(name.as_str()))
	}
}

impl Drop for MemoryInfo {
	#[tracing::instrument]
	fn drop(&mut self) {
		if !self.ptr.is_null() && self.should_release {
			ortsys![unsafe ReleaseMemoryInfo(self.ptr)];
		}

		self.ptr = std::ptr::null_mut();
	}
}

#[cfg(test)]
mod tests {
	use test_log::test;

	use super::*;

	#[test]
	fn create_memory_info() {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Device, MemType::Default).unwrap();
		std::mem::drop(memory_info);
	}
}
