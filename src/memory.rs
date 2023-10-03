use std::ffi::{c_char, c_int, CString};

use super::{
	error::{OrtError, OrtResult},
	ortsys, sys, AllocatorType, MemType
};
use crate::{char_p_to_string, error::status_to_result};

/// An ONNX Runtime allocator, used to manage the allocation of [`crate::Value`]s.
#[derive(Debug)]
pub struct Allocator {
	pub(crate) ptr: *mut sys::OrtAllocator,
	is_default: bool
}

impl Default for Allocator {
	fn default() -> Self {
		let mut allocator_ptr: *mut sys::OrtAllocator = std::ptr::null_mut();
		status_to_result(ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr); nonNull(allocator_ptr)]).unwrap();
		Self { ptr: allocator_ptr, is_default: true }
	}
}

impl Drop for Allocator {
	fn drop(&mut self) {
		// per GetAllocatorWithDefaultOptions docs: Returned value should NOT be freed
		// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a8dec797ae52ee1a681e4f88be1fb4bb3
		if !self.is_default {
			ortsys![unsafe ReleaseAllocator(self.ptr)];
		}
	}
}

/// Represents possible devices that have their own device allocator.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AllocationDevice {
	// https://github.com/microsoft/onnxruntime/blob/v1.15.1/include/onnxruntime/core/framework/allocator.h#L36-L45
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

impl TryFrom<&str> for AllocationDevice {
	type Error = String;

	fn try_from(value: &str) -> Result<Self, String> {
		match value {
			"Cpu" => Ok(AllocationDevice::CPU),
			"CUDA_CPU" => Ok(AllocationDevice::CPU),
			"Cuda" => Ok(AllocationDevice::CUDA),
			"CudaPinned" => Ok(AllocationDevice::CUDAPinned),
			"Cann" => Ok(AllocationDevice::CANN),
			"CannPinned" => Ok(AllocationDevice::CANNPinned),
			"Dml" => Ok(AllocationDevice::DirectML),
			"OpenVINO_CPU" => Ok(AllocationDevice::OpenVINOCPU),
			"OpenVINO_GPU" => Ok(AllocationDevice::OpenVINOGPU),
			other => Err(other.to_string())
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
			unsafe CreateCpuMemoryInfo(allocator.into(), memory_type.into(), &mut memory_info_ptr) -> OrtError::CreateMemoryInfo;
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
			unsafe CreateMemoryInfo(allocator_name.as_ptr(), allocator_type.into(), device_id, memory_type.into(), &mut memory_info_ptr)
				-> OrtError::CreateMemoryInfo;
			nonNull(memory_info_ptr)
		];
		Ok(Self {
			ptr: memory_info_ptr,
			should_release: true
		})
	}

	/// Returns the [`AllocationDevice`] this memory info
	pub fn allocation_device(&self) -> OrtResult<AllocationDevice> {
		let mut name_ptr: *const c_char = std::ptr::null_mut();
		ortsys![unsafe MemoryInfoGetName(self.ptr, &mut name_ptr) -> OrtError::GetAllocationDevice; nonNull(name_ptr)];
		// no need to free: "Do NOT free the returned pointer. It is valid for the lifetime of the OrtMemoryInfo"

		let name: String = char_p_to_string(name_ptr)?;
		AllocationDevice::try_from(name.as_str()).map_err(OrtError::UnknownAllocationDevice)
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
