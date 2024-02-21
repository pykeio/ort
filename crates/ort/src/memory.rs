use std::{
	ffi::{c_char, c_int, CString},
	ptr::NonNull
};

use super::{
	error::{Error, Result},
	ortsys, AllocatorType, MemoryType
};
use crate::{char_p_to_string, error::status_to_result};

/// An ONNX Runtime allocator, used to manage the allocation of [`crate::Value`]s.
#[derive(Debug)]
pub struct Allocator {
	pub(crate) ptr: NonNull<ort_sys::OrtAllocator>,
	is_default: bool
}

impl Allocator {
	pub(crate) unsafe fn from_raw_unchecked(ptr: *mut ort_sys::OrtAllocator) -> Allocator {
		Allocator {
			ptr: NonNull::new_unchecked(ptr),
			is_default: false
		}
	}

	pub(crate) unsafe fn free<T>(&self, ptr: *mut T) {
		self.ptr.as_ref().Free.unwrap_unchecked()(self.ptr.as_ptr(), ptr.cast());
	}
}

impl Default for Allocator {
	fn default() -> Self {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		status_to_result(ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr); nonNull(allocator_ptr)]).unwrap();
		Self {
			ptr: unsafe { NonNull::new_unchecked(allocator_ptr) },
			is_default: true
		}
	}
}

impl Drop for Allocator {
	fn drop(&mut self) {
		// per GetAllocatorWithDefaultOptions docs: Returned value should NOT be freed
		// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a8dec797ae52ee1a681e4f88be1fb4bb3
		if !self.is_default {
			ortsys![unsafe ReleaseAllocator(self.ptr.as_ptr())];
		}
	}
}

/// Represents possible devices that have their own device allocator.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AllocationDevice {
	// https://github.com/microsoft/onnxruntime/blob/v1.17.0/include/onnxruntime/core/framework/allocator.h#L43-L53
	CPU,
	CUDA,
	CUDAPinned,
	CANN,
	CANNPinned,
	DirectML,
	HIP,
	HIPPinned,
	OpenVINOCPU,
	OpenVINOGPU,
	WebGPUBuffer
}

impl AllocationDevice {
	#[must_use]
	pub fn as_str(&self) -> &'static str {
		match self {
			Self::CPU => "Cpu",
			Self::CUDA => "Cuda",
			Self::CUDAPinned => "CudaPinned",
			Self::CANN => "Cann",
			Self::CANNPinned => "CannPinned",
			Self::DirectML => "Dml",
			Self::HIP => "Hip",
			Self::HIPPinned => "HipPinned",
			Self::OpenVINOCPU => "OpenVINO_CPU",
			Self::OpenVINOGPU => "OpenVINO_GPU",
			Self::WebGPUBuffer => "WebGPU_Buffer"
		}
	}
}

impl TryFrom<String> for AllocationDevice {
	type Error = String;

	fn try_from(value: String) -> Result<Self, String> {
		match value.as_str() {
			"Cpu" | "CUDA_CPU" => Ok(AllocationDevice::CPU),
			"Cuda" => Ok(AllocationDevice::CUDA),
			"CudaPinned" => Ok(AllocationDevice::CUDAPinned),
			"Cann" => Ok(AllocationDevice::CANN),
			"CannPinned" => Ok(AllocationDevice::CANNPinned),
			"Dml" => Ok(AllocationDevice::DirectML),
			"Hip" => Ok(AllocationDevice::HIP),
			"HipPinned" => Ok(AllocationDevice::HIPPinned),
			"OpenVINO_CPU" => Ok(AllocationDevice::OpenVINOCPU),
			"OpenVINO_GPU" => Ok(AllocationDevice::OpenVINOGPU),
			"WebGPUBuffer" => Ok(AllocationDevice::WebGPUBuffer),
			_ => Err(value)
		}
	}
}

#[derive(Debug)]
pub struct MemoryInfo {
	pub(crate) ptr: NonNull<ort_sys::OrtMemoryInfo>,
	memory_type: MemoryType,
	should_release: bool
}

impl MemoryInfo {
	#[tracing::instrument]
	pub fn new_cpu(allocator: AllocatorType, memory_type: MemoryType) -> Result<Self> {
		let mut memory_info_ptr: *mut ort_sys::OrtMemoryInfo = std::ptr::null_mut();
		ortsys![
			unsafe CreateCpuMemoryInfo(allocator.into(), memory_type.into(), &mut memory_info_ptr) -> Error::CreateMemoryInfo;
			nonNull(memory_info_ptr)
		];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(memory_info_ptr) },
			memory_type,
			should_release: true
		})
	}

	#[tracing::instrument]
	pub fn new(allocation_device: AllocationDevice, device_id: c_int, allocator_type: AllocatorType, memory_type: MemoryType) -> Result<Self> {
		let mut memory_info_ptr: *mut ort_sys::OrtMemoryInfo = std::ptr::null_mut();
		let allocator_name = CString::new(allocation_device.as_str()).unwrap();
		ortsys![
			unsafe CreateMemoryInfo(allocator_name.as_ptr(), allocator_type.into(), device_id, memory_type.into(), &mut memory_info_ptr)
				-> Error::CreateMemoryInfo;
			nonNull(memory_info_ptr)
		];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(memory_info_ptr) },
			memory_type,
			should_release: true
		})
	}

	/// Returns the [`MemoryType`] this struct was created with.
	#[must_use]
	pub fn memory_type(&self) -> MemoryType {
		self.memory_type
	}

	/// Returns the [`AllocationDevice`] this struct was created with.
	pub fn allocation_device(&self) -> Result<AllocationDevice> {
		let mut name_ptr: *const c_char = std::ptr::null_mut();
		ortsys![unsafe MemoryInfoGetName(self.ptr.as_ptr(), &mut name_ptr) -> Error::GetAllocationDevice; nonNull(name_ptr)];
		// no need to free: "Do NOT free the returned pointer. It is valid for the lifetime of the OrtMemoryInfo"

		let name: String = char_p_to_string(name_ptr)?;
		AllocationDevice::try_from(name).map_err(Error::UnknownAllocationDevice)
	}
}

impl Drop for MemoryInfo {
	#[tracing::instrument]
	fn drop(&mut self) {
		if self.should_release {
			ortsys![unsafe ReleaseMemoryInfo(self.ptr.as_ptr())];
		}
	}
}

#[cfg(test)]
mod tests {
	use test_log::test;

	use super::*;

	#[test]
	fn create_memory_info() {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Device, MemoryType::Default).unwrap();
		std::mem::drop(memory_info);
	}
}
