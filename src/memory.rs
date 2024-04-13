use std::{
	ffi::{c_char, c_int, CString},
	ptr::NonNull
};

use super::{
	error::{Error, Result},
	ortsys
};
use crate::{char_p_to_string, error::status_to_result, Session};

/// An ONNX Runtime allocator, used to manage the allocation of [`crate::Value`]s.
#[derive(Debug)]
pub struct Allocator {
	pub(crate) ptr: NonNull<ort_sys::OrtAllocator>,
	is_default: bool,
	_info: Option<MemoryInfo>
}

impl Allocator {
	pub(crate) unsafe fn from_raw_unchecked(ptr: *mut ort_sys::OrtAllocator) -> Allocator {
		Allocator {
			ptr: NonNull::new_unchecked(ptr),
			is_default: false,
			_info: None
		}
	}

	pub(crate) unsafe fn free<T>(&self, ptr: *mut T) {
		self.ptr.as_ref().Free.unwrap_unchecked()(self.ptr.as_ptr(), ptr.cast());
	}

	/// Creates a new [`Allocator`] for the given session, to allocate memory on the device described in the
	/// [`MemoryInfo`].
	///
	/// For example, to create an allocator to allocate pinned memory for CUDA:
	/// ```no_run
	/// # use ort::{Allocator, Session, MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDAPinned, 0, AllocatorType::Device, MemoryType::CPUInput)?
	/// )?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(session: &Session, memory_info: MemoryInfo) -> Result<Self> {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe CreateAllocator(session.ptr(), memory_info.ptr.as_ptr(), &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(allocator_ptr) },
			is_default: false,
			_info: Some(memory_info)
		})
	}
}

impl Default for Allocator {
	fn default() -> Self {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		status_to_result(ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr); nonNull(allocator_ptr)]).unwrap();
		Self {
			ptr: unsafe { NonNull::new_unchecked(allocator_ptr) },
			is_default: true,
			_info: None
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

/// Execution provider allocator type.
#[derive(Debug, Copy, Clone)]
pub enum AllocatorType {
	/// Default device-specific allocator.
	Device,
	/// Arena allocator.
	Arena
}

impl From<AllocatorType> for ort_sys::OrtAllocatorType {
	fn from(val: AllocatorType) -> Self {
		match val {
			AllocatorType::Device => ort_sys::OrtAllocatorType::OrtDeviceAllocator,
			AllocatorType::Arena => ort_sys::OrtAllocatorType::OrtArenaAllocator
		}
	}
}

/// Memory types for allocated memory.
#[derive(Default, Debug, Copy, Clone)]
pub enum MemoryType {
	/// Any CPU memory used by non-CPU execution provider.
	CPUInput,
	/// CPU accessible memory outputted by non-CPU execution provider, i.e. CUDA_PINNED.
	CPUOutput,
	/// The default allocator for an execution provider.
	#[default]
	Default
}

impl MemoryType {
	/// Temporary CPU accessible memory allocated by non-CPU execution provider, i.e. `CUDA_PINNED`.
	pub const CPU: MemoryType = MemoryType::CPUOutput;
}

impl From<MemoryType> for ort_sys::OrtMemType {
	fn from(val: MemoryType) -> Self {
		match val {
			MemoryType::CPUInput => ort_sys::OrtMemType::OrtMemTypeCPUInput,
			MemoryType::CPUOutput => ort_sys::OrtMemType::OrtMemTypeCPUOutput,
			MemoryType::Default => ort_sys::OrtMemType::OrtMemTypeDefault
		}
	}
}

impl From<ort_sys::OrtMemType> for MemoryType {
	fn from(value: ort_sys::OrtMemType) -> Self {
		match value {
			ort_sys::OrtMemType::OrtMemTypeCPUInput => MemoryType::CPUInput,
			ort_sys::OrtMemType::OrtMemTypeCPUOutput => MemoryType::CPUOutput,
			ort_sys::OrtMemType::OrtMemTypeDefault => MemoryType::Default
		}
	}
}

#[derive(Debug)]
pub struct MemoryInfo {
	pub(crate) ptr: NonNull<ort_sys::OrtMemoryInfo>,
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
			should_release: true
		})
	}

	/// Returns the [`MemoryType`] described by this struct.
	pub fn memory_type(&self) -> Result<MemoryType> {
		let mut raw_type: ort_sys::OrtMemType = ort_sys::OrtMemType::OrtMemTypeDefault;
		ortsys![unsafe MemoryInfoGetMemType(self.ptr.as_ptr(), &mut raw_type) -> Error::GetMemoryType];
		Ok(MemoryType::from(raw_type))
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
