use std::{
	ffi::{c_char, c_int, CString},
	ptr::NonNull,
	sync::Arc
};

use super::{
	error::{Error, Result},
	ortsys
};
use crate::{char_p_to_string, error::status_to_result, Session, SharedSessionInner};

/// A device allocator used to manage the allocation of [`crate::Value`]s.
///
/// # Direct allocation
/// [`Allocator`] can be used to directly allocate device memory. This can be useful if you have a
/// postprocessing step that runs on the GPU.
/// ```no_run
/// # use ort::{Allocator, Session, Tensor, MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
/// # fn main() -> ort::Result<()> {
/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let allocator = Allocator::new(
/// 	&session,
/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
/// )?;
///
/// let mut tensor = Tensor::<f32>::new(&allocator, [1, 3, 224, 224])?;
/// // Here, `data_ptr` is a pointer to **device memory** inaccessible to the CPU; we'll need another crate, like
/// // `cudarc`, to access it.
/// let data_ptr = tensor.data_ptr_mut()?;
/// # Ok(())
/// # }
/// ```
///
/// Note that `ort` does not facilitate the transfer of data between host & device outside of session inputs &
/// outputs; you'll need to use a separate crate for that, like [`cudarc`](https://crates.io/crates/cudarc) for CUDA.
///
/// # Pinned allocation
/// Memory allocated on the host CPU is often *pageable* and may reside on the disk (swap memory). Transferring
/// pageable memory to another device is slow because the device has to go through the CPU to access the
/// memory. Many execution providers thus provide a "pinned" allocator type, which allocates *unpaged* CPU memory
/// that the device is able to access directly, bypassing the CPU and allowing for faster host-to-device data
/// transfer.
///
/// If you create a session with a device allocator that supports pinned memory, like CUDA or ROCm, you can create
/// an allocator for that session, and use it to allocate tensors with faster pinned memory:
/// ```no_run
/// # use ort::{Allocator, Session, Tensor, MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
/// # fn main() -> ort::Result<()> {
/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let allocator = Allocator::new(
/// 	&session,
/// 	MemoryInfo::new(AllocationDevice::CUDAPinned, 0, AllocatorType::Device, MemoryType::CPUInput)?
/// )?;
///
/// // Create a tensor with our pinned allocator.
/// let mut tensor = Tensor::<f32>::new(&allocator, [1, 3, 224, 224])?;
/// let data = tensor.extract_tensor_mut();
/// // ...fill `data` with data...
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Allocator {
	pub(crate) ptr: NonNull<ort_sys::OrtAllocator>,
	/// The 'default' CPU allocator, provided by `GetAllocatorWithDefaultOptions` and implemented by
	/// [`Allocator::default`], should **not** be released, so this field marks whether or not we should call
	/// `ReleaseAllocator` on drop.
	is_default: bool,
	_info: Option<MemoryInfo>,
	/// Hold a reference to the session if this allocator is tied to one.
	_session_inner: Option<Arc<SharedSessionInner>>
}

impl Allocator {
	pub(crate) unsafe fn from_raw_unchecked(ptr: *mut ort_sys::OrtAllocator) -> Allocator {
		Allocator {
			ptr: NonNull::new_unchecked(ptr),
			is_default: false,
			// currently, this function is only ever used in session creation, where we call `CreateAllocator` manually and store the allocator resulting from
			// this function in the `SharedSessionInner` - we don't need to hold onto the session, because the session is holding onto us.
			_session_inner: None,
			_info: None
		}
	}

	/// Frees an object allocated by this allocator, given the object's C pointer.
	pub(crate) unsafe fn free<T>(&self, ptr: *mut T) {
		self.ptr.as_ref().Free.unwrap_or_else(|| unreachable!("Allocator method `Free` is null"))(self.ptr.as_ptr(), ptr.cast());
	}

	/// Creates a new [`Allocator`] for the given session, to allocate memory on the device described in the
	/// [`MemoryInfo`].
	pub fn new(session: &Session, memory_info: MemoryInfo) -> Result<Self> {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe CreateAllocator(session.ptr(), memory_info.ptr.as_ptr(), &mut allocator_ptr) -> Error::CreateAllocator; nonNull(allocator_ptr)];
		Ok(Self {
			ptr: unsafe { NonNull::new_unchecked(allocator_ptr) },
			is_default: false,
			_session_inner: Some(session.inner()),
			_info: Some(memory_info)
		})
	}
}

impl Default for Allocator {
	/// Returns the default CPU allocator; equivalent to `MemoryInfo::new(AllocationDevice::CPU, 0,
	/// AllocatorType::Device, MemoryType::Default)`.
	///
	/// The allocator returned by this function is actually shared across all invocations (though this behavior is
	/// transparent to the user).
	fn default() -> Self {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		status_to_result(ortsys![unsafe GetAllocatorWithDefaultOptions(&mut allocator_ptr); nonNull(allocator_ptr)]).expect("Failed to get default allocator");
		Self {
			ptr: unsafe { NonNull::new_unchecked(allocator_ptr) },
			is_default: true,
			// The default allocator isn't tied to a session.
			_session_inner: None,
			_info: None
		}
	}
}

impl Drop for Allocator {
	fn drop(&mut self) {
		if !self.is_default {
			ortsys![unsafe ReleaseAllocator(self.ptr.as_ptr())];
		}
	}
}

/// Represents possible devices that have their own device allocator.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AllocationDevice {
	// https://github.com/microsoft/onnxruntime/blob/v1.18.0/include/onnxruntime/core/framework/allocator.h#L43-L53
	// ort will likely never support WebGPU, so I think it's best to leave `WebGPU_Buffer` out entirely to reduce confusion
	CPU,
	CUDA,
	CUDAPinned,
	CANN,
	CANNPinned,
	DirectML,
	HIP,
	HIPPinned,
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
			Self::HIP => "Hip",
			Self::HIPPinned => "HipPinned",
			Self::OpenVINOCPU => "OpenVINO_CPU",
			Self::OpenVINOGPU => "OpenVINO_GPU"
		}
	}

	/// Returns `true` if this memory is accessible by the CPU; meaning that, if a value were allocated on this device,
	/// it could be extracted to an `ndarray` or slice.
	pub fn is_cpu_accessible(&self) -> bool {
		matches!(self, Self::CPU | Self::CUDAPinned | Self::CANNPinned | Self::HIPPinned | Self::OpenVINOCPU)
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
			_ => Err(value)
		}
	}
}

/// Execution provider allocator type.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
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
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryType {
	/// Any CPU memory used by non-CPU execution provider.
	CPUInput,
	/// CPU-accessible memory output by a non-CPU execution provider, i.e. [`AllocatorDevice::CUDAPinned`].
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

/// Structure describing a memory location - the device on which the memory resides, the type of allocator (device
/// default, or arena) used, and the type of memory allocated (device-only, or CPU accessible).
///
/// `MemoryInfo` is used in the creation of [`Session`]s, [`Allocator`]s, and [`crate::Value`]s to describe on which
/// device value data should reside, and how that data should be accessible with regard to the CPU (if a non-CPU device
/// is requested).
#[derive(Debug)]
pub struct MemoryInfo {
	pub(crate) ptr: NonNull<ort_sys::OrtMemoryInfo>,
	should_release: bool
}

impl MemoryInfo {
	/// Creates a [`MemoryInfo`], describing a memory location on a device allocator.
	///
	/// # Examples
	/// `MemoryInfo` can be used to specify the device & memory type used by an [`Allocator`] to allocate tensors.
	/// See [`Allocator`] for more information & potential applications.
	/// ```no_run
	/// # use ort::{Allocator, Session, Tensor, MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
	/// )?;
	///
	/// let mut tensor = Tensor::<f32>::new(&allocator, [1, 3, 224, 224])?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(allocation_device: AllocationDevice, device_id: c_int, allocator_type: AllocatorType, memory_type: MemoryType) -> Result<Self> {
		let mut memory_info_ptr: *mut ort_sys::OrtMemoryInfo = std::ptr::null_mut();
		let allocator_name = CString::new(allocation_device.as_str()).unwrap_or_else(|_| unreachable!());
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

	pub(crate) fn from_raw(ptr: NonNull<ort_sys::OrtMemoryInfo>, should_release: bool) -> Self {
		MemoryInfo { ptr, should_release }
	}

	/// Returns the [`MemoryType`] described by this struct.
	/// ```
	/// # use ort::{MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// let mem = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
	/// assert_eq!(mem.memory_type()?, MemoryType::Default);
	/// # Ok(())
	/// # }
	/// ```
	pub fn memory_type(&self) -> Result<MemoryType> {
		let mut raw_type: ort_sys::OrtMemType = ort_sys::OrtMemType::OrtMemTypeDefault;
		ortsys![unsafe MemoryInfoGetMemType(self.ptr.as_ptr(), &mut raw_type) -> Error::GetMemoryType];
		Ok(MemoryType::from(raw_type))
	}

	/// Returns the [`AllocatorType`] described by this struct.
	/// ```
	/// # use ort::{MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// let mem = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
	/// assert_eq!(mem.allocator_type()?, AllocatorType::Device);
	/// # Ok(())
	/// # }
	/// ```
	pub fn allocator_type(&self) -> Result<AllocatorType> {
		let mut raw_type: ort_sys::OrtAllocatorType = ort_sys::OrtAllocatorType::OrtInvalidAllocator;
		ortsys![unsafe MemoryInfoGetType(self.ptr.as_ptr(), &mut raw_type) -> Error::GetAllocatorType];
		Ok(match raw_type {
			ort_sys::OrtAllocatorType::OrtArenaAllocator => AllocatorType::Arena,
			ort_sys::OrtAllocatorType::OrtDeviceAllocator => AllocatorType::Device,
			_ => unreachable!()
		})
	}

	/// Returns the [`AllocationDevice`] this struct was created with.
	/// ```
	/// # use ort::{MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// let mem = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
	/// assert_eq!(mem.allocation_device()?, AllocationDevice::CPU);
	/// # Ok(())
	/// # }
	/// ```
	pub fn allocation_device(&self) -> Result<AllocationDevice> {
		let mut name_ptr: *const c_char = std::ptr::null_mut();
		ortsys![unsafe MemoryInfoGetName(self.ptr.as_ptr(), &mut name_ptr) -> Error::GetAllocationDevice; nonNull(name_ptr)];
		// no need to free: "Do NOT free the returned pointer. It is valid for the lifetime of the OrtMemoryInfo"

		let name: String = char_p_to_string(name_ptr)?;
		AllocationDevice::try_from(name).map_err(Error::UnknownAllocationDevice)
	}

	/// Returns the ID of the [`AllocationDevice`] described by this struct.
	/// ```
	/// # use ort::{MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// let mem = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
	/// assert_eq!(mem.device_id()?, 0);
	/// # Ok(())
	/// # }
	/// ```
	pub fn device_id(&self) -> Result<i32> {
		let mut raw: ort_sys::c_int = 0;
		ortsys![unsafe MemoryInfoGetId(self.ptr.as_ptr(), &mut raw) -> Error::GetDeviceId];
		Ok(raw as _)
	}
}

impl Drop for MemoryInfo {
	fn drop(&mut self) {
		if self.should_release {
			ortsys![unsafe ReleaseMemoryInfo(self.ptr.as_ptr())];
		}
	}
}
