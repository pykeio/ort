use std::{
	ffi::{c_char, c_int, c_void, CString},
	mem,
	ptr::NonNull,
	sync::Arc
};

use crate::{
	error::{status_to_result, Result},
	ortsys,
	session::{Session, SharedSessionInner}
};

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
/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
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

	/// Allocates a block of memory, of size `size_of::<T>() * len` bytes, using this allocator.
	/// The memory will be automatically freed when the returned `AllocatedBlock` goes out of scope.
	///
	/// May return `None` if the allocation fails.
	///
	/// # Example
	/// ```
	/// # use ort::Allocator;
	/// let allocator = Allocator::default();
	/// let mut mem = allocator.alloc::<i32>(5).unwrap();
	/// unsafe {
	/// 	let ptr = mem.as_mut_ptr().cast::<i32>();
	/// 	*ptr.add(3) = 42;
	/// };
	/// ```
	pub fn alloc<T>(&self, len: usize) -> Option<AllocatedBlock<'_>> {
		let ptr = unsafe {
			self.ptr
				.as_ref()
				.Alloc
				.unwrap_or_else(|| unreachable!("Allocator method `Alloc` is null"))(self.ptr.as_ptr(), (len * mem::size_of::<T>()) as _)
		};
		if !ptr.is_null() { Some(AllocatedBlock { ptr, allocator: self }) } else { None }
	}

	/// Frees an object allocated by this allocator, given the object's C pointer.
	///
	/// # Safety
	/// The pointer **must** have been allocated using this exact allocator's [`alloc`](Allocator::alloc) function.
	///
	/// This function is meant to be used in situations where the lifetime restrictions of [`AllocatedBlock`] are hard
	/// to work with.
	///
	/// ```
	/// # use ort::Allocator;
	/// let allocator = Allocator::default();
	/// let mut mem = allocator.alloc::<i32>(5).unwrap();
	/// unsafe {
	/// 	let ptr = mem.as_mut_ptr().cast::<i32>();
	/// 	*ptr.add(3) = 42;
	///
	/// 	allocator.free(mem.into_raw());
	/// };
	/// ```
	pub unsafe fn free<T>(&self, ptr: *mut T) {
		self.ptr.as_ref().Free.unwrap_or_else(|| unreachable!("Allocator method `Free` is null"))(self.ptr.as_ptr(), ptr.cast());
	}

	/// Returns the [`MemoryInfo`] describing this allocator.
	pub fn memory_info(&self) -> MemoryInfo {
		let memory_info_ptr = unsafe { self.ptr.as_ref().Info.unwrap_or_else(|| unreachable!("Allocator method `Info` is null"))(self.ptr.as_ptr()) };
		MemoryInfo::from_raw(unsafe { NonNull::new_unchecked(memory_info_ptr.cast_mut()) }, false)
	}

	/// Creates a new [`Allocator`] for the given session, to allocate memory on the device described in the
	/// [`MemoryInfo`].
	pub fn new(session: &Session, memory_info: MemoryInfo) -> Result<Self> {
		let mut allocator_ptr: *mut ort_sys::OrtAllocator = std::ptr::null_mut();
		ortsys![unsafe CreateAllocator(session.ptr(), memory_info.ptr.as_ptr(), &mut allocator_ptr)?; nonNull(allocator_ptr)];
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

/// A block of memory allocated by an [`Allocator`].
pub struct AllocatedBlock<'a> {
	ptr: *mut c_void,
	allocator: &'a Allocator
}

impl<'a> AllocatedBlock<'a> {
	/// Returns a pointer to the allocated memory.
	///
	/// Note that, depending on the exact allocator used, this may not a pointer to memory accessible by the CPU.
	pub fn as_ptr(&self) -> *const c_void {
		self.ptr
	}

	/// Returns a mutable pointer to the allocated memory.
	///
	/// Note that, depending on the exact allocator used, this may not a pointer to memory accessible by the CPU.
	pub fn as_mut_ptr(&mut self) -> *mut c_void {
		self.ptr
	}

	/// Returns the [`Allocator`] that allocated this block of memory.
	pub fn allocator(&self) -> &Allocator {
		self.allocator
	}

	/// Consumes the [`AllocatedBlock`], returning the pointer to its data.
	///
	/// The pointer must be freed with [`Allocator::free`], using the allocator that initially created it. Not doing so
	/// will cause a memory leak.
	#[must_use = "the returned pointer must be freed with the allocator that created it"]
	pub fn into_raw(self) -> *mut c_void {
		let ptr = self.ptr;
		mem::forget(self);
		ptr
	}
}

impl<'a> Drop for AllocatedBlock<'a> {
	fn drop(&mut self) {
		unsafe { self.allocator.free(self.ptr) };
	}
}

/// Represents possible devices that have their own device allocator.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
// &'static str should be valid here since they're only ever defined in C++ with `const char *` literals
pub struct AllocationDevice(&'static str);

impl AllocationDevice {
	pub const CPU: AllocationDevice = AllocationDevice("Cpu");
	pub const CUDA: AllocationDevice = AllocationDevice("Cuda");
	pub const CUDA_PINNED: AllocationDevice = AllocationDevice("CudaPinned");
	pub const CANN: AllocationDevice = AllocationDevice("Cann");
	pub const CANN_PINNED: AllocationDevice = AllocationDevice("CannPinned");
	pub const DIRECTML: AllocationDevice = AllocationDevice("Dml");
	pub const DIRECTML_CPU: AllocationDevice = AllocationDevice("DML CPU");
	pub const HIP: AllocationDevice = AllocationDevice("Hip");
	pub const HIP_PINNED: AllocationDevice = AllocationDevice("HipPinned");
	pub const OPENVINO_CPU: AllocationDevice = AllocationDevice("OpenVINO_CPU");
	pub const OPENVINO_GPU: AllocationDevice = AllocationDevice("OpenVINO_GPU");
	pub const XNNPACK: AllocationDevice = AllocationDevice("XnnpackExecutionProvider");
	pub const TVM: AllocationDevice = AllocationDevice("TVM");

	pub fn as_str(&self) -> &'static str {
		self.0
	}

	/// Returns `true` if this memory is accessible by the CPU; meaning that, if a value were allocated on this device,
	/// it could be extracted to an `ndarray` or slice.
	pub fn is_cpu_accessible(&self) -> bool {
		self == &Self::CPU
			|| self == &Self::CUDA_PINNED
			|| self == &Self::CANN_PINNED
			|| self == &Self::HIP_PINNED
			|| self == &Self::OPENVINO_CPU
			|| self == &Self::DIRECTML_CPU
			|| self == &Self::XNNPACK
			|| self == &Self::TVM
	}
}

impl PartialEq<str> for AllocationDevice {
	fn eq(&self, other: &str) -> bool {
		self.0 == other
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
			unsafe CreateMemoryInfo(allocator_name.as_ptr(), allocator_type.into(), device_id, memory_type.into(), &mut memory_info_ptr)?;
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
		ortsys![unsafe MemoryInfoGetMemType(self.ptr.as_ptr(), &mut raw_type)?];
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
		ortsys![unsafe MemoryInfoGetType(self.ptr.as_ptr(), &mut raw_type)?];
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
		ortsys![unsafe MemoryInfoGetName(self.ptr.as_ptr(), &mut name_ptr)?; nonNull(name_ptr)];

		let mut len = 0;
		while unsafe { *name_ptr.add(len) } != 0x00 {
			len += 1;
		}

		// SAFETY: ONNX Runtime internally only ever defines allocation device names as ASCII. can't wait for this to blow up
		// one day regardless
		let name = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(name_ptr.cast::<u8>(), len)) };
		Ok(AllocationDevice(name))
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
		ortsys![unsafe MemoryInfoGetId(self.ptr.as_ptr(), &mut raw)?];
		Ok(raw as _)
	}
}

impl PartialEq<MemoryInfo> for MemoryInfo {
	fn eq(&self, other: &MemoryInfo) -> bool {
		let mut out = 0;
		ortsys![unsafe CompareMemoryInfo(self.ptr.as_ptr(), other.ptr.as_ptr(), &mut out)]; // implementation always returns ok status
		out == 0
	}
}

impl Drop for MemoryInfo {
	fn drop(&mut self) {
		if self.should_release {
			ortsys![unsafe ReleaseMemoryInfo(self.ptr.as_ptr())];
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};

	#[test]
	fn test_memory_info_eq() -> crate::Result<()> {
		let a = MemoryInfo::new(AllocationDevice::CUDA, 1, AllocatorType::Device, MemoryType::Default)?;
		let b = MemoryInfo::new(AllocationDevice::CUDA, 1, AllocatorType::Device, MemoryType::Default)?;
		assert_eq!(a, b);
		let c = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
		assert_ne!(a, c);
		Ok(())
	}
}
