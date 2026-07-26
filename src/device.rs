#[cfg(feature = "api-24")]
use alloc::ffi::CString;
use core::{ffi::CStr, marker::PhantomData, ptr::NonNull};
#[cfg(feature = "api-24")]
use core::{ffi::c_char, ptr};
#[cfg(all(feature = "api-24", feature = "std"))]
use std::path::Path;

#[cfg(feature = "api-23")]
use crate::memory::MemoryInfo;
use crate::{AsPointer, Error, Result, memory::DeviceType, ortsys};
#[cfg(feature = "api-24")]
use crate::{ep::ExecutionProvider, memory::Allocator};

pub struct Device<'e> {
	ptr: NonNull<ort_sys::OrtEpDevice>,
	_p: PhantomData<&'e ()>
}

impl<'e> Device<'e> {
	pub(crate) fn new(ptr: NonNull<ort_sys::OrtEpDevice>) -> Self {
		Self { ptr, _p: PhantomData }
	}

	/// Returns the [name of the EP](crate::ep::ExecutionProvider::name) this device belongs to.
	///
	/// ```
	/// # use ort::environment::Environment;
	/// # fn main() -> ort::Result<()> {
	/// let env = Environment::current()?;
	/// let cpu = env.devices().next().unwrap();
	/// assert!(matches!(cpu.ep(), Ok("CPUExecutionProvider")));
	/// # Ok(())
	/// # }
	/// ```
	pub fn ep(&self) -> Result<&'e str> {
		let name = ortsys![unsafe EpDevice_EpName(self.ptr.as_ptr())];
		unsafe { CStr::from_ptr(name) }.to_str().map_err(Error::from)
	}

	/// Returns the name of the EP vendor this device belongs to, e.g. `"Microsoft"` for DirectML devices.
	///
	/// For the *manufacturer* of the device, see [`Device::vendor`].
	///
	/// ```
	/// # use ort::environment::Environment;
	/// # fn main() -> ort::Result<()> {
	/// let env = Environment::current()?;
	/// let cpu = env.devices().next().unwrap();
	/// assert!(matches!(cpu.ep_vendor(), Ok("Microsoft")));
	/// # Ok(())
	/// # }
	/// ```
	pub fn ep_vendor(&self) -> Result<&'e str> {
		let vendor = ortsys![unsafe EpDevice_EpVendor(self.ptr.as_ptr())];
		unsafe { CStr::from_ptr(vendor) }.to_str().map_err(Error::from)
	}

	pub fn hardware_device(&self) -> HardwareDevice<'e> {
		HardwareDevice {
			ptr: NonNull::new(ortsys![unsafe EpDevice_Device(self.ptr.as_ptr())].cast_mut()).expect("invalid device"),
			_p: PhantomData
		}
	}

	#[cfg(feature = "api-23")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-23")))]
	pub fn memory_info(&self, host_accessible: bool) -> MemoryInfo<'_> {
		MemoryInfo::from_raw(
			NonNull::new(
				ortsys![unsafe EpDevice_MemoryInfo(self.ptr.as_ptr(), if host_accessible {
					ort_sys::OrtDeviceMemoryType::OrtDeviceMemoryType_HOST_ACCESSIBLE
				} else {
					ort_sys::OrtDeviceMemoryType::OrtDeviceMemoryType_DEFAULT
				})]
				.cast_mut()
			)
			.expect("infallible"),
			false
		)
	}

	#[cfg(feature = "api-24")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-24")))]
	pub fn compatibility<E: ExecutionProvider>(&self, compatibility_info: &CompatibilityInfo<E>) -> Result<DeviceCompatibility> {
		// API expects an array of devices
		let devices = [self.ptr.as_ptr().cast_const()];
		let mut out = ort_sys::OrtCompiledModelCompatibility::EP_NOT_APPLICABLE;
		ortsys![unsafe GetModelCompatibilityForEpDevices(devices.as_ptr(), 1, compatibility_info.str.as_ptr(), &mut out)?];
		Ok(match out {
			ort_sys::OrtCompiledModelCompatibility::EP_SUPPORTED_OPTIMAL => DeviceCompatibility::SupportedOptimal,
			ort_sys::OrtCompiledModelCompatibility::EP_SUPPORTED_PREFER_RECOMPILATION => DeviceCompatibility::SupportedPreferRecompilation,
			ort_sys::OrtCompiledModelCompatibility::EP_UNSUPPORTED => DeviceCompatibility::Unsupported,
			_ => DeviceCompatibility::NotApplicable
		})
	}
}

impl AsPointer for Device<'_> {
	type Sys = ort_sys::OrtEpDevice;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

pub struct HardwareDevice<'e> {
	ptr: NonNull<ort_sys::OrtHardwareDevice>,
	_p: PhantomData<&'e ()>
}

impl<'e> HardwareDevice<'e> {
	/// Returns the [type](DeviceType) of the device - CPU, GPU, or NPU.
	///
	/// ```
	/// # use ort::{environment::Environment, memory::DeviceType};
	/// # fn main() -> ort::Result<()> {
	/// let env = Environment::current()?;
	/// let cpu = env.devices().next().unwrap().hardware_device();
	/// assert_eq!(cpu.ty(), DeviceType::CPU);
	/// # Ok(())
	/// # }
	/// ```
	pub fn ty(&self) -> DeviceType {
		match ortsys![unsafe HardwareDevice_Type(self.ptr.as_ptr())] {
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_CPU => DeviceType::CPU,
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_GPU => DeviceType::GPU,
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU => DeviceType::NPU
		}
	}

	/// Returns the device ID.
	///
	/// The ID may be arbitrary and is not guaranteed to be the same as the device *index* for e.g. GPUs.
	pub fn id(&self) -> u32 {
		ortsys![unsafe HardwareDevice_DeviceId(self.ptr.as_ptr())]
	}

	/// Returns the name of the manufacturer of the device.
	///
	/// ```no_run
	/// # use ort::{environment::Environment, memory::DeviceType};
	/// # fn main() -> ort::Result<()> {
	/// let env = Environment::current()?;
	/// let cpu = env.devices().next().unwrap().hardware_device();
	/// assert_eq!(cpu.vendor().unwrap(), "Intel");
	/// # Ok(())
	/// # }
	/// ```
	pub fn vendor(&self) -> Result<&'e str> {
		let vendor = ortsys![unsafe HardwareDevice_Vendor(self.ptr.as_ptr())];
		unsafe { CStr::from_ptr(vendor) }.to_str().map_err(Error::from)
	}
}

impl AsPointer for HardwareDevice<'_> {
	type Sys = ort_sys::OrtHardwareDevice;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

#[cfg(feature = "api-24")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-24")))]
pub struct CompatibilityInfo<E> {
	str: NonNull<c_char>,
	allocator: Allocator,
	_p: PhantomData<E>
}

#[cfg(feature = "api-24")]
impl<E: ExecutionProvider> CompatibilityInfo<E> {
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn from_compiled_model_file(path: impl AsRef<Path>, ep: &E) -> Result<Option<Self>> {
		let path = crate::util::path_to_os_char(path);
		let mut allocator = Allocator::default();
		// In typical ONNX Runtime fashion there is zero information about what the hell `ep_type` is or what it comes from.
		// I also can't get any EP to produce a compiled model that even has the compatibility info field to check. So I'll
		// assume it's the EP name but of course that'll turn out to be wrong.
		let ep_type = CString::new(ep.name())?;
		let mut str = ptr::null_mut();
		ortsys![unsafe GetCompatibilityInfoFromModel(path.as_ptr(), ep_type.as_ptr(), allocator.ptr_mut(), &mut str)?];
		Ok(NonNull::new(str).map(|str| Self { str, allocator, _p: PhantomData }))
	}

	pub fn from_compiled_model_bytes(bytes: impl AsRef<[u8]>, ep: &E) -> Result<Option<Self>> {
		let bytes = bytes.as_ref();
		let mut allocator = Allocator::default();
		let ep_type = CString::new(ep.name())?;
		let mut str = ptr::null_mut();
		ortsys![unsafe GetCompatibilityInfoFromModelBytes(bytes.as_ptr().cast(), bytes.len(), ep_type.as_ptr(), allocator.ptr_mut(), &mut str)?];
		Ok(NonNull::new(str).map(|str| Self { str, allocator, _p: PhantomData }))
	}
}

#[cfg(feature = "api-24")]
impl<E> Drop for CompatibilityInfo<E> {
	fn drop(&mut self) {
		unsafe { self.allocator.free(self.str.as_ptr()) };
	}
}

#[cfg(feature = "api-23")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-23")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceCompatibility {
	/// The requested EP is not applicable for the model (the EP does not support the compatibility API or there is no
	/// compatibility info).
	NotApplicable,
	/// The requested EP supports the model optimally.
	SupportedOptimal,
	/// The requested EP supports the model, but it would perform better if the model were recompiled.
	SupportedPreferRecompilation,
	/// The requested EP does not support the model.
	Unsupported
}

#[cfg(feature = "api-23")]
impl DeviceCompatibility {
	#[inline]
	pub const fn is_supported(&self) -> bool {
		matches!(self, DeviceCompatibility::SupportedOptimal | DeviceCompatibility::SupportedPreferRecompilation)
	}
}

#[cfg(test)]
mod tests {
	use crate::{Result, environment::Environment, memory::DeviceType, session::Session};

	#[test]
	fn test_session_devices() -> Result<()> {
		let env = Environment::current()?;

		let _session1 = Session::builder()?
			.with_devices(env.devices().next(), None)?
			.commit_from_file("tests/data/upsample.onnx")?;

		let options = vec![
			("CPUExecutionProvider.use_arena".to_string(), "1".to_string()),
			("XnnpackExecutionProvider.num_threads".to_string(), "4".to_string()),
		];
		let _session2 = Session::builder()?
			.with_devices(env.devices().next(), Some(&options))?
			.commit_from_file("tests/data/upsample.onnx")?;

		Ok(())
	}
}
