use core::{ffi::CStr, marker::PhantomData, ptr::NonNull};

use crate::{AsPointer, Error, Result, memory::DeviceType, ortsys};

pub struct Device<'e> {
	ptr: NonNull<ort_sys::OrtEpDevice>,
	hw_ptr: NonNull<ort_sys::OrtHardwareDevice>,
	_env: PhantomData<&'e ()>
}

impl<'e> Device<'e> {
	pub(crate) fn new(ptr: NonNull<ort_sys::OrtEpDevice>) -> Self {
		Self {
			ptr,
			hw_ptr: NonNull::new(ortsys![unsafe EpDevice_Device(ptr.as_ptr())].cast_mut()).expect("invalid device"),
			_env: PhantomData
		}
	}

	/// Returns the [name of the EP](crate::ep::ExecutionProvider::name) this device belongs to.
	pub fn ep(&self) -> Result<&'e str> {
		let name = ortsys![unsafe EpDevice_EpName(self.ptr.as_ptr())];
		unsafe { CStr::from_ptr(name) }.to_str().map_err(Error::from)
	}

	/// Returns the name of the EP vendor this device belongs to, e.g. `"Microsoft"` for DirectML devices.
	///
	/// For the *manufacturer* of the device, see [`Device::vendor`].
	pub fn ep_vendor(&self) -> Result<&'e str> {
		let vendor = ortsys![unsafe EpDevice_EpVendor(self.ptr.as_ptr())];
		unsafe { CStr::from_ptr(vendor) }.to_str().map_err(Error::from)
	}

	/// Returns the [type](DeviceType) of the device - CPU, GPU, or NPU.
	pub fn ty(&self) -> DeviceType {
		match ortsys![unsafe HardwareDevice_Type(self.hw_ptr.as_ptr())] {
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_CPU => DeviceType::CPU,
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_GPU => DeviceType::GPU,
			ort_sys::OrtHardwareDeviceType::OrtHardwareDeviceType_NPU => DeviceType::NPU
		}
	}

	/// Returns the device ID.
	///
	/// Appears to be arbitrary and is **not** the same as the device *index* (but is unique per device).
	pub fn id(&self) -> u32 {
		ortsys![unsafe HardwareDevice_DeviceId(self.hw_ptr.as_ptr())]
	}

	/// Returns the name of the manufacturer of the device.
	pub fn vendor(&self) -> Result<&'e str> {
		let vendor = ortsys![unsafe HardwareDevice_Vendor(self.hw_ptr.as_ptr())];
		unsafe { CStr::from_ptr(vendor) }.to_str().map_err(Error::from)
	}
}

impl AsPointer for Device<'_> {
	type Sys = ort_sys::OrtEpDevice;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

#[cfg(test)]
mod tests {
	use crate::{Result, memory::DeviceType, session::Session};

	#[test]
	fn test_device_meta() -> Result<()> {
		let env = crate::environment::current()?;
		// CPUExecutionProvider should always be first (for now anyways...)
		let device = env.devices().next().expect("");
		assert!(matches!(device.ep(), Ok("CPUExecutionProvider")));
		assert!(matches!(device.ep_vendor(), Ok("Microsoft")));
		assert_eq!(device.ty(), DeviceType::CPU);

		Ok(())
	}

	#[test]
	fn test_session_devices() -> Result<()> {
		let env = crate::environment::current()?;

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
