use core::{
	ops::Deref,
	ptr::{self, NonNull}
};

use super::{ExecutionProvider, RegisterError};
use crate::{
	AsPointer,
	error::{Error, Result},
	memory::Allocator,
	ortsys,
	session::builder::SessionBuilder,
	util::OnceLock
};

#[derive(Default, Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum PerformancePreference {
	#[default]
	Default,
	HighPerformance,
	MinimumPower
}

#[derive(Default, Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum DeviceFilter {
	#[default]
	Gpu,
	Npu,
	Any
}

/// [DirectML execution provider](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) for
/// DirectX 12-enabled hardware on Windows.
///
/// # Performance considerations
/// The DirectML EP performs best when the size of inputs & outputs are known when the session is created. For graphs
/// with dynamically sized inputs, you can override individual dimensions by constructing the session with
/// [`SessionBuilder::with_dimension_override`]:
/// ```no_run
/// # use ort::{ep, session::Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_execution_providers([ep::DirectML::default().build()])?
/// 	.with_dimension_override("batch", 1)?
/// 	.with_dimension_override("seq_len", 512)?
/// 	.commit_from_file("gpt2.onnx")?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct DirectML {
	device_id: Option<i32>,
	performance_preference: PerformancePreference,
	device_filter: DeviceFilter
}

super::impl_ep!(DirectML);

impl DirectML {
	/// Configures which device the EP should use.
	///
	/// This overrides the [device filter](Self::with_device_filter) and [performance
	/// preference](Self::with_performance_preference).
	///
	/// ```
	/// # use ort::{ep, session::Session};
	/// # fn main() -> ort::Result<()> {
	/// let ep = ep::DirectML::default().with_device_id(1).build();
	/// # Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_device_id(mut self, device_id: i32) -> Self {
		self.device_id = Some(device_id);
		self
	}

	#[must_use]
	pub fn with_performance_preference(mut self, pref: PerformancePreference) -> Self {
		self.performance_preference = pref;
		self
	}

	#[must_use]
	pub fn with_device_filter(mut self, filter: DeviceFilter) -> Self {
		self.device_filter = filter;
		self
	}
}

impl ExecutionProvider for DirectML {
	fn name(&self) -> &'static str {
		"DmlExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(target_os = "windows")
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "directml"))]
		{
			let api = api().ok_or_else(|| Error::new("DirectML is not supported in this build of ONNX Runtime"))?;
			if let Some(device_id) = self.device_id {
				unsafe { Error::result_from_status((api.SessionOptionsAppendExecutionProvider_DML)(session_builder.ptr_mut(), device_id as _)) }?;
			} else {
				let device_options = ort_sys::OrtDmlDeviceOptions {
					Filter: match self.device_filter {
						DeviceFilter::Any => ort_sys::OrtDmlDeviceFilter::Any,
						DeviceFilter::Gpu => ort_sys::OrtDmlDeviceFilter::Gpu,
						DeviceFilter::Npu => ort_sys::OrtDmlDeviceFilter::Npu
					},
					Preference: match self.performance_preference {
						PerformancePreference::Default => ort_sys::OrtDmlPerformancePreference::Default,
						PerformancePreference::HighPerformance => ort_sys::OrtDmlPerformancePreference::HighPerformance,
						PerformancePreference::MinimumPower => ort_sys::OrtDmlPerformancePreference::MinimumPower
					}
				};
				unsafe { Error::result_from_status((api.SessionOptionsAppendExecutionProvider_DML2)(session_builder.ptr_mut(), &device_options)) }?;
			}

			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}

/// Returns the [`OrtDmlApi`](ort_sys::OrtDmlApi) if this build of ONNX Runtime supports DirectML.
pub fn api() -> Option<&'static ort_sys::OrtDmlApi> {
	static DML_API: OnceLock<&'static ort_sys::OrtDmlApi> = OnceLock::new();
	DML_API
		.get_or_try_init(|| {
			let mut ptr: *const ort_sys::c_void = ptr::null();
			ortsys![unsafe GetExecutionProviderApi(c"DML".as_ptr(), 0, &mut ptr)?; nonNull(ptr)];
			let ptr: NonNull<*const ort_sys::c_void> = ptr; // weird type inference stuff
			Ok(unsafe { ptr.cast::<ort_sys::OrtDmlApi>().as_ref() })
		})
		.ok()
		.copied()
}

/// Creates a DirectML resource from a Direct3D resource.
///
/// # Safety
/// `resource` must be a valid, non-null `ID3D12Resource`.
pub unsafe fn resource_from_d3d(resource: *mut ()) -> Result<DMLResource> {
	let mut ptr: *mut ort_sys::c_void = ptr::null_mut();
	let dml_api = api().ok_or_else(|| Error::new("DirectML is not supported in this build of ONNX Runtime"))?;
	unsafe { Error::result_from_status((dml_api.CreateGPUAllocationFromD3DResource)(resource.cast(), &mut ptr)) }?;
	Ok(DMLResource(ptr))
}

#[repr(transparent)]
#[derive(Debug)]
pub struct DMLResource(*mut ort_sys::c_void);

impl DMLResource {
	pub fn d3d_resource(&self, allocator: &Allocator) -> Result<*mut ()> {
		let mut ptr: *mut ort_sys::c_void = ptr::null_mut();
		let dml_api = api().ok_or_else(|| Error::new("DirectML is not supported in this build of ONNX Runtime"))?;
		unsafe { Error::result_from_status((dml_api.GetD3D12ResourceFromAllocation)(allocator.ptr().cast_mut(), self.0.cast(), &mut ptr)) }?;
		Ok(ptr.cast())
	}
}

impl Deref for DMLResource {
	type Target = *mut ort_sys::c_void;

	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl AsPointer for DMLResource {
	type Sys = ();

	fn ptr(&self) -> *const Self::Sys {
		self.0.cast()
	}
}

impl Drop for DMLResource {
	fn drop(&mut self) {
		let Some(dml_api) = api() else {
			return;
		};

		let _ = unsafe { Error::result_from_status((dml_api.FreeGPUAllocation)(self.0)) };
	}
}
