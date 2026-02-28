use core::ptr;

use super::{ExecutionProvider, RegisterError};
use crate::{error::Result, ortsys, session::builder::SessionBuilder, util::OnceLock};

#[allow(unused)]
fn get_dml_api() -> Result<&'static ort_sys::OrtDmlApi> {
	static DML_API: OnceLock<ort_sys::OrtDmlApi> = OnceLock::new();
	DML_API.get_or_try_init(|| {
		let mut ptr: *const ort_sys::c_void = ptr::null();
		ortsys![unsafe GetExecutionProviderApi(c"DML".as_ptr(), 0, &mut ptr)?];
		assert!(!ptr.is_null());
		Ok(unsafe { (*ptr.cast::<ort_sys::OrtDmlApi>()).clone() })
	})
}

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
			use crate::{AsPointer, Error};

			let api = get_dml_api()?;
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
