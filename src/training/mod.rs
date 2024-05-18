use std::{
	ptr::NonNull,
	sync::{
		atomic::{AtomicPtr, Ordering},
		OnceLock
	}
};

use crate::{ortsys, Error, Result};

pub(crate) static TRAINING_API: OnceLock<AtomicPtr<ort_sys::OrtTrainingApi>> = OnceLock::new();

/// Returns a pointer to the global [`ort_sys::OrtTrainingApi`] object, or errors if the Training API is not enabled.
///
/// # Panics
/// May panic if:
/// - Getting the `OrtApi` struct fails, due to `ort` loading an unsupported version of ONNX Runtime.
/// - Loading the ONNX Runtime dynamic library fails if the `load-dynamic` feature is enabled.
pub fn training_api() -> Result<NonNull<ort_sys::OrtTrainingApi>> {
	NonNull::new(
		TRAINING_API
			.get_or_init(|| {
				let training_api = ortsys![unsafe GetTrainingApi(ort_sys::ORT_API_VERSION)];
				AtomicPtr::new(training_api.cast_mut())
			})
			.load(Ordering::Relaxed)
	)
	.ok_or(Error::TrainingNotEnabled)
}
