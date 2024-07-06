use std::{
	path::Path,
	ptr::{self, NonNull},
	sync::{
		atomic::{AtomicPtr, Ordering},
		OnceLock
	}
};

use crate::{ortsys, Error, Result, RunOptions};

mod simple;
mod trainer;

pub use self::{
	simple::{iterable_data_loader, CheckpointStrategy, DataLoader, EvaluationStrategy, IterableDataLoader, TrainingArguments},
	trainer::Trainer
};

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

macro_rules! trainsys {
	($method:ident) => {
		$crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))
	};
	(unsafe $method:ident) => {
		unsafe { $crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null"))) }
	};
	($method:ident($($n:expr),+ $(,)?)) => {
		$crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)
	};
	(unsafe $method:ident($($n:expr),+ $(,)?)) => {
		unsafe { $crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }
	};
	($method:ident($($n:expr),+ $(,)?).expect($e:expr)) => {
		$crate::error::status_to_result($crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).expect($e)
	};
	(unsafe $method:ident($($n:expr),+ $(,)?).expect($e:expr)) => {
		$crate::error::status_to_result(unsafe { $crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).expect($e)
	};
	($method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+);
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?); nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		let _x = unsafe { $crate::training_api().unwrap().as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) };
		$($crate::error::assert_non_null_pointer($check, stringify!($method)).unwrap();)+
		_x
	}};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result($crate::training_api()?.as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).map_err($err)?;
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr$(;)?) => {
		$crate::error::status_to_result(unsafe { $crate::training_api()?.as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).map_err($err)?;
	};
	($method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {
		$crate::error::status_to_result($crate::training_api()?.as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+)).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	};
	(unsafe $method:ident($($n:expr),+ $(,)?) -> $err:expr; nonNull($($check:expr),+ $(,)?)$(;)?) => {{
		$crate::error::status_to_result(unsafe { $crate::training_api()?.as_ref().$method.unwrap_or_else(|| unreachable!(concat!("Method `", stringify!($method), "` is null")))($($n),+) }).map_err($err)?;
		$($crate::error::assert_non_null_pointer($check, stringify!($method))?;)+
	}};
}
pub(crate) use trainsys;

#[derive(Debug)]
pub struct Checkpoint {
	pub(crate) ptr: NonNull<ort_sys::OrtCheckpointState>
}

impl Checkpoint {
	pub fn load(path: impl AsRef<Path>) -> Result<Self> {
		let path = crate::util::path_to_os_char(path);
		let mut ptr: *mut ort_sys::OrtCheckpointState = ptr::null_mut();
		trainsys![unsafe LoadCheckpoint(path.as_ptr(), &mut ptr) -> Error::CreateSession; nonNull(ptr)];
		Ok(Checkpoint {
			ptr: unsafe { NonNull::new_unchecked(ptr) }
		})
	}

	pub fn save(&self, path: impl AsRef<Path>, include_optimizer_state: bool) -> Result<()> {
		let path = crate::util::path_to_os_char(path);
		trainsys![unsafe SaveCheckpoint(self.ptr.as_ptr(), path.as_ptr(), include_optimizer_state) -> Error::CreateSession];
		Ok(())
	}
}

impl Drop for Checkpoint {
	fn drop(&mut self) {
		tracing::trace!("dropping checkpoint");
		trainsys![unsafe ReleaseCheckpointState(self.ptr.as_ptr())];
	}
}

#[derive(Debug)]
pub struct Optimizer(NonNull<ort_sys::OrtTrainingSession>);

impl Optimizer {
	pub fn reset_grad(&self) -> Result<()> {
		trainsys![unsafe LazyResetGrad(self.0.as_ptr()) -> Error::CreateSession];
		Ok(())
	}

	pub fn lr(&self) -> Result<f32> {
		let mut lr = f32::NAN;
		trainsys![unsafe GetLearningRate(self.0.as_ptr(), &mut lr) -> Error::CreateSession];
		Ok(lr)
	}

	pub fn set_lr(&self, lr: f32) -> Result<()> {
		trainsys![unsafe SetLearningRate(self.0.as_ptr(), lr) -> Error::CreateSession];
		Ok(())
	}

	pub fn step(&self) -> Result<()> {
		self.step_with_options(RunOptions::new()?)
	}

	pub fn step_with_options(&self, options: RunOptions) -> Result<()> {
		trainsys![unsafe OptimizerStep(self.0.as_ptr(), options.run_options_ptr.as_ptr()) -> Error::CreateSession];
		Ok(())
	}
}
