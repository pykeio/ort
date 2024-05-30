use std::{
	ffi::CString,
	path::Path,
	ptr::{self, NonNull},
	sync::{
		atomic::{AtomicPtr, Ordering},
		Arc, OnceLock
	}
};

use ort_sys::c_char;

use crate::{
	char_p_to_string,
	error::{assert_non_null_pointer, status_to_result},
	ortsys, Allocator, Error, Result, RunOptions, SessionBuilder, SessionInputValue, SessionInputs, SessionOutputs, Value
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

#[derive(Debug)]
pub struct Trainer {
	pub(crate) ptr: NonNull<ort_sys::OrtTrainingSession>,
	train_output_names: Vec<String>,
	optimizer: Optimizer,
	ckpt: Checkpoint,
	_allocator: Allocator
}

impl Trainer {
	pub fn new(
		session_options: SessionBuilder,
		allocator: Allocator,
		ckpt: Checkpoint,
		training_model_path: impl AsRef<Path>,
		eval_model_path: impl AsRef<Path>,
		optimizer_model_path: impl AsRef<Path>
	) -> Result<Self> {
		let training_model_path = crate::util::path_to_os_char(training_model_path);
		let eval_model_path = crate::util::path_to_os_char(eval_model_path);
		let optimizer_model_path = crate::util::path_to_os_char(optimizer_model_path);

		let env = crate::get_environment()?;

		let mut ptr: *mut ort_sys::OrtTrainingSession = ptr::null_mut();
		trainsys![unsafe CreateTrainingSession(env.ptr(), session_options.session_options_ptr.as_ptr(), ckpt.ptr.as_ptr(), training_model_path.as_ptr(), eval_model_path.as_ptr(), optimizer_model_path.as_ptr(), &mut ptr) -> Error::CreateSession; nonNull(ptr)];

		let ptr = unsafe { NonNull::new_unchecked(ptr) };

		let mut train_output_len = 0;
		trainsys![unsafe TrainingSessionGetTrainingModelOutputCount(ptr.as_ptr(), &mut train_output_len) -> Error::CreateSession];
		let train_output_names = (0..train_output_len)
			.map(|i| {
				let mut name_bytes: *mut c_char = std::ptr::null_mut();
				trainsys![unsafe TrainingSessionGetTrainingModelOutputName(ptr.as_ptr(), i, allocator.ptr.as_ptr(), &mut name_bytes) -> Error::CreateSession];
				let name = match char_p_to_string(name_bytes) {
					Ok(name) => name,
					Err(e) => {
						unsafe { allocator.free(name_bytes) };
						return Err(e);
					}
				};
				unsafe { allocator.free(name_bytes) };
				Ok(name)
			})
			.collect::<Result<Vec<String>>>()?;

		Ok(Self {
			ptr,
			_allocator: allocator,
			train_output_names,
			optimizer: Optimizer(ptr),
			ckpt
		})
	}

	pub fn step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s>> {
		match inputs.into() {
			SessionInputs::ValueSlice(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.step_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.step_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueArray(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.step_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.step_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
		}
	}

	fn step_inner<'s, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = &'i1 SessionInputValue<'v1>>,
		run_options: Option<Arc<RunOptions>>
	) -> Result<SessionOutputs<'s>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.train_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr().cast_const()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};

		trainsys![unsafe TrainStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr()) -> Error::SessionRun];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				// TODO: `Value` should absolutely be refactored to accept a different backing pointer than `SharedSessionInner`.
				// but for now, nobody should be using the loss tensor past the lifetime of the trainer... right...? 😣
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), None)
			})
			.collect();

		Ok(SessionOutputs::new(self.train_output_names.iter().map(|o| o.as_str()), outputs))
	}

	pub fn eval<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s>> {
		match inputs.into() {
			SessionInputs::ValueSlice(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.eval_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueArray(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.eval_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
		}
	}

	fn eval_inner<'s, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = &'i1 SessionInputValue<'v1>>,
		run_options: Option<Arc<RunOptions>>
	) -> Result<SessionOutputs<'s>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.train_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr().cast_const()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};

		trainsys![unsafe EvalStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr()) -> Error::SessionRun];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				// TODO: `Value` should absolutely be refactored to accept a different backing pointer than `SharedSessionInner`.
				// but for now, nobody should be using the loss tensor past the lifetime of the trainer... right...? 😣
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), None)
			})
			.collect();

		Ok(SessionOutputs::new(self.train_output_names.iter().map(|o| o.as_str()), outputs))
	}

	pub fn export<O: AsRef<str>>(&self, out_path: impl AsRef<Path>, output_names: impl AsRef<[O]>) -> Result<()> {
		let out_path = crate::util::path_to_os_char(out_path);

		let output_names_ptr: Vec<*const c_char> = output_names
			.as_ref()
			.iter()
			.map(|output| CString::new(output.as_ref()).unwrap_or_else(|_| unreachable!()))
			.map(|n| n.into_raw().cast_const())
			.collect();

		let res = trainsys![unsafe ExportModelForInferencing(self.ptr.as_ptr(), out_path.as_ptr(), output_names_ptr.len(), output_names_ptr.as_ptr())];

		// Reconvert name ptrs to CString so drop impl is called and memory is freed
		drop(
			output_names_ptr
				.into_iter()
				.map(|p| {
					assert_non_null_pointer(p, "c_char for CString")?;
					unsafe { Ok(CString::from_raw(p.cast_mut().cast())) }
				})
				.collect::<Result<Vec<_>>>()?
		);

		status_to_result(res).map_err(Error::CreateSession)?;

		Ok(())
	}

	pub fn optimizer(&self) -> &Optimizer {
		&self.optimizer
	}

	pub fn ckpt(&self) -> &Checkpoint {
		&self.ckpt
	}
}

impl Drop for Trainer {
	fn drop(&mut self) {
		tracing::trace!("dropping trainer");
		trainsys![unsafe ReleaseTrainingSession(self.ptr.as_ptr())];
	}
}
