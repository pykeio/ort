use alloc::ffi::CString;
use core::ptr::{self, NonNull};
use std::path::Path;

use ort_sys::c_char;

use super::{Checkpoint, Optimizer, trainsys};
use crate::{
	AsPointer, char_p_to_string,
	error::{Result, assert_non_null_pointer, status_to_result},
	memory::Allocator,
	session::{RunOptions, SessionInputValue, SessionInputs, SessionOutputs, builder::SessionBuilder},
	value::Value
};

#[derive(Debug)]
pub struct Trainer {
	ptr: NonNull<ort_sys::OrtTrainingSession>,
	train_output_names: Vec<String>,
	eval_output_names: Vec<String>,
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

		let env = crate::environment::get_environment()?;

		let mut ptr: *mut ort_sys::OrtTrainingSession = ptr::null_mut();
		trainsys![unsafe CreateTrainingSession(env.ptr(), session_options.ptr(), ckpt.ptr.as_ptr(), training_model_path.as_ptr(), eval_model_path.as_ptr(), optimizer_model_path.as_ptr(), &mut ptr)?; nonNull(ptr)];

		let ptr = unsafe { NonNull::new_unchecked(ptr) };
		Self::new_inner(ptr, allocator, ckpt)
	}

	pub fn new_from_artifacts(
		session_options: SessionBuilder,
		allocator: Allocator,
		base_dir: impl AsRef<Path>,
		override_ckpt: Option<Checkpoint>
	) -> Result<Self> {
		let base_dir = base_dir.as_ref();
		let ckpt = if let Some(ckpt) = override_ckpt {
			ckpt
		} else {
			Checkpoint::load(base_dir.join("checkpoint"))?
		};
		Self::new(
			session_options,
			allocator,
			ckpt,
			base_dir.join("training_model.onnx"),
			base_dir.join("eval_model.onnx"),
			base_dir.join("optimizer_model.onnx")
		)
	}

	pub fn new_from_memory(
		session_options: SessionBuilder,
		allocator: Allocator,
		ckpt: Checkpoint,
		training_model: &[u8],
		eval_model: &[u8],
		optimizer_model: &[u8]
	) -> Result<Self> {
		let env = crate::environment::get_environment()?;

		let mut ptr: *mut ort_sys::OrtTrainingSession = ptr::null_mut();
		trainsys![
			unsafe CreateTrainingSessionFromBuffer(
				env.ptr(),
				session_options.ptr(),
				ckpt.ptr.as_ptr(),
				training_model.as_ptr().cast(),
				training_model.len(),
				eval_model.as_ptr().cast(),
				eval_model.len(),
				optimizer_model.as_ptr().cast(),
				optimizer_model.len(),
				&mut ptr
			)?;
			nonNull(ptr)
		];

		let ptr = unsafe { NonNull::new_unchecked(ptr) };
		Self::new_inner(ptr, allocator, ckpt)
	}

	fn new_inner(ptr: NonNull<ort_sys::OrtTrainingSession>, allocator: Allocator, ckpt: Checkpoint) -> Result<Self> {
		let mut train_output_len = 0;
		trainsys![unsafe TrainingSessionGetTrainingModelOutputCount(ptr.as_ptr(), &mut train_output_len)?];
		let train_output_names = (0..train_output_len)
			.map(|i| {
				let mut name_bytes: *mut c_char = std::ptr::null_mut();
				trainsys![unsafe TrainingSessionGetTrainingModelOutputName(ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_bytes)?];
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

		let mut eval_output_len = 0;
		trainsys![unsafe TrainingSessionGetEvalModelOutputCount(ptr.as_ptr(), &mut eval_output_len)?];
		let eval_output_names = (0..eval_output_len)
			.map(|i| {
				let mut name_bytes: *mut c_char = std::ptr::null_mut();
				trainsys![unsafe TrainingSessionGetEvalModelOutputName(ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_bytes)?];
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
			eval_output_names,
			ckpt
		})
	}

	pub fn step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s, 's>> {
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

	fn step_inner<'r, 's: 'r, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = &'i1 SessionInputValue<'v1>>,
		run_options: Option<&'r RunOptions>
	) -> Result<SessionOutputs<'r, 's>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.train_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr() } else { std::ptr::null() };

		trainsys![unsafe TrainStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr())?];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				// TODO: `Value` should absolutely be refactored to accept a different backing pointer than `SharedSessionInner`.
				// but for now, nobody should be using the loss tensor past the lifetime of the trainer... right...? 😣
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), None)
			})
			.collect();

		Ok(SessionOutputs::new(self.train_output_names.iter().map(String::as_str).collect(), outputs))
	}

	pub fn eval_step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s, 's>> {
		match inputs.into() {
			SessionInputs::ValueSlice(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_step_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.eval_step_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueArray(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_step_inner(input_values.iter().chain(labels), None),
				SessionInputs::ValueArray(labels) => self.eval_step_inner(input_values.iter().chain(labels.iter()), None),
				SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
			},
			SessionInputs::ValueMap(_) => unimplemented!("named values not supported?")
		}
	}

	fn eval_step_inner<'r, 's: 'r, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = &'i1 SessionInputValue<'v1>>,
		run_options: Option<&'r RunOptions>
	) -> Result<SessionOutputs<'r, 's>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.eval_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr() } else { std::ptr::null() };

		trainsys![unsafe EvalStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr())?];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				// TODO: `Value` should absolutely be refactored to accept a different backing pointer than `SharedSessionInner`.
				// but for now, nobody should be using the loss tensor past the lifetime of the trainer... right...? 😣
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), None)
			})
			.collect();

		Ok(SessionOutputs::new(self.eval_output_names.iter().map(String::as_str).collect(), outputs))
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

		unsafe { status_to_result(res) }?;

		Ok(())
	}

	pub fn optimizer(&self) -> Optimizer<'_> {
		Optimizer::new(self.ptr)
	}

	pub fn checkpoint(&self) -> &Checkpoint {
		&self.ckpt
	}
}

impl AsPointer for Trainer {
	type Sys = ort_sys::OrtTrainingSession;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for Trainer {
	fn drop(&mut self) {
		crate::trace!("dropping trainer");
		trainsys![unsafe ReleaseTrainingSession(self.ptr.as_ptr())];
	}
}
