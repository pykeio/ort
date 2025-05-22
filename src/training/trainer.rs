use alloc::borrow::Cow;
use core::{
	fmt,
	ptr::{self, NonNull}
};
use std::path::Path;

use ort_sys::c_char;

use super::{Checkpoint, Optimizer, training_api};
use crate::{
	AsPointer, char_p_to_string,
	error::{Result, status_to_result},
	memory::Allocator,
	ortsys,
	session::{RunOptions, SessionInputValue, SessionInputs, SessionOutputs, builder::SessionBuilder},
	tensor::IntoTensorElementType,
	util::with_cstr_ptr_array,
	value::{Tensor, Value}
};

#[derive(Debug)]
pub struct Trainer {
	ptr: NonNull<ort_sys::OrtTrainingSession>,
	train_output_names: Vec<String>,
	eval_output_names: Vec<String>,
	train_input_names: Vec<String>,
	eval_input_names: Vec<String>,
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
		ortsys![@training:
			unsafe CreateTrainingSession(
				env.ptr(),
				session_options.ptr(),
				ckpt.ptr.as_ptr(),
				training_model_path.as_ptr(),
				eval_model_path.as_ptr(),
				optimizer_model_path.as_ptr(),
				&mut ptr
			)?;
			nonNull(ptr)
		];
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
		ortsys![@training:
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
		Self::new_inner(ptr, allocator, ckpt)
	}

	fn new_inner(ptr: NonNull<ort_sys::OrtTrainingSession>, allocator: Allocator, ckpt: Checkpoint) -> Result<Self> {
		let api = training_api()?;
		let train_output_names =
			extract_io_names(ptr, &allocator, api.TrainingSessionGetTrainingModelOutputCount, api.TrainingSessionGetTrainingModelOutputName)?;
		let eval_output_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetEvalModelOutputCount, api.TrainingSessionGetEvalModelOutputName)?;

		let train_input_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetTrainingModelInputCount, api.TrainingSessionGetTrainingModelInputName)?;
		let eval_input_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetEvalModelInputCount, api.TrainingSessionGetEvalModelInputName)?;

		Ok(Self {
			ptr,
			_allocator: allocator,
			train_output_names,
			train_input_names,
			eval_output_names,
			eval_input_names,
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
				SessionInputs::ValueSlice(labels) => self.step_inner(input_values.iter().chain(labels).map(Some), None),
				SessionInputs::ValueArray(labels) => self.step_inner(input_values.iter().chain(labels.iter()).map(Some), None),
				SessionInputs::ValueMap(labels) => {
					let labels = mapped_inputs(&self.train_input_names, &labels);
					self.step_inner(input_values.iter().map(Some).chain(labels), None)
				}
			},
			SessionInputs::ValueArray(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.step_inner(input_values.iter().chain(labels).map(Some), None),
				SessionInputs::ValueArray(labels) => self.step_inner(input_values.iter().chain(labels.iter()).map(Some), None),
				SessionInputs::ValueMap(labels) => {
					let labels = mapped_inputs(&self.train_input_names, &labels);
					self.step_inner(input_values.iter().map(Some).chain(labels), None)
				}
			},
			SessionInputs::ValueMap(input_values) => {
				let input_values = mapped_inputs(&self.train_input_names, &input_values);
				match labels.into() {
					SessionInputs::ValueSlice(labels) => self.step_inner(input_values.into_iter().chain(labels.iter().map(Some)), None),
					SessionInputs::ValueArray(labels) => self.step_inner(input_values.into_iter().chain(labels.iter().map(Some)), None),
					SessionInputs::ValueMap(labels) => {
						let labels = mapped_inputs(&self.train_input_names, &labels);
						self.step_inner(input_values.into_iter().chain(labels), None)
					}
				}
			}
		}
	}

	fn step_inner<'r, 's: 'r, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = Option<&'i1 SessionInputValue<'v1>>>,
		run_options: Option<&'r RunOptions>
	) -> Result<SessionOutputs<'r>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![ptr::null_mut(); self.train_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|v| v.map_or(ptr::null(), |v| v.ptr())).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr() } else { ptr::null() };

		ortsys![@training: unsafe TrainStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr())?];

		let outputs = output_tensor_ptrs
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
	) -> Result<SessionOutputs<'s>> {
		match inputs.into() {
			SessionInputs::ValueSlice(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_step_inner(input_values.iter().chain(labels).map(Some), None),
				SessionInputs::ValueArray(labels) => self.eval_step_inner(input_values.iter().chain(labels.iter()).map(Some), None),
				SessionInputs::ValueMap(labels) => {
					let labels = mapped_inputs(&self.eval_input_names, &labels);
					self.eval_step_inner(input_values.iter().map(Some).chain(labels), None)
				}
			},
			SessionInputs::ValueArray(input_values) => match labels.into() {
				SessionInputs::ValueSlice(labels) => self.eval_step_inner(input_values.iter().chain(labels).map(Some), None),
				SessionInputs::ValueArray(labels) => self.eval_step_inner(input_values.iter().chain(labels.iter()).map(Some), None),
				SessionInputs::ValueMap(labels) => {
					let labels = mapped_inputs(&self.eval_input_names, &labels);
					self.eval_step_inner(input_values.iter().map(Some).chain(labels), None)
				}
			},
			SessionInputs::ValueMap(input_values) => {
				let input_values = mapped_inputs(&self.eval_input_names, &input_values);
				match labels.into() {
					SessionInputs::ValueSlice(labels) => self.eval_step_inner(input_values.into_iter().chain(labels.iter().map(Some)), None),
					SessionInputs::ValueArray(labels) => self.eval_step_inner(input_values.into_iter().chain(labels.iter().map(Some)), None),
					SessionInputs::ValueMap(labels) => {
						let labels = mapped_inputs(&self.eval_input_names, &labels);
						self.eval_step_inner(input_values.into_iter().chain(labels), None)
					}
				}
			}
		}
	}

	fn eval_step_inner<'r, 's: 'r, 'i1, 'v1: 'i1, 'i2, 'v2: 'i2>(
		&'s self,
		input_values: impl Iterator<Item = Option<&'i1 SessionInputValue<'v1>>>,
		run_options: Option<&'r RunOptions>
	) -> Result<SessionOutputs<'r>> {
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![ptr::null_mut(); self.eval_output_names.len()];

		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|v| v.map_or(ptr::null(), |v| v.ptr())).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr() } else { ptr::null() };

		ortsys![@training: unsafe EvalStep(self.ptr.as_ptr(), run_options_ptr, input_ort_values.len(), input_ort_values.as_ptr(), output_tensor_ptrs.len(), output_tensor_ptrs.as_mut_ptr())?];

		let outputs = output_tensor_ptrs
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
		with_cstr_ptr_array(output_names.as_ref(), &|output_name_ptrs| {
			ortsys![@training: unsafe ExportModelForInferencing(self.ptr.as_ptr(), out_path.as_ptr(), output_name_ptrs.len(), output_name_ptrs.as_ptr())?];
			Ok(())
		})?;
		Ok(())
	}

	pub fn num_params(&self, trainable_only: bool) -> Result<usize> {
		let mut out = 0;
		ortsys![@training: unsafe GetParametersSize(self.ptr.as_ptr(), &mut out, trainable_only)?];
		Ok(out)
	}

	pub fn copy_parameters_to<T: IntoTensorElementType + fmt::Debug>(&self, value: &mut Tensor<T>, trainable_only: bool) -> Result<()> {
		ortsys![@training: unsafe CopyParametersToBuffer(self.ptr.as_ptr(), value.ptr_mut(), trainable_only)?];
		Ok(())
	}

	pub fn copy_parameters_from<T: IntoTensorElementType + fmt::Debug>(&mut self, value: &Tensor<T>, trainable_only: bool) -> Result<()> {
		ortsys![@training: unsafe CopyBufferToParameters(self.ptr.as_ptr(), value.ptr().cast_mut(), trainable_only)?];
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
		ortsys![@training: unsafe ReleaseTrainingSession(self.ptr.as_ptr())];
	}
}

fn mapped_inputs<'v, 'a>(input_names: &[String], values: &'a [(Cow<'_, str>, SessionInputValue<'v>)]) -> Vec<Option<&'a SessionInputValue<'v>>> {
	let mut out = Vec::with_capacity(input_names.len());
	'o: for want_name in input_names {
		for (name, value) in values {
			if want_name == name {
				out.push(Some(value));
				continue 'o;
			}
		}
		out.push(None);
	}
	out
}

fn extract_io_names(
	ptr: NonNull<ort_sys::OrtTrainingSession>,
	allocator: &Allocator,
	get_count: unsafe extern "system" fn(sess: *const ort_sys::OrtTrainingSession, out: *mut usize) -> ort_sys::OrtStatusPtr,
	get_name: unsafe extern "system" fn(
		sess: *const ort_sys::OrtTrainingSession,
		index: usize,
		allocator: *mut ort_sys::OrtAllocator,
		output: *mut *const c_char
	) -> ort_sys::OrtStatusPtr
) -> Result<Vec<String>> {
	let mut count = 0;
	unsafe { status_to_result(get_count(ptr.as_ptr(), &mut count)) }?;
	(0..count)
		.map(|i| {
			let mut name_bytes: *const c_char = ptr::null();
			unsafe { status_to_result(get_name(ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_bytes)) }?;
			let name = match char_p_to_string(name_bytes) {
				Ok(name) => name,
				Err(e) => {
					unsafe { allocator.free(name_bytes.cast_mut()) };
					return Err(e);
				}
			};
			unsafe { allocator.free(name_bytes.cast_mut()) };
			Ok(name)
		})
		.collect::<Result<Vec<String>>>()
}
