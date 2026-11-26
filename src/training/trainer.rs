use alloc::{borrow::Cow, sync::Arc};
use core::{
	any::Any,
	cell::Cell,
	fmt,
	marker::PhantomData,
	mem,
	ptr::{self, NonNull}
};
use std::path::Path;

use ort_sys::c_char;
use smallvec::SmallVec;

use super::{Checkpoint, Optimizer, training_api};
use crate::{
	AsPointer, ErrorCode,
	environment::Environment,
	error::{Error, Result},
	memory::{Allocator, MemoryInfo},
	ortsys,
	session::{SessionInputValue, SessionInputs, SessionOutputs, builder::SessionBuilder},
	util::{char_p_to_string, run_on_drop, with_cstr_ptr_array},
	value::{IntoTensorElementType, Tensor, Value, ValueFlags, ValueInner, ValueType}
};

#[derive(Debug)]
pub struct Trainer {
	inner: Arc<TrainingSessionInner>,
	train_output_names: Vec<String>,
	eval_output_names: Vec<String>,
	train_input_names: Vec<String>,
	eval_input_names: Vec<String>,
	// force !Sync because TrainingSession has zero internal synchronization
	_p: PhantomData<Cell<()>>
}

pub(crate) struct TrainingSessionInner {
	ptr: NonNull<ort_sys::OrtTrainingSession>,
	ckpt: Checkpoint,
	_allocator: Allocator,
	_environment: Environment,
	// hold onto operators/logger/thread manager/etc
	_extras: SmallVec<[Arc<dyn Any>; 4]>
}

unsafe impl Send for TrainingSessionInner {}
unsafe impl Sync for TrainingSessionInner {}

impl fmt::Debug for TrainingSessionInner {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("TrainingSessionInner").field("ptr", &self.ptr).finish_non_exhaustive()
	}
}

impl Drop for TrainingSessionInner {
	fn drop(&mut self) {
		crate::logging::drop!(Trainer, self.ptr);
		ortsys![@training: unsafe ReleaseTrainingSession(self.ptr.as_ptr())];
	}
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

		let mut ptr: *mut ort_sys::OrtTrainingSession = ptr::null_mut();
		ortsys![@training:
			unsafe CreateTrainingSession(
				session_options.environment.ptr(),
				session_options.ptr(),
				ckpt.ptr.as_ptr(),
				training_model_path.as_ptr(),
				eval_model_path.as_ptr(),
				optimizer_model_path.as_ptr(),
				&mut ptr
			)?;
			nonNull(ptr)
		];
		Self::new_inner(ptr, session_options, allocator, ckpt)
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
		let mut ptr: *mut ort_sys::OrtTrainingSession = ptr::null_mut();
		ortsys![@training:
			unsafe CreateTrainingSessionFromBuffer(
				session_options.environment.ptr(),
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
		Self::new_inner(ptr, session_options, allocator, ckpt)
	}

	fn new_inner(ptr: NonNull<ort_sys::OrtTrainingSession>, session_options: SessionBuilder, allocator: Allocator, ckpt: Checkpoint) -> Result<Self> {
		let session_guard = run_on_drop(|| ortsys![@training: unsafe ReleaseTrainingSession(ptr.as_ptr())]);

		let api = training_api()?;
		let train_output_names =
			extract_io_names(ptr, &allocator, api.TrainingSessionGetTrainingModelOutputCount, api.TrainingSessionGetTrainingModelOutputName)?;
		let eval_output_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetEvalModelOutputCount, api.TrainingSessionGetEvalModelOutputName)?;

		let train_input_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetTrainingModelInputCount, api.TrainingSessionGetTrainingModelInputName)?;
		let eval_input_names = extract_io_names(ptr, &allocator, api.TrainingSessionGetEvalModelInputCount, api.TrainingSessionGetEvalModelInputName)?;

		mem::forget(session_guard);
		crate::logging::create!(Trainer, ptr);

		Ok(Self {
			inner: Arc::new(TrainingSessionInner {
				ptr,
				ckpt,
				_allocator: allocator,
				_extras: session_options.extras(),
				_environment: session_options.environment
			}),
			train_output_names,
			train_input_names,
			eval_output_names,
			eval_input_names,
			_p: PhantomData
		})
	}

	pub fn step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s mut self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s>> {
		self.run_step(Step::Train, &inputs.into(), &labels.into())
	}

	pub fn eval_step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s mut self,
		inputs: impl Into<SessionInputs<'i1, 'v1, N1>>,
		labels: impl Into<SessionInputs<'i2, 'v2, N2>>
	) -> Result<SessionOutputs<'s>> {
		self.run_step(Step::Eval, &inputs.into(), &labels.into())
	}

	fn run_step<'s, 'i1, 'v1: 'i1, 'i2: 'i1, 'v2: 'i2 + 'i1, const N1: usize, const N2: usize>(
		&'s mut self,
		step: Step,
		inputs: &SessionInputs<'i1, 'v1, N1>,
		labels: &SessionInputs<'i2, 'v2, N2>
	) -> Result<SessionOutputs<'s>> {
		let (input_names, output_names) = match step {
			Step::Train => (&self.train_input_names, &self.train_output_names),
			Step::Eval => (&self.eval_input_names, &self.eval_output_names)
		};
		let input_ort_values: Vec<*const ort_sys::OrtValue> = resolve_inputs(input_names, inputs)?
			.into_iter()
			.chain(resolve_inputs(input_names, labels)?)
			.map(|v| v.ptr())
			.collect();
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![ptr::null_mut(); output_names.len()];

		let api = training_api()?;
		let run = match step {
			Step::Train => api.TrainStep,
			Step::Eval => api.EvalStep
		};
		unsafe {
			Error::result_from_status(run(
				self.inner.ptr.as_ptr(),
				ptr::null(),
				input_ort_values.len(),
				input_ort_values.as_ptr(),
				output_tensor_ptrs.len(),
				output_tensor_ptrs.as_mut_ptr()
			))
		}?;

		let outputs = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| {
				let Some(ptr) = NonNull::new(tensor_ptr) else {
					panic!("OrtValue ptr returned from session Run should not be null");
				};

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(ptr.as_ptr(), &mut is_tensor).expect("infallible")];
				assert!(is_tensor != 0);

				let (memory_info, dtype) = unsafe { (Some(MemoryInfo::from_tensor_value(ptr)), ValueType::from_tensor_value(ptr)) };

				Value {
					inner: ValueInner::new(ptr, dtype, memory_info, Some(self.inner.clone() as Arc<_>), ValueFlags::DROP | ValueFlags::UPGRADABLE, None),
					_markers: PhantomData
				}
			})
			.collect();

		Ok(SessionOutputs::new(output_names.iter().map(String::as_str).collect(), outputs))
	}

	pub fn export<O: AsRef<str>>(&self, out_path: impl AsRef<Path>, output_names: impl AsRef<[O]>) -> Result<()> {
		let out_path = crate::util::path_to_os_char(out_path);
		with_cstr_ptr_array(output_names.as_ref(), &mut |output_name_ptrs| {
			ortsys![@training: unsafe ExportModelForInferencing(self.inner.ptr.as_ptr(), out_path.as_ptr(), output_name_ptrs.len(), output_name_ptrs.as_ptr())?];
			Ok(())
		})?;
		Ok(())
	}

	pub fn num_params(&self, trainable_only: bool) -> Result<usize> {
		let mut out = 0;
		ortsys![@training: unsafe GetParametersSize(self.inner.ptr.as_ptr(), &mut out, trainable_only)?];
		Ok(out)
	}

	pub fn copy_parameters_to<T: IntoTensorElementType + fmt::Debug>(&self, value: &mut Tensor<T>, trainable_only: bool) -> Result<()> {
		ortsys![@training: unsafe CopyParametersToBuffer(self.inner.ptr.as_ptr(), value.ptr_mut(), trainable_only)?];
		Ok(())
	}

	pub fn copy_parameters_from<T: IntoTensorElementType + fmt::Debug>(&mut self, value: &Tensor<T>, trainable_only: bool) -> Result<()> {
		ortsys![@training: unsafe CopyBufferToParameters(self.inner.ptr.as_ptr(), value.ptr().cast_mut(), trainable_only)?];
		Ok(())
	}

	pub fn optimizer(&self) -> Optimizer<'_> {
		Optimizer::new(self.inner.ptr)
	}

	pub fn checkpoint(&self) -> &Checkpoint {
		&self.inner.ckpt
	}
}

impl AsPointer for Trainer {
	type Sys = ort_sys::OrtTrainingSession;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.ptr.as_ptr()
	}
}

#[derive(Clone, Copy)]
enum Step {
	Train,
	Eval
}

fn resolve_inputs<'a, 'v, const N: usize>(input_names: &[String], inputs: &'a SessionInputs<'_, 'v, N>) -> Result<Vec<&'a SessionInputValue<'v>>> {
	match inputs {
		SessionInputs::ValueSlice(values) => Ok(values.iter().collect()),
		SessionInputs::ValueArray(values) => Ok(values.iter().collect()),
		SessionInputs::ValueMap(values) => mapped_inputs(input_names, values)
	}
}

fn mapped_inputs<'v, 'a>(input_names: &[String], values: &'a [(Cow<'_, str>, SessionInputValue<'v>)]) -> Result<Vec<&'a SessionInputValue<'v>>> {
	let mut out = Vec::with_capacity(input_names.len());
	'o: for want_name in input_names {
		for (name, value) in values {
			if want_name == name {
				out.push(value);
				continue 'o;
			}
		}
		return Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("missing input '{want_name}'")));
	}
	Ok(out)
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
	unsafe { Error::result_from_status(get_count(ptr.as_ptr(), &mut count)) }?;
	(0..count)
		.map(|i| {
			let mut name_bytes: *const c_char = ptr::null();
			unsafe { Error::result_from_status(get_name(ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_bytes)) }?;
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
