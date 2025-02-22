use std::path::Path;

use super::TrainingArguments;
use crate::{
	error::Result,
	session::input::SessionInputs,
	training::{Checkpoint, Optimizer, Trainer}
};

#[derive(Clone)]
#[non_exhaustive]
pub struct TrainerState {
	pub epoch: Option<f32>,
	/// The total number of weight updates performed on the model.
	pub global_step: usize,
	/// The total number of training batches the model has seen.
	pub iter_step: usize,
	pub gradient_accumulation_steps: usize,
	pub max_steps: usize,
	pub current_lr: f32
}

impl TrainerState {
	pub(crate) fn new<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize>(
		args: &TrainingArguments<I, L, NI, NL>
	) -> Self {
		Self {
			epoch: None,
			global_step: 0,
			iter_step: 0,
			gradient_accumulation_steps: args.gradient_accumulation_steps,
			max_steps: args.max_steps,
			current_lr: args.lr
		}
	}
}

/// Allows callbacks in [`TrainerCallbacks`] to control the training of the model. This includes halting training,
/// updating the learning rate, or exporting the model.
pub struct TrainerControl<'t> {
	pub(crate) halt: bool,
	pub(crate) lr: Option<f32>,
	trainer: &'t Trainer
}

impl<'t> TrainerControl<'t> {
	pub(crate) fn new(trainer: &'t Trainer) -> Self {
		Self { halt: false, trainer, lr: None }
	}

	/// Halts training. Once all callbacks have been called, training will immediately end.
	///
	/// Halting training will fire [`TrainerCallbacks::end`].
	pub fn halt(&mut self) {
		self.halt = true;
	}

	/// Sets the optimizer's learning rate.
	pub fn set_lr(&mut self, lr: f32) {
		self.lr = Some(lr);
	}

	/// Export the model as a complete ONNX graph.
	pub fn export<O: AsRef<str>>(&self, out_path: impl AsRef<Path>, output_names: impl AsRef<[O]>) -> Result<()> {
		self.trainer.export(out_path, output_names)
	}

	pub fn optimizer(&self) -> Optimizer<'_> {
		self.trainer.optimizer()
	}

	pub fn checkpoint(&self) -> &Checkpoint {
		self.trainer.checkpoint()
	}
}

#[allow(unused_variables)]
pub trait TrainerCallbacks: Send {
	/// Called at the beginning of a new epoch.
	fn epoch(&mut self, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}

	/// Called when evaluation is about to begin.
	fn eval_begin(&mut self, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}
	/// Called when evaluation has ended. The `eval_loss` is the average loss over all batches in the evaluation
	/// dataset.
	fn eval_end(&mut self, eval_loss: f32, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}

	/// Called immediately after performing a single forward & backward pass. See also
	/// [`TrainerCallbacks::optimizer_step`], which is called immediately after updating the optimizer.
	///
	/// In the case where [`TrainingArguments::with_gradient_accumulation`] > 1, this will fire as many times as the
	/// gradient accumulation steps is configured before [`TrainerCallbacks::optimizer_step`] is fired.
	fn train_step(&mut self, train_loss: f32, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}
	/// Called immediately after updating the model weights. `loss` is the loss of the last batch.
	fn optimizer_step(&mut self, loss: f32, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}

	/// Called when training ends, either via [`TrainerControl::halt`], or if the maximum steps have been reached.
	fn end(&mut self, state: &TrainerState, control: &mut TrainerControl<'_>) -> Result<()> {
		Ok(())
	}
}
