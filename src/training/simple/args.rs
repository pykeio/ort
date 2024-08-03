use std::path::PathBuf;

use super::{DataLoader, TrainerCallbacks};
use crate::session::input::SessionInputs;

pub enum EvaluationStrategy {
	None,
	Steps(usize),
	Epochs(usize)
}

impl EvaluationStrategy {
	pub(crate) fn should_fire(&self, _global_step: usize, iter_step: usize, dataloader_size: Option<usize>) -> bool {
		match self {
			Self::None => false,
			Self::Steps(steps) => iter_step > 0 && iter_step % steps == 0,
			Self::Epochs(epochs) => {
				if let Some(dataloader_size) = dataloader_size {
					iter_step > 0 && iter_step % (dataloader_size * epochs) == 0
				} else {
					false
				}
			}
		}
	}
}

pub enum CheckpointStrategy {
	None,
	Steps(usize),
	Epochs(usize)
}

impl CheckpointStrategy {
	pub(crate) fn should_fire(&self, _global_step: usize, iter_step: usize, dataloader_size: Option<usize>) -> bool {
		match self {
			Self::None => false,
			Self::Steps(steps) => iter_step > 0 && iter_step % steps == 0,
			Self::Epochs(epochs) => {
				if let Some(dataloader_size) = dataloader_size {
					iter_step > 0 && iter_step % (dataloader_size * epochs) == 0
				} else {
					false
				}
			}
		}
	}
}

pub struct TrainingArguments<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize> {
	pub(crate) loader: Box<dyn DataLoader<I, L>>,
	pub(crate) eval_loader: Option<Box<dyn DataLoader<I, L>>>,
	pub(crate) eval_strategy: EvaluationStrategy,
	pub(crate) ckpt_strategy: CheckpointStrategy,
	pub(crate) ckpt_path: PathBuf,
	pub(crate) lr: f32,
	pub(crate) max_saved_ckpts: usize,
	pub(crate) gradient_accumulation_steps: usize,
	pub(crate) max_steps: usize,
	pub(crate) max_eval_steps: usize,
	pub(crate) callbacks: Vec<Box<dyn TrainerCallbacks>>
}

impl<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize>
	TrainingArguments<I, L, NI, NL>
{
	pub fn new<D: DataLoader<I, L> + 'static>(train_loader: D) -> Self {
		Self {
			loader: Box::new(train_loader),
			eval_loader: None,
			eval_strategy: EvaluationStrategy::None,
			ckpt_strategy: CheckpointStrategy::Epochs(1),
			ckpt_path: PathBuf::from("checkpoints"),
			lr: 1e-4,
			gradient_accumulation_steps: 1,
			max_saved_ckpts: 1,
			max_steps: usize::MAX,
			max_eval_steps: usize::MAX,
			callbacks: Vec::new()
		}
	}

	pub fn with_lr(mut self, lr: f32) -> Self {
		self.lr = lr;
		self
	}

	pub fn with_max_steps(mut self, steps: usize) -> Self {
		self.max_steps = steps;
		self
	}

	pub fn with_epochs(mut self, epochs: f32) -> Self {
		self.max_steps = self.loader.len().map(|l| (l as f32 * epochs).trunc() as usize).unwrap_or(usize::MAX);
		self
	}

	pub fn with_max_eval_steps(mut self, steps: usize) -> Self {
		self.max_eval_steps = steps;
		self
	}

	pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
		self.gradient_accumulation_steps = steps.max(1);
		self
	}

	pub fn with_ckpt_path(mut self, path: impl Into<PathBuf>) -> Self {
		self.ckpt_path = path.into();
		self
	}

	pub fn with_ckpt_strategy(mut self, strategy: CheckpointStrategy) -> Self {
		self.ckpt_strategy = strategy;
		self
	}

	pub fn with_max_saved_ckpts(mut self, max_ckpts: usize) -> Self {
		self.max_saved_ckpts = max_ckpts;
		self
	}

	pub fn with_eval_loader<D: DataLoader<I, L> + 'static>(mut self, eval_loader: D) -> Self {
		self.eval_loader = Some(Box::new(eval_loader));
		self
	}

	pub fn with_eval_strategy(mut self, strategy: EvaluationStrategy) -> Self {
		self.eval_strategy = strategy;
		self
	}

	pub fn with_callbacks(mut self, callbacks: impl TrainerCallbacks + 'static) -> Self {
		self.callbacks.push(Box::new(callbacks));
		self
	}
}
