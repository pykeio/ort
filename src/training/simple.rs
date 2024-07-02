use std::{collections::VecDeque, fs, path::PathBuf};

use crate::{Result, SessionInputs};

#[allow(clippy::len_without_is_empty)]
pub trait DataLoader<I, L> {
	fn load(&mut self, idx: usize) -> Result<(I, L)>;

	fn len(&self) -> Option<usize> {
		None
	}
}

pub struct IterableDataLoader<T, I, L, C: Fn(&T) -> Result<(I, L)>> {
	items: Box<[T]>,
	collator: C
}

impl<T, I, L, C: Fn(&T) -> Result<(I, L)>> DataLoader<I, L> for IterableDataLoader<T, I, L, C> {
	fn load(&mut self, idx: usize) -> Result<(I, L)> {
		(self.collator)(&self.items[idx])
	}

	fn len(&self) -> Option<usize> {
		Some(self.items.len())
	}
}

pub fn iterable_data_loader<T, I, L, C: Fn(&T) -> Result<(I, L)>>(iterable: impl Iterator<Item = T>, collator: C) -> IterableDataLoader<T, I, L, C> {
	IterableDataLoader {
		items: iterable.collect::<Vec<T>>().into_boxed_slice(),
		collator
	}
}

impl<I, L, F: FnMut(usize) -> Result<(I, L)>> DataLoader<I, L> for F {
	fn load(&mut self, idx: usize) -> Result<(I, L)> {
		(self)(idx)
	}

	fn len(&self) -> Option<usize> {
		None
	}
}

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
	loader: Box<dyn DataLoader<I, L>>,
	eval_loader: Option<Box<dyn DataLoader<I, L>>>,
	eval_strategy: EvaluationStrategy,
	ckpt_strategy: CheckpointStrategy,
	ckpt_path: PathBuf,
	lr: f32,
	max_saved_ckpts: usize,
	gradient_accumulation_steps: usize,
	max_steps: usize,
	max_eval_steps: usize
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
			max_eval_steps: usize::MAX
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

	pub fn with_max_eval_steps(mut self, steps: usize) -> Self {
		self.max_eval_steps = steps;
		self
	}

	pub fn with_gradient_accumulation(mut self, steps: usize) -> Self {
		self.gradient_accumulation_steps = steps;
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
}

impl super::Trainer {
	pub fn train<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize>(
		&self,
		mut args: TrainingArguments<I, L, NI, NL>
	) -> crate::Result<()> {
		let optimizer = self.optimizer();
		optimizer.set_lr(args.lr)?;

		let mut saved_ckpts = VecDeque::new();
		let mut global_step = 0;
		for (iter_step, _) in (0..args.max_steps).enumerate() {
			let epoch = iter_step / args.loader.len().unwrap_or(usize::MAX);
			let (inputs, labels) = args.loader.load(iter_step)?;
			let (inputs, labels) = (inputs.into(), labels.into());

			let outputs = self.step(inputs, labels)?;
			let loss = outputs[0].try_extract_scalar::<f32>()?;
			println!("epoch={epoch} step={global_step} loss={loss}");

			if iter_step % args.gradient_accumulation_steps == 0 {
				optimizer.step()?;
				optimizer.reset_grad()?;
				global_step += 1;
			}

			if args.ckpt_strategy.should_fire(global_step, iter_step, args.loader.len()) {
				if !args.ckpt_path.exists() {
					let _ = fs::create_dir_all(&args.ckpt_path);
				}

				let ckpt_path = args.ckpt_path.join(format!("epoch={epoch},step={global_step}.ortckpt"));
				self.checkpoint().save(&ckpt_path, true)?;

				saved_ckpts.push_front(ckpt_path.clone());
				while saved_ckpts.len() > args.max_saved_ckpts {
					let Some(old_ckpt) = saved_ckpts.pop_back() else {
						break;
					};
					let _ = fs::remove_file(old_ckpt);
				}
			}

			if args
				.eval_strategy
				.should_fire(global_step, iter_step, args.eval_loader.as_ref().and_then(|d| d.len()))
			{
				let eval_loss = self.eval_inner(&mut args)?;
				println!("eval_loss={eval_loss}");
			}
		}
		Ok(())
	}

	pub(crate) fn eval_inner<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize>(
		&self,
		args: &mut TrainingArguments<I, L, NI, NL>
	) -> crate::Result<f32> {
		let Some(eval_loader) = &mut args.eval_loader else {
			return Ok(0.0);
		};

		let mut total_loss = 0.0;
		for step in 0..args.max_eval_steps.min(eval_loader.len().unwrap_or(usize::MAX)) {
			let (inputs, labels) = eval_loader.load(step)?;
			let (inputs, labels) = (inputs.into(), labels.into());

			let outputs = self.eval_step(inputs, labels)?;
			let loss = outputs[0].try_extract_scalar::<f32>()?;
			total_loss = (total_loss * (step as f32) + loss) / (step as f32 + 1.);
		}

		Ok(total_loss)
	}
}
