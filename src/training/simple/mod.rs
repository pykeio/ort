use std::{collections::VecDeque, fs};

use crate::{error::Result, session::input::SessionInputs, training::Trainer};

mod dataloader;
pub use self::dataloader::{DataLoader, IterableDataLoader, iterable_data_loader};
mod args;
pub use self::args::{CheckpointStrategy, EvaluationStrategy, TrainingArguments};
mod callbacks;
pub use self::callbacks::{TrainerCallbacks, TrainerControl, TrainerState};

macro_rules! callback {
	($which:ident($self:expr, $optimizer:expr, $args:expr, $state:expr)) => {
		let mut halt = false;
		for cb in &mut $args.callbacks {
			let mut control = TrainerControl::new($self);
			cb.$which(&$state, &mut control)?;
			halt = halt || control.halt;
			if let Some(lr) = control.lr {
				$optimizer.set_lr(lr)?;
			}
		}
		if halt {
			return $self.handle_halt(&mut $args.callbacks, &$state);
		}
	};
	($which:ident($self:expr, $optimizer:expr, $args:expr, $state:expr), $($addt:expr),*) => {
		let mut halt = false;
		for cb in &mut $args.callbacks {
			let mut control = TrainerControl::new($self);
			cb.$which($($addt,)* &$state, &mut control)?;
			halt = halt || control.halt;
			if let Some(lr) = control.lr {
				$optimizer.set_lr(lr)?;
			}
		}
		if halt {
			return $self.handle_halt(&mut $args.callbacks, &$state);
		}
	};
}

impl Trainer {
	pub fn train<I: Into<SessionInputs<'static, 'static, NI>>, L: Into<SessionInputs<'static, 'static, NL>>, const NI: usize, const NL: usize>(
		&self,
		mut args: TrainingArguments<I, L, NI, NL>
	) -> Result<()> {
		let mut optimizer = self.optimizer();
		optimizer.set_lr(args.lr)?;

		let mut saved_ckpts = VecDeque::new();
		let mut state = TrainerState::new(&args);
		let mut last_epoch = -1.0;
		for (iter_step, _) in (0..args.max_steps).enumerate() {
			state.iter_step = iter_step;
			state.epoch = args.loader.len().map(|dl_len| iter_step as f32 / dl_len as f32);

			if let Some(epoch) = state.epoch {
				if epoch.trunc() != last_epoch {
					callback!(epoch(self, optimizer, args, state));
				}

				last_epoch = epoch.trunc();
			}

			let (inputs, labels) = args.loader.load(iter_step)?;
			let (inputs, labels) = (inputs.into(), labels.into());

			let outputs = self.step(inputs, labels)?;
			let loss = outputs[0].try_extract_scalar::<f32>()?;
			callback!(train_step(self, optimizer, args, state), loss);

			if iter_step % args.gradient_accumulation_steps == 0 {
				optimizer.step()?;
				optimizer.reset_grad()?;
				state.global_step += 1;
				callback!(optimizer_step(self, optimizer, args, state), loss);
			}

			if args.ckpt_strategy.should_fire(state.global_step, iter_step, args.loader.len()) {
				if !args.ckpt_path.exists() {
					let _ = fs::create_dir_all(&args.ckpt_path);
				}

				let ckpt_path =
					args.ckpt_path
						.join(format!("epoch={},step={}.ortckpt", state.epoch.map(f32::trunc).unwrap_or(0.0) as usize, state.global_step));
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
				.should_fire(state.global_step, iter_step, args.eval_loader.as_ref().and_then(|d| d.len()))
			{
				callback!(eval_begin(self, optimizer, args, state));
				let eval_loss = self.eval_inner(&mut args)?;
				callback!(eval_end(self, optimizer, args, state), eval_loss);
			}
		}
		Ok(())
	}

	fn handle_halt(&self, cbs: &mut Vec<Box<dyn TrainerCallbacks>>, state: &TrainerState) -> Result<()> {
		for cb in cbs {
			let mut control = TrainerControl::new(self);
			cb.end(state, &mut control)?;
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
