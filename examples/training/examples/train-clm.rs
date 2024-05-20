use ndarray::{ArrayView0, Array1, Array2};
use ort::{Allocator, Checkpoint, SessionBuilder, Trainer};

fn main() -> ort::Result<()> {
	ort::init().commit()?;

	let trainer = Trainer::new(
		SessionBuilder::new()?,
		Allocator::default(),
		Checkpoint::load("tools/train-data/mini-clm/checkpoint")?,
		"tools/train-data/mini-clm/training_model.onnx",
		"tools/train-data/mini-clm/eval_model.onnx",
		"tools/train-data/mini-clm/optimizer_model.onnx"
	)?;

	let optimizer = trainer.optimizer();
	optimizer.set_lr(1e-4)?;

	let inputs = Array2::<i64>::from_shape_vec([1, 5], vec![0, 1, 2, 3, 4]).unwrap();
	let labels = Array1::<i64>::from_shape_vec([5], vec![1, 2, 3, 4, 5]).unwrap();

	for _ in 0..50 {
		let outputs = trainer.step(ort::inputs![inputs.view()]?, ort::inputs![labels.view()]?)?;
		let loss: ArrayView0<f32> = outputs[0].try_extract_tensor::<f32>()?.into_dimensionality().unwrap();
		println!("{}", loss.into_scalar());
		optimizer.step()?;
		optimizer.reset_grad()?;
	}

	Ok(())
}
