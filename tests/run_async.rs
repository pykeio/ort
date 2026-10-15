use ort::{
	error::ErrorCode,
	memory::Allocator,
	session::{OutputSelector, RunOptions, Session},
	value::Tensor
};

#[test]
fn run_async_rejects_extra_inputs() -> ort::Result<()> {
	let env = ort::test_util::test_env();
	let mut session = Session::builder(env)?.commit_from_file("tests/data/upsample.onnx")?;
	let input = Tensor::<f32>::new(&Allocator::default(), [1i64, 64, 64, 3])?;
	let options = RunOptions::new()?;

	let result = session.run_async(ort::inputs![input.clone(), input.clone(), input.clone()], &options);
	let err = result.err().expect("run_async should reject more inputs than the model accepts");
	assert_eq!(err.code(), ErrorCode::InvalidArgument);
	Ok(())
}

#[test]
fn run_async_error_and_success() -> ort::Result<()> {
	let env = ort::test_util::test_env();
	let input = Tensor::<f32>::new(&Allocator::default(), [1i64, 64, 64, 3])?;
	let options = RunOptions::new()?;

	// ONNX Runtime refuses to run async with a single intra-op thread, which fails before the callback is set up.
	let mut session = Session::builder(env)?
		.with_intra_threads(1)?
		.commit_from_file("tests/data/upsample.onnx")?;
	assert!(session.run_async(ort::inputs![input.clone()], &options).is_err());

	let mut session = Session::builder(env)?
		.with_intra_threads(2)?
		.commit_from_file("tests/data/upsample.onnx")?;
	let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
	for _ in 0..3 {
		let outputs = rt.block_on(session.run_async(ort::inputs![input.clone()], &options)?)?;
		assert_eq!(&**outputs[0].shape(), &[1, 128, 128, 3]);
	}
	Ok(())
}

#[test]
fn run_async_preallocated_output() -> ort::Result<()> {
	let env = ort::test_util::test_env();
	let mut session = Session::builder(env)?
		.with_intra_threads(2)?
		.commit_from_file("tests/data/upsample.onnx")?;
	let input = Tensor::<f32>::new(&Allocator::default(), [1i64, 64, 64, 3])?;
	let output0 = session.outputs()[0].name().to_string();
	let options = RunOptions::new()?.with_outputs(
		OutputSelector::no_default()
			.with(&output0)
			.preallocate(&output0, Tensor::<f32>::new(&Allocator::default(), [1_usize, 128, 128, 3])?)
	);

	let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
	for _ in 0..3 {
		let outputs = rt.block_on(session.run_async(ort::inputs![input.clone()], &options)?)?;
		assert_eq!(&**outputs[0].shape(), &[1, 128, 128, 3]);
	}
	drop(options);
	Ok(())
}
