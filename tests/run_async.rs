use ndarray::Array4;
use ort::{
	error::ErrorCode,
	session::{RunOptions, Session},
	value::TensorRef
};

#[test]
fn run_async_rejects_extra_inputs() -> ort::Result<()> {
	let env = ort::test_util::test_env();
	let mut session = Session::builder(env)?.commit_from_file("tests/data/upsample.onnx")?;
	let input = Array4::<f32>::zeros((1, 64, 64, 3));
	let options = RunOptions::new()?;

	let result = session
		.run_async(ort::inputs![TensorRef::from_array_view(&input)?, TensorRef::from_array_view(&input)?, TensorRef::from_array_view(&input)?], &options);
	let err = result.err().expect("run_async should reject more inputs than the model accepts");
	assert_eq!(err.code(), ErrorCode::InvalidArgument);
	Ok(())
}

#[test]
fn run_async_error_and_success() -> ort::Result<()> {
	let env = ort::test_util::test_env();
	let input = Array4::<f32>::zeros((1, 64, 64, 3));
	let options = RunOptions::new()?;

	// ONNX Runtime refuses to run async with a single intra-op thread, which fails before the callback is set up.
	let mut session = Session::builder(env)?
		.with_intra_threads(1)?
		.commit_from_file("tests/data/upsample.onnx")?;
	assert!(session.run_async(ort::inputs![TensorRef::from_array_view(&input)?], &options).is_err());

	let mut session = Session::builder(env)?
		.with_intra_threads(2)?
		.commit_from_file("tests/data/upsample.onnx")?;
	let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
	for _ in 0..3 {
		let outputs = rt.block_on(session.run_async(ort::inputs![TensorRef::from_array_view(&input)?], &options)?)?;
		assert_eq!(&**outputs[0].shape(), &[1, 128, 128, 3]);
	}
	Ok(())
}
