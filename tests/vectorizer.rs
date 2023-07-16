use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ort::{environment::Environment, value::Value, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder};
use test_log::test;

#[test]
#[cfg(not(target_arch = "aarch64"))]
fn vectorizer() -> OrtResult<()> {
	let environment = Environment::default().into_arc();

	#[cfg(not(feature = "cuda"))]
	assert_eq!(ExecutionProvider::CUDA(Default::default()).is_available(), false);
	assert!(ExecutionProvider::CPU(Default::default()).is_available());

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("vectorizer.onnx"))
		.expect("Could not load model");

	let metadata = session.metadata()?;
	assert_eq!(metadata.producer()?, "skl2onnx");
	assert_eq!(metadata.description()?, "test description");
	assert_eq!(metadata.custom("custom_key")?.as_deref(), Some("custom_value"));

	let array = ndarray::CowArray::from(ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap().into_dyn());

	// Just one input
	let input_tensor_values = vec![Value::from_array(session.allocator(), &array)?];

	// Perform the inference
	let outputs = session.run(input_tensor_values)?;
	assert_eq!(
		*outputs[0].try_extract::<f32>()?.view(),
		ArrayD::from_shape_vec(IxDyn(&[1, 9]), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			.unwrap()
			.view()
	);

	Ok(())
}
