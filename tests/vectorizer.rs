use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ort::{inputs, Environment, GraphOptimizationLevel, OrtResult, SessionBuilder, Value};
use test_log::test;

#[test]
#[cfg(not(target_arch = "aarch64"))]
fn vectorizer() -> OrtResult<()> {
	let environment = Environment::default().into_arc();

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("vectorizer.onnx"))
		.expect("Could not load model");

	let metadata = session.metadata()?;
	assert_eq!(metadata.producer()?, "skl2onnx");
	assert_eq!(metadata.description()?, "test description");
	assert_eq!(metadata.custom("custom_key")?.as_deref(), Some("custom_value"));

	let array = ndarray::CowArray::from(ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap());

	// Just one input
	let input_tensor_values = inputs![Value::from_string_array(session.allocator(), &array)?]?;

	// Perform the inference
	let outputs = session.run(input_tensor_values)?;
	assert_eq!(
		*outputs[0].extract_tensor::<f32>()?.view(),
		ArrayD::from_shape_vec(IxDyn(&[1, 9]), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
			.unwrap()
			.view()
	);

	Ok(())
}
