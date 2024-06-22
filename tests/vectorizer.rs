#![cfg(not(target_arch = "aarch64"))]

use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use ort::{inputs, GraphOptimizationLevel, Session, Tensor};
use test_log::test;

#[test]
fn vectorizer() -> ort::Result<()> {
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("vectorizer.onnx"))
		.expect("Could not load model");

	let metadata = session.metadata()?;
	assert_eq!(metadata.producer()?, "skl2onnx");
	assert_eq!(metadata.description()?, "test description");
	assert_eq!(metadata.custom("custom_key")?.as_deref(), Some("custom_value"));

	let array = ndarray::CowArray::from(ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap());

	// Just one input
	let input_tensor_values = inputs![Tensor::from_string_array(&array)?]?;

	// Perform the inference
	let outputs = session.run(input_tensor_values)?;
	assert_eq!(outputs[0].try_extract_tensor::<f32>()?, ArrayD::from_shape_vec(IxDyn(&[1, 9]), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap());

	Ok(())
}
