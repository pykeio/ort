use std::path::Path;

use image::{imageops::FilterType, ImageBuffer, Luma, Pixel};
use ort::{
	download::vision::DomainBasedImageClassification, inputs, Environment, GraphOptimizationLevel, LoggingLevel, NdArrayExtensions, OrtOwnedTensor, OrtResult,
	SessionBuilder
};
use test_log::test;

#[test]
fn mnist_5() -> OrtResult<()> {
	const IMAGE_TO_LOAD: &str = "mnist_5.jpg";

	let environment = Environment::builder()
		.with_name("integration_test")
		.with_log_level(LoggingLevel::Warning)
		.build()?
		.into_arc();

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_downloaded(DomainBasedImageClassification::Mnist)
		.expect("Could not download model from file");

	let metadata = session.metadata()?;
	assert_eq!(metadata.name()?, "CNTKGraph");
	assert_eq!(metadata.producer()?, "CNTK");

	let input0_shape: Vec<usize> = session.inputs[0].map(|d| d.unwrap()).collect();
	let output0_shape: Vec<usize> = session.outputs[0].map(|d| d.unwrap()).collect();

	assert_eq!(input0_shape, [1, 1, 28, 28]);
	assert_eq!(output0_shape, [1, 10]);

	// Load image and resize to model's shape, converting to RGB format
	let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(IMAGE_TO_LOAD))
		.unwrap()
		.resize(input0_shape[2] as u32, input0_shape[3] as u32, FilterType::Nearest)
		.to_luma8();

	let array = ndarray::CowArray::from(
		ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();

			// range [0, 255] -> range [0, 1]
			(channels[c] as f32) / 255.0
		})
		.into_dyn()
	);

	// Perform the inference
	let outputs = session.run(inputs![&array]?)?;

	let output: OrtOwnedTensor<_> = outputs[0].extract_tensor()?;
	let mut probabilities: Vec<(usize, f32)> = output.view().softmax(ndarray::Axis(1)).iter().copied().enumerate().collect::<Vec<_>>();

	// Sort probabilities so highest is at beginning of vector.
	probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

	assert_eq!(probabilities[0].0, 5, "Expecting class for {} is '5' (not {})", IMAGE_TO_LOAD, probabilities[0].0);

	Ok(())
}
