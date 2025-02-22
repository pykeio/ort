use std::path::Path;

use image::{ImageBuffer, Luma, Pixel, imageops::FilterType};
use ort::{
	inputs,
	session::{Session, builder::GraphOptimizationLevel},
	tensor::ArrayExtensions,
	value::TensorRef
};

#[test]
fn mnist_5() -> ort::Result<()> {
	const IMAGE_TO_LOAD: &str = "mnist_5.jpg";

	ort::set_api(ort_tract::api());

	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)?
		.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx")
		.expect("Could not download model from file");

	// Load image and resize to model's shape, converting to RGB format
	let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(
		Path::new(env!("CARGO_MANIFEST_DIR"))
			.parent()
			.unwrap()
			.parent()
			.unwrap()
			.join("tests")
			.join("data")
			.join(IMAGE_TO_LOAD)
	)
	.unwrap()
	.resize(28, 28, FilterType::Nearest)
	.to_luma8();

	let array = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();

		// range [0, 255] -> range [0, 1]
		(channels[c] as f32) / 255.0
	});

	// Perform the inference
	let outputs = session.run(inputs![TensorRef::from_array_view(&array)?])?;

	let mut probabilities: Vec<(usize, f32)> = outputs[0]
		.try_extract_tensor()?
		.softmax(ndarray::Axis(1))
		.iter()
		.copied()
		.enumerate()
		.collect::<Vec<_>>();

	// Sort probabilities so highest is at beginning of vector.
	probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

	assert_eq!(probabilities[0].0, 5, "Expecting class for {} is '5' (not {})", IMAGE_TO_LOAD, probabilities[0].0);

	Ok(())
}
