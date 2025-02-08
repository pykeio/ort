use std::path::Path;

use image::RgbImage;
use ndarray::{Array, ArrayViewD, CowArray, Ix4};
use ort::{
	inputs,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef
};
use test_log::test;

fn load_input_image<P: AsRef<Path>>(name: P) -> RgbImage {
	// Load image, converting to RGB format
	image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name))
		.unwrap()
		.to_rgb8()
}

fn convert_image_to_cow_array(img: &RgbImage) -> CowArray<'_, f32, Ix4> {
	let array = Array::from_shape_vec((1, img.height() as usize, img.width() as usize, 3), img.to_vec())
		.unwrap()
		.map(|x| *x as f32 / 255.0);
	CowArray::from(array)
}

/// This test verifies that dynamically sized inputs and outputs work. It loads and runs
/// upsample.onnx, which was produced via:
///
/// ```python
/// import subprocess
/// from tensorflow import keras
///
/// m = keras.Sequential([
/// 	keras.layers.UpSampling2D(size=2)
/// ])
/// m.build(input_shape=(None, None, None, 3))
/// m.summary()
/// m.save('saved_model')
///
/// subprocess.check_call([
/// 	'python', '-m', 'tf2onnx.convert',
/// 	'--saved-model', 'saved_model',
/// 	'--opset', '12',
/// 	'--output', 'upsample.onnx'
/// ])
/// ```
#[test]
fn upsample() -> ort::Result<()> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	ort::init().with_name("integration_test").commit()?;

	let session_data =
		std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("upsample.onnx")).expect("Could not open model from file");
	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_memory(&session_data)
		.expect("Could not read model from memory");

	{
		let metadata = session.metadata()?;
		assert_eq!(metadata.name()?, "tf2onnx");
		assert_eq!(metadata.producer()?, "tf2onnx");

		assert_eq!(session.inputs[0].input_type.tensor_dimensions().expect("input0 to be a tensor type"), &[-1, -1, -1, 3]);
		assert_eq!(session.outputs[0].output_type.tensor_dimensions().expect("output0 to be a tensor type"), &[-1, -1, -1, 3]);
	}

	// Load image, converting to RGB format
	let image_buffer = load_input_image(IMAGE_TO_LOAD);
	let array = convert_image_to_cow_array(&image_buffer);

	// Perform the inference
	let outputs = session.run(inputs![TensorRef::from_array_view(&array)?])?;

	assert_eq!(outputs.len(), 1);
	let output: ArrayViewD<f32> = outputs[0].try_extract_tensor()?;

	// The image should have doubled in size
	assert_eq!(output.shape(), [1, 448, 448, 3]);

	Ok(())
}

/// The upsample.ort can be produced by
/// ```shell
/// python -m onnxruntime.tools.convert_onnx_models_to_ort tests/data/upsample.onnx
/// ```
#[test]
fn upsample_with_ort_model() -> ort::Result<()> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	ort::init().with_name("integration_test").commit()?;

	let session_data =
		std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("upsample.ort")).expect("Could not open model from file");
	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_memory_directly(&session_data) // Zero-copy.
		.expect("Could not read model from memory");

	assert_eq!(session.inputs[0].input_type.tensor_dimensions().expect("input0 to be a tensor type"), &[-1, -1, -1, 3]);
	assert_eq!(session.outputs[0].output_type.tensor_dimensions().expect("output0 to be a tensor type"), &[-1, -1, -1, 3]);

	// Load image, converting to RGB format
	let image_buffer = load_input_image(IMAGE_TO_LOAD);
	let array = convert_image_to_cow_array(&image_buffer);

	// Perform the inference
	let outputs = session.run(inputs![TensorRef::from_array_view(&array)?])?;

	assert_eq!(outputs.len(), 1);
	let output: ArrayViewD<f32> = outputs[0].try_extract_tensor()?;

	// The image should have doubled in size
	assert_eq!(output.shape(), [1, 448, 448, 3]);

	Ok(())
}
