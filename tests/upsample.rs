use std::path::Path;

use image::RgbImage;
use ndarray::{Array, CowArray, IxDyn};
use ort::{inputs, Environment, GraphOptimizationLevel, LoggingLevel, OrtOwnedTensor, OrtResult, SessionBuilder};
use test_log::test;

fn load_input_image<P: AsRef<Path>>(name: P) -> RgbImage {
	// Load image, converting to RGB format
	image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name))
		.unwrap()
		.to_rgb8()
}

fn convert_image_to_cow_array(img: &RgbImage) -> CowArray<'_, f32, IxDyn> {
	let array = Array::from_shape_vec((1, img.height() as usize, img.width() as usize, 3), img.to_vec())
		.unwrap()
		.map(|x| *x as f32 / 255.0)
		.into_dyn();
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
fn upsample() -> OrtResult<()> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	let environment = Environment::builder()
		.with_name("integration_test")
		.with_log_level(LoggingLevel::Warning)
		.build()?
		.into_arc();

	let session_data =
		std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("upsample.onnx")).expect("Could not open model from file");
	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_from_memory(&session_data)
		.expect("Could not read model from memory");

	let metadata = session.metadata()?;
	assert_eq!(metadata.name()?, "tf2onnx");
	assert_eq!(metadata.producer()?, "tf2onnx");

	assert_eq!(session.inputs[0].collect::<Vec<_>>(), [None, None, None, Some(3)]);
	assert_eq!(session.outputs[0].collect::<Vec<_>>(), [None, None, None, Some(3)]);

	// Load image, converting to RGB format
	let image_buffer = load_input_image(IMAGE_TO_LOAD);
	let array = convert_image_to_cow_array(&image_buffer);

	// Just one input
	let input_tensor_values = inputs![&array]?;

	// Perform the inference
	let outputs = session.run(input_tensor_values)?;

	assert_eq!(outputs.len(), 1);
	let output: OrtOwnedTensor<f32> = outputs[0].extract_tensor()?;

	// The image should have doubled in size
	assert_eq!(output.view().shape(), [1, 448, 448, 3]);

	Ok(())
}

/// The upsample.ort can be produced by
/// ```shell
/// python -m onnxruntime.tools.convert_onnx_models_to_ort tests/data/upsample.onnx
/// ```
#[test]
fn upsample_with_ort_model() -> OrtResult<()> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	let environment = Environment::builder()
		.with_name("integration_test")
		.with_log_level(LoggingLevel::Warning)
		.build()?
		.into_arc();

	let session_data =
		std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("upsample.ort")).expect("Could not open model from file");
	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_from_memory_directly(&session_data) // Zero-copy.
		.expect("Could not read model from memory");

	assert_eq!(session.inputs[0].collect::<Vec<_>>(), [None, None, None, Some(3)]);
	assert_eq!(session.outputs[0].collect::<Vec<_>>(), [None, None, None, Some(3)]);

	// Load image, converting to RGB format
	let image_buffer = load_input_image(IMAGE_TO_LOAD);
	let array = convert_image_to_cow_array(&image_buffer);

	// Perform the inference
	let outputs = session.run(inputs![&array]?)?;

	assert_eq!(outputs.len(), 1);
	let output: OrtOwnedTensor<f32> = outputs[0].extract_tensor()?;

	// The image should have doubled in size
	assert_eq!(output.view().shape(), [1, 448, 448, 3]);

	Ok(())
}
