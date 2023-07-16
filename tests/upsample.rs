use std::path::Path;

use image::{ImageBuffer, Pixel, Rgb};
use ort::{
	environment::Environment, execution_providers::CPUExecutionProviderOptions, tensor::OrtOwnedTensor, value::Value, ExecutionProvider,
	GraphOptimizationLevel, LoggingLevel, OrtResult, SessionBuilder
};
use test_log::test;

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
		.with_execution_providers([ExecutionProvider::CPU(CPUExecutionProviderOptions { use_arena: true })])
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

	assert_eq!(session.inputs[0].dimensions().collect::<Vec<_>>(), [None, None, None, Some(3)]);
	assert_eq!(session.outputs[0].dimensions().collect::<Vec<_>>(), [None, None, None, Some(3)]);

	// Load image, converting to RGB format
	let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(IMAGE_TO_LOAD))
		.unwrap()
		.to_rgb8();

	let array = ndarray::CowArray::from(
		ndarray::Array::from_shape_fn((1, 224, 224, 3), |(_, j, i, c)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();

			// range [0, 255] -> range [0, 1]
			(channels[c] as f32) / 255.0
		})
		.into_dyn()
	);

	// Just one input
	let input_tensor_values = vec![Value::from_array(session.allocator(), &array)?];

	// Perform the inference
	let outputs: Vec<Value> = session.run(input_tensor_values)?;

	assert_eq!(outputs.len(), 1);
	let output: OrtOwnedTensor<'_, f32, ndarray::Dim<ndarray::IxDynImpl>> = outputs[0].try_extract()?;

	// The image should have doubled in size
	assert_eq!(output.view().shape(), [1, 448, 448, 3]);

	Ok(())
}
