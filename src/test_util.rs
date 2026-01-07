#[cfg(feature = "ndarray")]
pub mod mnist {
	use image::{ImageBuffer, Luma, Pixel};
	use ndarray::{Array4, Axis};

	use crate::{
		Result,
		tensor::ArrayExtensions,
		value::{TensorValueTypeMarker, Value}
	};

	pub const MODEL_URL: &str = "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/mnist.onnx";

	pub fn get_image() -> Array4<f32> {
		let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open("tests/data/mnist_5.jpg").expect("failed to load image").to_luma8();
		ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();
			(channels[c] as f32) / 255.0
		})
	}

	pub fn extract_probabilities<T: TensorValueTypeMarker>(output: &Value<T>) -> Result<Vec<(usize, f32)>> {
		let mut probabilities: Vec<(usize, f32)> = output
			.try_extract_array()?
			.softmax(Axis(1))
			.iter()
			.copied()
			.enumerate()
			.collect::<Vec<_>>();
		probabilities.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
		Ok(probabilities)
	}
}
