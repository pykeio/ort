#![allow(clippy::manual_retain)]

use std::ops::Mul;

use image::{imageops::FilterType, GenericImageView, ImageBuffer, Rgba};
use ndarray::Array;
use ort::{inputs, CUDAExecutionProvider, Session};
fn main() -> ort::Result<()> {
	// read args from command line
	let model_path = std::env::args().nth(1).unwrap();
	let input_image_path = std::env::args().nth(2).unwrap();
	let output_image_path = std::env::args().nth(3).unwrap();
	tracing_subscriber::fmt::init();

	ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let original_img = image::open(input_image_path).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(512, 512, FilterType::Triangle);
	let mut input = Array::zeros((1, 3, 512, 512));
	for pixel in img.pixels() {
		let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = (r as f32 - 127.5) / 127.5;
		input[[0, 1, y, x]] = (g as f32 - 127.5) / 127.5;
		input[[0, 2, y, x]] = (b as f32 - 127.5) / 127.5;
	}
	let model = Session::builder()?.with_model_from_file(model_path)?;
	let outputs = model.run(inputs!["input" => input.view()]?)?;
	let binding = outputs["output"].extract_tensor::<f32>().unwrap();
	let output = binding.view();
	let output = output.mul(255.0).map(|x| *x as u8);
	let output = output.into_raw_vec();
	// change to rgba
	let output_img = ImageBuffer::from_fn(512, 512, |x, y| {
		let i = (x + y * 512) as usize;
		Rgba([output[i], output[i], output[i], 255])
	});
	let mut output = image::imageops::resize(&output_img, img_width, img_height, FilterType::Triangle);
	output.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
		let origin = original_img.get_pixel(x, y);
		pixel.0[3] = pixel.0[0];
		pixel.0[0] = origin.0[0];
		pixel.0[1] = origin.0[1];
		pixel.0[2] = origin.0[2];
	});
	image::save_buffer(output_image_path, &output, img_width, img_height, image::ColorType::Rgba8).unwrap();
	Ok(())
}
