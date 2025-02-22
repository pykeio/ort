#![allow(clippy::manual_retain)]

use std::{ops::Mul, path::Path};

use image::{GenericImageView, ImageBuffer, Rgba, imageops::FilterType};
use ndarray::Array;
use ort::{execution_providers::CUDAExecutionProvider, inputs, session::Session, value::TensorRef};
use show_image::{AsImageView, WindowOptions, event};

#[show_image::main]
fn main() -> ort::Result<()> {
	tracing_subscriber::fmt::init();

	ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let mut session =
		Session::builder()?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/modnet_photographic_portrait_matting.onnx")?;

	let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("photo.jpg")).unwrap();
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

	let outputs = session.run(inputs!["input" => TensorRef::from_array_view(input.view())?])?;

	let output = outputs["output"].try_extract_tensor::<f32>()?;

	// convert to 8-bit
	let output = output.mul(255.0).map(|x| *x as u8);
	let (output, _) = output.into_raw_vec_and_offset();

	// change rgb to rgba
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

	let window = show_image::context()
		.run_function_wait(move |context| -> Result<_, String> {
			let mut window = context
				.create_window("ort + modnet", WindowOptions {
					size: Some([img_width, img_height]),
					..WindowOptions::default()
				})
				.map_err(|e| e.to_string())?;
			window.set_image("photo", &output.as_image_view().map_err(|e| e.to_string())?);
			Ok(window.proxy())
		})
		.unwrap();

	for event in window.event_channel().unwrap() {
		if let event::WindowEvent::KeyboardInput(event) = event {
			if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
				break;
			}
		}
	}

	Ok(())
}
