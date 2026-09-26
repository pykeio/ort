use std::{path::Path, sync::Arc};

use cutile::prelude::*;
use image::{GenericImageView, ImageBuffer, Rgba, imageops::FilterType};
use ort::{
	ep,
	memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
	session::Session,
	value::{Shape, TensorRefMut}
};
use show_image::{AsImageView, WindowOptions, event};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const SIZE: usize = 512;
const TILE: usize = 1024;

// A cuTile kernel that normalizes raw pixel values from [0, 255] to [-1, 1] on the GPU.
#[cutile::module]
mod kernels {
	use cutile::core::*;

	#[cutile::entry()]
	fn normalize<const S: [i32; 1]>(y: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>, scale: f32, bias: f32) {
		let tx = x.load_like(y);
		y.store(tx * scale.broadcast(y.shape()) + bias.broadcast(y.shape()));
	}
}

#[show_image::main]
fn main() -> anyhow::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	#[rustfmt::skip]
	let env = ort::init()
		.with_execution_providers([
			ep::CUDA::default()
				.build()
				// exit the program with an error if the CUDA EP fails to register
				.error_on_failure()
		])
		.build()?;

	let mut session =
		Session::builder(&env)?.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/modnet_photographic_portrait_matting.onnx")?;

	let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("photo.jpg")).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(SIZE as u32, SIZE as u32, FilterType::Triangle);

	// Lay out the raw pixels as NCHW on the host; normalization happens on the GPU.
	let mut pixels = vec![0.0f32; 3 * SIZE * SIZE];
	for (x, y, pixel) in img.pixels() {
		let i = y as usize * SIZE + x as usize;
		let [r, g, b, _] = pixel.0;
		pixels[i] = r as f32;
		pixels[SIZE * SIZE + i] = g as f32;
		pixels[2 * SIZE * SIZE + i] = b as f32;
	}

	let raw: Arc<cutile::tensor::Tensor<f32>> = api::copy_host_vec_to_device(&Arc::new(pixels)).sync()?.into();
	let input = api::zeros::<f32>(&[3 * SIZE * SIZE]).sync()?.partition([TILE]);
	let (input, _raw, _, _) = kernels::normalize(input, raw, 1.0 / 127.5, -1.0).sync()?;
	let input = input.unpartition();

	// Hand the cuTile owned device buffer straight to ONNX Runtime, no copy back to the host.
	// `input` must stay alive until `session.run` returns.
	let tensor: TensorRefMut<'_, f32> = unsafe {
		TensorRefMut::from_raw(
			MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
			(input.device_pointer().cu_deviceptr() as usize as *mut ()).cast(),
			Shape::from([1i64, 3, SIZE as i64, SIZE as i64])
		)
		.unwrap()
	};
	let outputs = session.run(ort::inputs![tensor])?;

	let (_, output) = outputs["output"].try_extract_tensor::<f32>()?;

	// convert to 8-bit rgba
	let output_img = ImageBuffer::from_fn(SIZE as u32, SIZE as u32, |x, y| {
		let v = (output[x as usize + y as usize * SIZE] * 255.0) as u8;
		Rgba([v, v, v, 255])
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
				.create_window(
					"ort + cutile + modnet",
					WindowOptions {
						size: Some([img_width, img_height]),
						..WindowOptions::default()
					}
				)
				.map_err(|e| e.to_string())?;
			window.set_image("photo", &output.as_image_view().map_err(|e| e.to_string())?);
			Ok(window.proxy())
		})
		.unwrap();

	for event in window.event_channel().unwrap() {
		if let event::WindowEvent::KeyboardInput(event) = event
			&& event.input.key_code == Some(event::VirtualKeyCode::Escape)
			&& event.input.state.is_pressed()
		{
			break;
		}
	}

	Ok(())
}
