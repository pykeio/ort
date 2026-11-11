use core::fmt::Write;
use std::{path::PathBuf, sync::OnceLock};

use ort::{Error, environment::Environment};
use ureq::{
	config::Config,
	tls::{RootCerts, TlsConfig, TlsProvider}
};

pub fn test_env() -> &'static Environment {
	static TEST_ENV: OnceLock<Environment> = OnceLock::new();
	TEST_ENV.get_or_init(|| ort::init().build().expect("failed to create test environment"))
}

pub fn download_model(url: &str) -> ort::Result<PathBuf> {
	let mut download_dir = ort::sys::internal::dirs::cache_dir()
		.expect("could not determine cache directory")
		.join("models");
	if std::fs::create_dir_all(&download_dir).is_err() {
		download_dir = std::env::current_dir().expect("Failed to obtain current working directory");
	}

	let model_filename = <sha2::Sha256 as sha2::Digest>::digest(url).into_iter().fold(String::new(), |mut s, b| {
		let _ = write!(&mut s, "{:02x}", b);
		s
	});
	let model_filepath = download_dir.join(&model_filename);
	if model_filepath.exists() {
		Ok(model_filepath)
	} else {
		let agent = Config::builder()
			.tls_config(
				TlsConfig::builder()
					.provider(TlsProvider::NativeTls)
					.root_certs(RootCerts::PlatformVerifier)
					.build()
			)
			.build()
			.new_agent();

		let resp = agent.get(url).call().map_err(|e| Error::new(format!("Error downloading to file: {e}")))?;

		let len = resp
			.headers()
			.get("Content-Length")
			.and_then(|h| h.to_str().ok())
			.and_then(|s| s.parse::<usize>().ok())
			.expect("Missing Content-Length header");

		let mut reader = resp.into_body().into_with_config().limit(u64::MAX).reader();
		let temp_filepath = download_dir.join(format!("tmp_{}.{model_filename}", ort::sys::internal::random_identifier()));

		let f = std::fs::File::create(&temp_filepath).expect("Failed to create model file");
		let mut writer = std::io::BufWriter::new(f);

		let bytes_io_count = std::io::copy(&mut reader, &mut writer).map_err(Error::wrap)?;
		if bytes_io_count != len as u64 {
			return Err(Error::new(format!("Failed to download entire model; file only has {bytes_io_count} bytes, expected {len}")));
		}

		drop(writer);

		match std::fs::rename(&temp_filepath, &model_filepath) {
			Ok(()) => Ok(model_filepath),
			Err(e) => {
				if model_filepath.exists() {
					let _ = std::fs::remove_file(temp_filepath);
					Ok(model_filepath)
				} else {
					Err(Error::new(format!("Failed to download model: {e}")))
				}
			}
		}
	}
}

pub mod mnist {
	use image::{ImageBuffer, Luma, Pixel};
	use ndarray::{Array4, Axis};
	use ort::{
		Result,
		util::ArrayExt,
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
