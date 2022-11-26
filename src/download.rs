#[cfg(feature = "fetch-models")]
use std::{
	fs, io,
	path::{Path, PathBuf},
	time::Duration
};

#[cfg(feature = "fetch-models")]
use tracing::info;

#[cfg(feature = "fetch-models")]
use crate::error::{OrtDownloadError, OrtResult};

pub mod language;
pub mod vision;

/// Available pre-trained models to download from the [ONNX Model Zoo](https://github.com/onnx/models).
#[derive(Debug, Clone)]
pub enum OnnxModel {
	/// Computer vision models
	Vision(vision::Vision),
	/// Language models
	Language(language::Language)
}

trait ModelUrl {
	fn fetch_url(&self) -> &'static str;
}

impl ModelUrl for OnnxModel {
	fn fetch_url(&self) -> &'static str {
		match self {
			OnnxModel::Vision(model) => model.fetch_url(),
			OnnxModel::Language(model) => model.fetch_url()
		}
	}
}

impl OnnxModel {
	#[cfg(feature = "fetch-models")]
	#[tracing::instrument]
	pub(crate) fn download_to<P>(&self, download_dir: P) -> OrtResult<PathBuf>
	where
		P: AsRef<Path> + std::fmt::Debug
	{
		let url = self.fetch_url();

		let model_filename = PathBuf::from(url.split('/').last().unwrap());
		let model_filepath = download_dir.as_ref().join(model_filename);
		if model_filepath.exists() {
			info!(model_filepath = format!("{}", model_filepath.display()).as_str(), "Model already exists, skipping download");
			Ok(model_filepath)
		} else {
			info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{:?}", url).as_str(), "Downloading model");

			let resp = ureq::get(url)
				.timeout(Duration::from_secs(180))
				.call()
				.map_err(Box::new)
				.map_err(OrtDownloadError::FetchError)?;

			assert!(resp.has("Content-Length"));
			let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
			info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_reader();

			let f = fs::File::create(&model_filepath).unwrap();
			let mut writer = io::BufWriter::new(f);

			let bytes_io_count = io::copy(&mut reader, &mut writer).map_err(OrtDownloadError::IoError)?;
			if bytes_io_count == len as u64 {
				Ok(model_filepath)
			} else {
				Err(OrtDownloadError::CopyError {
					expected: len as u64,
					io: bytes_io_count
				}
				.into())
			}
		}
	}
}
