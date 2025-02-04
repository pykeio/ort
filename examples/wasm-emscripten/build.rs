fn copy_dir_all(src: impl AsRef<std::path::Path>, dst: impl AsRef<std::path::Path>) -> std::io::Result<()> {
	std::fs::create_dir_all(&dst)?;
	for entry in std::fs::read_dir(src)? {
		let entry = entry?;
		let ty = entry.file_type()?;
		if ty.is_dir() {
			copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
		} else {
			std::fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
		}
	}
	Ok(())
}

fn main() {
	// Download precompiled libonnxruntime.a.
	{
		use std::{fs::File, io::Write};

		use data_downloader::{DownloadRequest, InZipDownloadRequest, get};

		let request = InZipDownloadRequest {
			parent: &DownloadRequest {
				url: "https://github.com/alfatraining/onnxruntime-wasm-builds/releases/download/v1.20.1/libonnxruntime-v1.20.1-wasm.zip",
				sha256_hash: &hex_literal::hex!("bf22b0bf1336babf116839fa58a257aa91112e9bb2dae7fcd4c4a4dee11b70af")
			},
			path: "Release/libonnxruntime.a",
			sha256_hash: &hex_literal::hex!("abcf64d106168458d08905f97114f0289ebad0912ee96b92e8130670297b5c22")
		};
		let bytes = get(&request).expect("Cannot request libonnxruntime.a.");
		let mut file = File::create("libonnxruntime.a").expect("Cannot create libonnxruntime.a.");
		file.write_all(&bytes).expect("Cannot store libonnxruntime.a.");
	}

	// Download model.
	{
		use std::{
			fs::{File, create_dir_all},
			io::Write
		};

		use data_downloader::{DownloadRequest, get};

		let request = DownloadRequest {
			url: "https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx",
			sha256_hash: &hex_literal::hex!("e91bd39ce15420f9623d5e3d46580f7ce4dfc85d061e4ee2e7b78ccd3e5b9453")
		};
		let bytes = get(&request).expect("Cannot request model.");
		create_dir_all("models").expect("Cannot create models directory.");
		let mut file = File::create("models/yolov8m.onnx").expect("Cannot create model file.");
		file.write_all(&bytes).expect("Cannot store model file.");
	}

	// Copy index.html and pictures to target directory.
	{
		let mode = match cfg!(debug_assertions) {
			true => "debug",
			false => "release"
		};
		println!("cargo:rerun-if-changed=index.html");
		std::fs::copy("index.html", format!("../../target/wasm32-unknown-emscripten/{mode}/index.html")).expect("Cannot copy index.html.");

		println!("cargo:rerun-if-changed=pictures/*");
		copy_dir_all("pictures", format!("../../target/wasm32-unknown-emscripten/{mode}/pictures")).expect("Cannot copy pictures.");
	}
}
