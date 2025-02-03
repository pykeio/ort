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
			url: "https://github.com/snakers4/silero-vad/raw/refs/heads/master/src/silero_vad/data/silero_vad.onnx",
			sha256_hash: &hex_literal::hex!("2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f")
		};
		let bytes = get(&request).expect("Cannot request model.");
		create_dir_all("models").expect("Cannot create models directory.");
		let mut file = File::create("models/silero_vad.onnx").expect("Cannot create model file.");
		file.write_all(&bytes).expect("Cannot store model file.");
	}

	// Copy index.html to target directory.
	{
		println!("cargo:rerun-if-changed=index.html");
		let mode = match cfg!(debug_assertions) {
			true => "debug",
			false => "release"
		};
		std::fs::copy("index.html", format!("../../target/wasm32-unknown-emscripten/{mode}/index.html")).expect("Cannot find target directory.");
	}
}
