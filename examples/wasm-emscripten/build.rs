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
	use std::{
		fs::File,
		io::{Cursor, Read, Write}
	};

	use reqwest::blocking::get;
	use zip::ZipArchive;

	// Determine mode.
	let mode = match cfg!(debug_assertions) {
		true => "debug",
		false => "release"
	};

	// Download precompiled libonnxruntime.a.
	{
		// Request archive.
		let mut request = get("https://github.com/alfatraining/onnxruntime-wasm-builds/releases/download/v1.20.1/libonnxruntime-v1.20.1-wasm.zip")
			.expect("Cannot request precompiled onnxruntime.");
		let mut buf = Vec::<u8>::new();
		request.read_to_end(&mut buf).expect("Cannot read precompiled onnxruntime.");

		// Prepare extraction.
		let reader = Cursor::new(buf);
		let mut zip = ZipArchive::new(reader).expect("Cannot incept unzipper.");

		// Extract precompiled library.
		{
			let mut buf = Vec::<u8>::new();
			let mut mode_title_case = mode.to_string();
			mode_title_case = format!("{}{mode_title_case}", mode_title_case.remove(0).to_uppercase());
			zip.by_name(format!("{mode_title_case}/libonnxruntime.a").as_str())
				.expect("Cannot find precompiled onnxruntime.")
				.read_to_end(&mut buf)
				.expect("Cannot read precompiled onnxruntime.");
			File::create("./libonnxruntime.a")
				.expect("Cannot create precompiled onnxruntime.")
				.write_all(&buf)
				.expect("Cannot store precompiled onnxruntime.");
		}
	}

	// Download model.
	{
		let mut request = get("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx").expect("Cannot request model.");
		let mut buf = Vec::<u8>::new();
		request.read_to_end(&mut buf).expect("Cannot read model.");
		let mut file = File::create("./yolov8m.onnx").expect("Cannot create model file.");
		file.write_all(&buf).expect("Cannot store model.");
	}

	// Copy index.html and pictures to target directory.
	{
		println!("cargo:rerun-if-changed=index.html");
		std::fs::copy("index.html", format!("../../target/wasm32-unknown-emscripten/{mode}/index.html")).expect("Cannot copy index.html.");

		println!("cargo:rerun-if-changed=pictures/*");
		copy_dir_all("pictures", format!("../../target/wasm32-unknown-emscripten/{mode}/pictures")).expect("Cannot copy pictures.");
	}
}
