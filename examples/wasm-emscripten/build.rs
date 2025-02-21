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
		let mut request = get("https://github.com/alfatraining/ort-artifacts-staging/releases/download/754ee21/ort_static-main-wasm32-unknown-emscripten.zip")
			.expect("Cannot request precompiled onnxruntime.");
		let mut buf = Vec::<u8>::new();
		request.read_to_end(&mut buf).expect("Cannot read precompiled onnxruntime.");

		// Prepare extraction.
		let reader = Cursor::new(buf);
		let mut zip = ZipArchive::new(reader).expect("Cannot incept unzipper.");

		// Extract precompiled library.
		// TODO: For debug builds, link to a debug build of onnxruntime.
		{
			let mut buf = Vec::<u8>::new();

			zip.by_name("onnxruntime/lib/libonnxruntime.a")
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
