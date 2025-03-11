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
		io::{Read, Write}
	};

	use reqwest::blocking::get;

	// Determine mode.
	let mode = match cfg!(debug_assertions) {
		true => "debug",
		false => "release"
	};

	// Download model.
	{
		let mut request = get("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/yolov8m.onnx").expect("Cannot request model.");
		let mut buf = Vec::<u8>::new();
		request.read_to_end(&mut buf).expect("Cannot read model.");
		let mut file = File::create("./yolov8m.onnx").expect("Cannot create model file.");
		file.write_all(&buf).expect("Cannot store model.");
	}

	// Copy index.html and pictures to target directory.
	{
		println!("cargo:rerun-if-changed=index.html");
		std::fs::copy("index.html", format!("./target/wasm32-unknown-emscripten/{mode}/index.html")).expect("Cannot copy index.html.");

		println!("cargo:rerun-if-changed=pictures/*");
		copy_dir_all("pictures", format!("./target/wasm32-unknown-emscripten/{mode}/pictures")).expect("Cannot copy pictures.");
	}
}
