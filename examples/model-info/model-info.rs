use std::{env, process};

use ort::session::Session;

// Include common code for `ort` examples that allows using the various feature flags to enable different EPs and
// backends.
#[path = "../common/mod.rs"]
mod common;

fn main() -> ort::Result<()> {
	// Register backends based on feature flags - this isn't crucial for usage and can be removed.
	common::init()?;

	let Some(path) = env::args().nth(1) else {
		eprintln!("usage: ./model-info <model>.onnx");
		process::exit(0);
	};

	let session = Session::builder()?.commit_from_file(path)?;

	let meta = session.metadata()?;
	if let Some(x) = meta.name() {
		println!("Name: {x}");
	}
	if let Some(x) = meta.description() {
		println!("Description: {x}");
	}
	if let Some(x) = meta.producer() {
		println!("Produced by {x}");
	}

	if let Ok(custom_keys) = meta.custom_keys()
		&& !custom_keys.is_empty()
	{
		println!("=== Custom keys ===");
		for key in custom_keys {
			if let Some(value) = meta.custom(&key) {
				println!("    {key}: {value}");
			}
		}
	};

	println!("=== Inputs ===");
	for (i, input) in session.inputs().iter().enumerate() {
		println!("    {i} {}: {}", input.name(), input.dtype());
	}
	println!("=== Outputs ===");
	for (i, output) in session.outputs().iter().enumerate() {
		println!("    {i} {}: {}", output.name(), output.dtype());
	}

	Ok(())
}
