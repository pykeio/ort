use std::{env, process};

use ort::{Session, TensorElementType, ValueType};

fn display_element_type(t: TensorElementType) -> &'static str {
	match t {
		TensorElementType::Bfloat16 => "bf16",
		TensorElementType::Bool => "bool",
		TensorElementType::Float16 => "f16",
		TensorElementType::Float32 => "f32",
		TensorElementType::Float64 => "f64",
		TensorElementType::Int16 => "i16",
		TensorElementType::Int32 => "i32",
		TensorElementType::Int64 => "i64",
		TensorElementType::Int8 => "i8",
		TensorElementType::String => "str",
		TensorElementType::Uint16 => "u16",
		TensorElementType::Uint32 => "u32",
		TensorElementType::Uint64 => "u64",
		TensorElementType::Uint8 => "u8"
	}
}

fn display_value_type(value: &ValueType) -> String {
	match value {
		ValueType::Tensor { ty, dimensions } => {
			format!(
				"Tensor<{}>({})",
				display_element_type(*ty),
				dimensions
					.iter()
					.map(|c| if *c == -1 { "dyn".to_string() } else { c.to_string() })
					.collect::<Vec<_>>()
					.join(", ")
			)
		}
		ValueType::Map { key, value } => format!("Map<{}, {}>", display_element_type(*key), display_element_type(*value)),
		ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner))
	}
}

fn main() -> ort::Result<()> {
	let Some(path) = env::args().nth(1) else {
		eprintln!("usage: ./model-info <model>.onnx");
		process::exit(0);
	};

	let session = Session::builder()?.commit_from_file(path)?;

	let meta = session.metadata()?;
	if let Ok(x) = meta.name() {
		println!("Name: {x}");
	}
	if let Ok(x) = meta.description() {
		println!("Description: {x}");
	}
	if let Ok(x) = meta.producer() {
		println!("Produced by {x}");
	}

	println!("Inputs:");
	for (i, input) in session.inputs.iter().enumerate() {
		println!("    {i} {}: {}", input.name, display_value_type(&input.input_type));
	}
	println!("Outputs:");
	for (i, output) in session.outputs.iter().enumerate() {
		println!("    {i} {}: {}", output.name, display_value_type(&output.output_type));
	}

	Ok(())
}
