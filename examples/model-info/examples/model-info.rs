use std::{env, process};

use ort::{Session, TensorElementDataType, ValueType};

fn display_element_type(t: TensorElementDataType) -> &'static str {
	match t {
		TensorElementDataType::Bfloat16 => "bf16",
		TensorElementDataType::Bool => "bool",
		TensorElementDataType::Float16 => "f16",
		TensorElementDataType::Float32 => "f32",
		TensorElementDataType::Float64 => "f64",
		TensorElementDataType::Int16 => "i16",
		TensorElementDataType::Int32 => "i32",
		TensorElementDataType::Int64 => "i64",
		TensorElementDataType::Int8 => "i8",
		TensorElementDataType::String => "str",
		TensorElementDataType::Uint16 => "u16",
		TensorElementDataType::Uint32 => "u32",
		TensorElementDataType::Uint64 => "u64",
		TensorElementDataType::Uint8 => "u8"
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

	let session = Session::builder()?.with_model_from_file(path)?;

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
