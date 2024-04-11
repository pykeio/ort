use ort::{Session, TensorElementType, ValueType};
use wasm_bindgen::prelude::wasm_bindgen;

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

pub fn display() -> ort::Result<()> {
	let session = Session::builder()?;
	println!("builder?");
	let session = session.commit_from_memory_directly(&[0])?;

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

macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

#[wasm_bindgen]
pub fn test() {
	console_log!("depo");
	if let Err(e) = display() {
		console_log!("err: {e:?}");
	}
}

#[cfg(test)]
#[wasm_bindgen_test::wasm_bindgen_test]
fn run() {
	use ort::wasm::libc_shims::*;
	use tracing_subscriber::fmt;
	use tracing_subscriber_wasm::MakeConsoleWriter;

	ort::wasm::_initialize();

	fmt()
		.with_writer(MakeConsoleWriter::default().map_trace_level_to(tracing::Level::DEBUG))
		.without_time()
		.init();

	console_log!("1");

	wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

	console_log!("2");

	std::panic::set_hook(Box::new(console_error_panic_hook::hook));

	console_log!("3");

	unsafe {
		let a = malloc(128);
		*a.add(0) = 42;
		free(a);
	};
	console_log!("free 1");

	unsafe {
		let a = malloc(640);
		*a.add(0) = 42;
		free(a);
	};
	console_log!("free 2");

	unsafe {
		let mut a = std::ptr::null_mut();
		posix_memalign(&mut a, 128, 16);
		*a.add(0) = 42;
		free(a);
	};
	console_log!("free 3");

	test();
}
