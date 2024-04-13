use ndarray::{Array4, ArrayViewD};
use ort::Session;
use wasm_bindgen::prelude::*;

static MODEL_BYTES: &[u8] = include_bytes!("upsample.ort");

pub fn upsample_inner() -> ort::Result<()> {
	let session = Session::builder()?
		.commit_from_memory_directly(MODEL_BYTES)
		.expect("Could not read model from memory");

	let array = Array4::<f32>::zeros((1, 224, 224, 3));

	let outputs = session.run(ort::inputs![array]?)?;

	assert_eq!(outputs.len(), 1);
	let output: ArrayViewD<f32> = outputs[0].try_extract_tensor()?;

	assert_eq!(output.shape(), [1, 448, 448, 3]);

	Ok(())
}

macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

#[wasm_bindgen]
pub fn upsample() {
	if let Err(e) = upsample_inner() {
		console_log!("Error occurred while performing inference: {e:?}");
	}
}

#[cfg(test)]
#[wasm_bindgen_test::wasm_bindgen_test]
fn run_test() {
	use tracing::Level;
	use tracing_subscriber::fmt;
	use tracing_subscriber_wasm::MakeConsoleWriter;

	ort::wasm::_initialize();

	fmt()
		.with_ansi(false)
		.with_max_level(Level::DEBUG)
		.with_writer(MakeConsoleWriter::default().map_trace_level_to(Level::DEBUG))
		.without_time()
		.init();

	wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

	std::panic::set_hook(Box::new(console_error_panic_hook::hook));

	upsample();
}
