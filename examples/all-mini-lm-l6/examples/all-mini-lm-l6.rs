use std::path::Path;

use ndarray::{Array1, Axis};
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

/// all-mini-lm-l6 embeddings generation
///
/// This is a sentence-transformers model: It maps sentences & paragraphs to a 384
///
/// dimensional dense vector space and can be used for tasks like clustering or semantic search.
fn main() -> ort::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::fmt::init();

	// Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
	ort::init()
		.with_name("all-Mini-LM-L6")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	// Load our model
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_url("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx")?;

	// Load the tokenizer and encode the text.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
	let tokens = tokenizer.encode("test", false)?;
	let mask = tokens.get_attention_mask().iter().map(|i| *i as i64).collect::<Vec<i64>>();
	let ids = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<i64>>();
	let a_ids = Array1::from_vec(ids);
	let a_mask = Array1::from_vec(mask);
	let input_ids = a_ids.view().insert_axis(Axis(0));
	let input_mask = a_mask.view().insert_axis(Axis(0));
	let outputs = session.run(ort::inputs![input_ids, input_mask]?)?;
	let tensor = outputs[1].try_extract_tensor::<f32>();
	println!("{:?}", tensor);
	Ok(())
}
