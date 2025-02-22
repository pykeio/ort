use std::path::Path;

use ndarray::{Axis, Ix2};
use ort::{
	Error,
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef
};
use tokenizers::Tokenizer;

/// Example usage of a text embedding model like Sentence Transformers' `all-mini-lm-l6` model for semantic textual
/// similarity.
///
/// Text embedding models map sentences & paragraphs to an n-dimensional dense vector space, which can then be used for
/// tasks like clustering or semantic search.
fn main() -> ort::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::fmt::init();

	// Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
	ort::init()
		.with_name("sbert")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	// Load our model
	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/all-MiniLM-L6-v2.onnx")?;

	// Load the tokenizer and encode the text.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();

	let inputs = vec!["The weather outside is lovely.", "It's so sunny outside!", "She drove to the stadium."];

	// Encode our input strings. `encode_batch` will pad each input to be the same length.
	let encodings = tokenizer.encode_batch(inputs.clone(), false).map_err(|e| Error::new(e.to_string()))?;

	// Get the padded length of each encoding.
	let padded_token_length = encodings[0].len();

	// Get our token IDs & mask as a flattened array.
	let ids: Vec<i64> = encodings.iter().flat_map(|e| e.get_ids().iter().map(|i| *i as i64)).collect();
	let mask: Vec<i64> = encodings.iter().flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64)).collect();

	// Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
	let a_ids = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*ids))?;
	let a_mask = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*mask))?;

	// Run the model.
	let outputs = session.run(ort::inputs![a_ids, a_mask])?;

	// Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
	let embeddings = outputs[1].try_extract_tensor::<f32>()?.into_dimensionality::<Ix2>().unwrap();

	println!("Similarity for '{}'", inputs[0]);
	let query = embeddings.index_axis(Axis(0), 0);
	for (embeddings, sentence) in embeddings.axis_iter(Axis(0)).zip(inputs.iter()).skip(1) {
		// Calculate cosine similarity against the 'query' sentence.
		let dot_product: f32 = query.iter().zip(embeddings.iter()).map(|(a, b)| a * b).sum();
		println!("\t'{}': {:.1}%", sentence, dot_product * 100.);
	}

	Ok(())
}
