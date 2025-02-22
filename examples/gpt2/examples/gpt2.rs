use std::{
	io::{self, Write},
	path::Path
};

use ort::{
	execution_providers::CUDAExecutionProvider,
	inputs,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef
};
use rand::Rng;
use tokenizers::Tokenizer;

const PROMPT: &str = "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
/// Max tokens to generate
const GEN_TOKENS: i32 = 90;
/// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
const TOP_K: usize = 5;

/// GPT-2 Text Generation
///
/// This Rust program demonstrates text generation using the GPT-2 language model with `ort`.
fn main() -> ort::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::fmt::init();

	// Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
	ort::init()
		.with_name("GPT-2")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let mut stdout: io::Stdout = io::stdout();
	let mut rng = rand::thread_rng();

	// Load our model
	let mut session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx")?;

	// Load the tokenizer and encode the prompt into a sequence of tokens.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
	let tokens = tokenizer.encode(PROMPT, false).unwrap();
	let mut tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	print!("{PROMPT}");
	stdout.flush().unwrap();

	for _ in 0..GEN_TOKENS {
		// Raw tensor construction takes a tuple of (dimensions, data).
		// The model expects our input to have shape [B, _, S]
		let input = TensorRef::from_array_view((vec![1, 1, tokens.len() as i64], tokens.as_slice()))?;
		let outputs = session.run(inputs![input])?;
		let (dim, mut probabilities) = outputs["output1"].try_extract_raw_tensor()?;

		// The output tensor will have shape [B, _, S, V]
		// We want only the probabilities for the last token in this sequence, which will be the next most likely token
		// according to the model
		let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
		probabilities = &probabilities[(seq_len - 1) * vocab_size..];

		// Sort each token by probability
		let mut probabilities: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
		probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

		// Sample using top-k sampling
		let token = probabilities[rng.gen_range(0..=TOP_K)].0 as i64;

		// Add our generated token to the input sequence
		tokens.push(token);

		let token_str = tokenizer.decode(&[token as u32], true).unwrap();
		print!("{}", token_str);
		stdout.flush().unwrap();
	}

	println!();

	Ok(())
}
