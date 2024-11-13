use std::{
	io::{self, Write},
	path::Path
};

use ndarray::{Array1, ArrayViewD, Axis, array, concatenate, s};
use ort::{
	execution_providers::CUDAExecutionProvider,
	inputs,
	session::{Session, builder::GraphOptimizationLevel}
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
/// The program initializes the model, tokenizes a prompt, and generates a sequence of tokens.
/// It utilizes top-k sampling for diverse and contextually relevant text generation.
fn main() -> ort::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::fmt::init();

	// Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
	ort::init()
		.with_name("GPT-2")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let mut stdout = io::stdout();
	let mut rng = rand::thread_rng();

	// Load our model
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")?;

	// Load the tokenizer and encode the prompt into a sequence of tokens.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
	let tokens = tokenizer.encode(PROMPT, false).unwrap();
	let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	let mut tokens = Array1::from_iter(tokens.iter().cloned());

	print!("{PROMPT}");
	stdout.flush().unwrap();

	for _ in 0..GEN_TOKENS {
		let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
		let outputs = session.run(inputs![array]?)?;
		let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor()?;

		// Collect and sort logits
		let probabilities = &mut generated_tokens
			.slice(s![0, 0, -1, ..])
			.insert_axis(Axis(0))
			.to_owned()
			.iter()
			.cloned()
			.enumerate()
			.collect::<Vec<_>>();
		probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

		// Sample using top-k sampling
		let token = probabilities[rng.gen_range(0..=TOP_K)].0;
		tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

		let token_str = tokenizer.decode(&[token as _], true).unwrap();
		print!("{}", token_str);
		stdout.flush().unwrap();
	}

	println!();

	Ok(())
}
