use std::{
	io::{self, Write},
	path::Path
};

use ndarray::{array, concatenate, s, Array1, Axis};
use ort::{download::language::machine_comprehension::GPT2, inputs, CUDAExecutionProvider, Environment, GraphOptimizationLevel, SessionBuilder, Tensor};
use rand::Rng;
use tokenizers::Tokenizer;

/// Prompt
const PROMPT: &str = "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
/// Max Tokens to Generate
const GEN_TOKENS: i32 = 90;
/// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
const TOP_K: usize = 5;

/// GPT-2 Text Generation
///
/// This Rust program demonstrates text generation using the GPT-2 language model with the ONNX Runtime.
/// The program initializes the model, tokenizes a prompt, and generates a sequence of tokens.
/// It utilizes top-k sampling for diverse and contextually relevant text generation.
///
/// # Constants
/// - `PROMPT`: The initial prompt for text generation.
/// - `GEN_TOKENS`: The maximum number of tokens to generate.
/// - `TOP_K`: Parameter for top-k sampling, influencing the diversity of generated text.
///
/// # Usage
/// Ensure that the required dependencies are installed and run the Rust script to generate text using the GPT-2 model.
///
/// # Main Function
/// The main function initializes dependencies, loads the GPT-2 model, tokenizes the prompt,
/// and iteratively generates text based on the model's output probabilities.
///
/// ## Steps
/// 1. Initialize tracing, stdout, and the random number generator.
/// 2. Create the ONNX Runtime environment and session for the GPT-2 model.
/// 3. Load the tokenizer and encode the prompt into a sequence of tokens.
/// 4. Iteratively generate tokens using the GPT-2 model and top-k sampling.
/// 5. Print the generated text to the console.
///
/// # Panics
/// The program panics if there is an issue with the ONNX Runtime or tokenizer.
///
/// # Errors
/// Returns an `ort::Result` indicating success or an error during ONNX Runtime execution.
///
/// # Examples
/// ```rust
/// fn main() -> ort::Result<()> {
/// 	// ... (see the main function for the complete example)
/// }
/// ```
fn main() -> ort::Result<()> {
	tracing_subscriber::fmt::init();

	let mut stdout = io::stdout();
	let mut rng = rand::thread_rng();

	let environment = Environment::builder()
		.with_name("GPT-2")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.build()?
		.into_arc();

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_downloaded(GPT2::GPT2LmHead)?;

	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
	let tokens = tokenizer.encode(PROMPT, false).unwrap();
	let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	let mut tokens = Array1::from_iter(tokens.iter().cloned());

	print!("{PROMPT}");
	stdout.flush().unwrap();

	for _ in 0..GEN_TOKENS {
		let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
		let outputs = session.run(inputs![array]?)?;
		let generated_tokens: Tensor<f32> = outputs["output1"].extract_tensor()?;
		let generated_tokens = generated_tokens.view();

		let probabilities = &mut generated_tokens
			.slice(s![0, 0, -1, ..])
			.insert_axis(Axis(0))
			.to_owned()
			.iter()
			.cloned()
			.enumerate()
			.collect::<Vec<_>>();
		probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

		let token = probabilities[rng.gen_range(0..=TOP_K)].0;
		tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

		let token_str = tokenizer.decode(&[token as _], true).unwrap();
		print!("{}", token_str);
		stdout.flush().unwrap();
	}

	println!();

	Ok(())
}
