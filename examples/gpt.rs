use std::io::{self, Write};

use ndarray::{array, concatenate, s, Array1, Axis, CowArray};
use ort::{
	download::language::machine_comprehension::GPT2, tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder,
	Value
};
use rand::Rng;
use tokenizers::Tokenizer;

const PROMPT: &str = "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
const GEN_TOKENS: i32 = 90;
const TOP_K: usize = 5;

fn main() -> OrtResult<()> {
	tracing_subscriber::fmt::init();

	let mut stdout = io::stdout();
	let mut rng = rand::thread_rng();

	let environment = Environment::builder()
		.with_name("GPT-2")
		.with_execution_providers([ExecutionProvider::CUDA(Default::default())])
		.build()?
		.into_arc();

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_downloaded(GPT2::GPT2LmHead)?;

	let tokenizer = Tokenizer::from_file("tests/data/gpt2-tokenizer.json").unwrap();
	let tokens = tokenizer.encode(PROMPT, false).unwrap();
	let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

	let mut tokens = CowArray::from(Array1::from_iter(tokens.iter().cloned()));

	print!("{PROMPT}");
	stdout.flush().unwrap();

	for _ in 0..GEN_TOKENS {
		let n_tokens = tokens.shape()[0];
		let array = tokens.clone().insert_axis(Axis(0)).into_shape((1, 1, n_tokens)).unwrap().into_dyn();
		let inputs = vec![Value::from_array(session.allocator(), &array)?];
		let outputs: Vec<Value> = session.run(inputs)?;
		let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
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
		tokens = CowArray::from(concatenate![Axis(0), tokens, array![token.try_into().unwrap()]]);

		let token_str = tokenizer.decode(&[token as _], true).unwrap();
		print!("{}", token_str);
		stdout.flush().unwrap();
	}

	println!();

	Ok(())
}
