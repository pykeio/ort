use std::{
	env,
	fs::File,
	io::{BufRead, BufReader, BufWriter, Write},
	path::Path
};

use simd_json::derived::ValueObjectAccessAsScalar;
use tokenizers::Tokenizer;

const MAX_TOKENS: usize = 500_000;

fn main() {
	let input = env::args().nth(1).expect("provide input jsonl");
	let output = env::args().nth(2).unwrap_or_else(|| "dataset.bin".into());

	let input = BufReader::new(File::open(input).unwrap());
	let mut output = BufWriter::new(File::create(output).unwrap());

	let tokenizer = Tokenizer::from_file(
		Path::new(env!("CARGO_MANIFEST_DIR"))
			.parent()
			.unwrap()
			.join("gpt2")
			.join("data")
			.join("tokenizer.json")
	)
	.unwrap();
	let mut bytes_written = 0;

	for line in input.lines() {
		let line: simd_json::OwnedValue = unsafe { simd_json::from_str(&mut line.unwrap()).unwrap() };
		let tokenized = tokenizer
			.encode(format!("<|endoftext|>{}", line.get_str("message").unwrap()), false)
			.unwrap();
		let id_bytes: Vec<u8> = tokenized.get_ids().iter().flat_map(|c| (*c as u16).to_le_bytes()).collect();
		output.write_all(&id_bytes).unwrap();
		bytes_written += id_bytes.len();
		if bytes_written >= MAX_TOKENS * 2 {
			output.flush().unwrap();
			break;
		}
	}
}
