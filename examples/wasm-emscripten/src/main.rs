#![no_main]

// Embed models into the .wasm file.
#[derive(rust_embed::RustEmbed)]
#[folder = "models/"]
pub struct Models;

static RUN_COUNT: u32 = 10;

#[no_mangle]
pub fn hello_ort() {
	ort::init()
		.with_global_thread_pool(ort::environment::GlobalThreadPoolOptions::default())
		.with_execution_providers([ort::execution_providers::cpu::CPUExecutionProvider::default().build()])
		.commit()
		.expect("Cannot initialize ort.");

	let model = Models::get("silero_vad.onnx").expect("Cannot find model.").data.to_vec();

	let session = ort::session::Session::builder()
		.expect("Cannot create Session builder.")
		.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
		.expect("Cannot optimize graph.")
		.with_parallel_execution(true)
		.expect("Cannot activate parallel execution.")
		.with_intra_threads(2)
		.expect("Cannot set intra thread count.")
		.with_inter_threads(1)
		.expect("Cannot set inter thread count.")
		.commit_from_memory(&model)
		.expect("Cannot commit model.");

	let chunk = ort::value::Tensor::from_array(ndarray::Array2::<f32>::zeros((1, 512))).expect("Cannot create chunk.");
	let state = ort::value::Tensor::from_array(ndarray::ArrayD::<f32>::zeros([2, 1, 128].as_slice())).expect("Cannot create state.");
	let sample_rate = ort::value::Tensor::from_array(ndarray::Array1::<i64>::from_vec(vec![16_000_i64])).expect("Cannot create sample rate.");

	let before = std::time::Instant::now();
	for _ in 0..RUN_COUNT {
		let _result = session
			.run([chunk.view().into(), state.view().into(), sample_rate.view().into()])
			.expect("Cannot run session.");
	}

	let us = ((std::time::Instant::now() - before) / RUN_COUNT).as_micros();
	println!("Inference took {us} microseconds on average.");
}
