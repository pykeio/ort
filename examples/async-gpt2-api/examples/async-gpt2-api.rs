use std::{path::Path, sync::Arc};

use axum::{
	Router,
	extract::{FromRef, State},
	response::{
		Sse,
		sse::{Event, KeepAlive}
	},
	routing::post
};
use futures::Stream;
use ort::{
	execution_providers::CUDAExecutionProvider,
	session::{Session, builder::GraphOptimizationLevel},
	value::TensorRef
};
use rand::Rng;
use tokenizers::Tokenizer;
use tokio::{net::TcpListener, sync::Mutex};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	// Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
	ort::init()
		.with_name("GPT-2")
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	// Load our model
	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(4)?
		.commit_from_url("https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/gpt2.onnx")?;

	// Load the tokenizer and encode the prompt into a sequence of tokens.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();

	let app_state = AppState {
		session: Arc::new(Mutex::new(session)),
		tokenizer: Arc::new(tokenizer)
	};

	let app = Router::new().route("/generate", post(generate)).with_state(app_state).into_make_service();
	let listener = TcpListener::bind("127.0.0.1:7216").await?;
	tracing::info!("Listening on {}", listener.local_addr()?);

	axum::serve(listener, app).await?;

	Ok(())
}

#[derive(Clone)]
struct AppState {
	session: Arc<Mutex<Session>>,
	tokenizer: Arc<Tokenizer>
}

fn generate_stream(
	tokenizer: Arc<Tokenizer>,
	session: Arc<Mutex<Session>>,
	mut tokens: Vec<i64>,
	gen_tokens: usize
) -> impl Stream<Item = ort::Result<Event>> + Send {
	async_stream_lite::try_async_stream(|yielder| async move {
		for _ in 0..gen_tokens {
			let input = TensorRef::from_array_view((vec![1, 1, tokens.len() as i64], tokens.as_slice()))?;
			let probabilities = {
				let mut session = session.lock().await;
				let outputs = session.run_async(ort::inputs![input])?.await?;
				let (dim, probabilities) = outputs["output1"].try_extract_raw_tensor()?;

				// Collect and sort logits
				let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
				let mut probabilities: Vec<(usize, f32)> = probabilities[(seq_len - 1) * vocab_size..].iter().copied().enumerate().collect();
				probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));
				probabilities
			};

			// Sample using top-k sampling
			let token = {
				let mut rng = rand::thread_rng();
				probabilities[rng.gen_range(0..=5)].0 as i64
			};
			tokens.push(token);

			let token_str = tokenizer.decode(&[token as _], true).unwrap();
			yielder.r#yield(Event::default().data(token_str)).await;
		}

		Ok(())
	})
}

impl FromRef<AppState> for Arc<Mutex<Session>> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.session)
	}
}
impl FromRef<AppState> for Arc<Tokenizer> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.tokenizer)
	}
}

async fn generate(State(session): State<Arc<Mutex<Session>>>, State(tokenizer): State<Arc<Tokenizer>>) -> Sse<impl Stream<Item = ort::Result<Event>>> {
	Sse::new(generate_stream(tokenizer, session, vec![0], 50)).keep_alive(KeepAlive::new())
}
