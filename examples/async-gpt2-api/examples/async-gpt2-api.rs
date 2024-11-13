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
use ndarray::{Array1, ArrayViewD, Axis, array, concatenate, s};
use ort::{
	execution_providers::CUDAExecutionProvider,
	inputs,
	session::{Session, builder::GraphOptimizationLevel}
};
use rand::Rng;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
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
		.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")?;

	// Load the tokenizer and encode the prompt into a sequence of tokens.
	let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();

	let app_state = AppState {
		session: Arc::new(session),
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
	session: Arc<Session>,
	tokenizer: Arc<Tokenizer>
}

fn generate_stream(tokenizer: Arc<Tokenizer>, session: Arc<Session>, tokens: Vec<i64>, gen_tokens: usize) -> impl Stream<Item = ort::Result<Event>> + Send {
	async_stream::try_stream! {
		let mut tokens = Array1::from_iter(tokens.iter().cloned());

		for _ in 0..gen_tokens {
			let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
			let outputs = session.run_async(inputs![array]?)?.await?;
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
			let token = {
				let mut rng = rand::thread_rng();
				probabilities[rng.gen_range(0..=5)].0
			};
			tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

			let token_str = tokenizer.decode(&[token as _], true).unwrap();
			yield Event::default().data(token_str);
		}
	}
}

impl FromRef<AppState> for Arc<Session> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.session)
	}
}
impl FromRef<AppState> for Arc<Tokenizer> {
	fn from_ref(input: &AppState) -> Self {
		Arc::clone(&input.tokenizer)
	}
}

async fn generate(State(session): State<Arc<Session>>, State(tokenizer): State<Arc<Tokenizer>>) -> Sse<impl Stream<Item = ort::Result<Event>>> {
	Sse::new(generate_stream(tokenizer, session, vec![0], 50)).keep_alive(KeepAlive::new())
}
