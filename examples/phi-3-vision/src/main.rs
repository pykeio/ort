mod image_process;
use std::{path::Path, time::Instant};

use anyhow::Result;
use image::DynamicImage;
use ndarray::{Array, Array2, Array3, Array4, ArrayView, Ix3, Ix4, s};
use ort::{session::Session, value::Tensor};
use tokenizers::Tokenizer;

const VISION_MODEL_NAME: &str = "phi-3-v-128k-instruct-vision.onnx";
const TEXT_EMBEDDING_MODEL_NAME: &str = "phi-3-v-128k-instruct-text-embedding.onnx";
const GENERATION_MODEL_NAME: &str = "phi-3-v-128k-instruct-text.onnx";

const MAX_LENGTH: usize = 1000; // max length of the generated text
const EOS_TOKEN_ID: i64 = 32007; // <|end|>
const USER_TOKEN_ID: i64 = 32010; // <|user|>
const VOCAB_SIZE: usize = 32064;

#[allow(dead_code)]
fn get_current_time() -> Instant {
	Instant::now()
}

fn get_image_embedding(vision_model: &Session, img: &Option<DynamicImage>) -> Result<Array3<f32>> {
	let visual_features = if let Some(img) = img {
		let image_processor = image_process::Phi3VImageProcessor::new();
		let result = image_processor.preprocess(img)?;
		tracing::debug!(
			"image process result, num_img_tokens: {num_img_tokens:?}, pixel_values: {pixel_values:?}, image_sizes: {image_sizes:?}",
			num_img_tokens = result.num_img_tokens,
			pixel_values = result.pixel_values.shape(),
			image_sizes = result.image_sizes.shape(),
		);
		let model_inputs = ort::inputs![
			"pixel_values" => result.pixel_values,
			"image_sizes" => result.image_sizes,
		]?;
		let outputs = vision_model.run(model_inputs)?;
		let predictions_view: ArrayView<f32, _> = outputs["visual_features"].try_extract_tensor::<f32>()?;
		predictions_view.into_dimensionality::<Ix3>()?.to_owned()
	} else {
		Array::zeros((1, 0, 0))
	};
	Ok(visual_features)
}

fn get_text_embedding(text_embedding_model: &Session, input_ids: &Array2<i64>) -> Result<Array3<f32>> {
	let model_inputs = ort::inputs![
		"input_ids" => input_ids.to_owned(),
	]?;
	let outputs = text_embedding_model.run(model_inputs)?;
	let inputs_embeds_view: ArrayView<f32, _> = outputs["inputs_embeds"].try_extract_tensor::<f32>()?;
	let inputs_embeds = inputs_embeds_view.into_dimensionality::<Ix3>()?.to_owned();
	Ok(inputs_embeds)
}

fn merge_text_and_image_embeddings(
	inputs_embeds: &Array3<f32>,
	attention_mask: &Array2<i64>,
	visual_features: &Array3<f32>,
	image_token_position: usize
) -> (Array3<f32>, Array2<i64>) {
	let mut combined_embeds = Array3::zeros((1, inputs_embeds.shape()[1] + visual_features.shape()[1], inputs_embeds.shape()[2]));

	// Copy text embeddings up to the <|image_1|> token
	combined_embeds
		.slice_mut(s![.., ..image_token_position, ..])
		.assign(&inputs_embeds.slice(s![.., ..image_token_position, ..]));

	// Insert visual features
	combined_embeds
		.slice_mut(s![.., image_token_position..(image_token_position + visual_features.shape()[1]), ..])
		.assign(visual_features);

	// Copy the remaining text embeddings
	combined_embeds
		.slice_mut(s![.., (image_token_position + visual_features.shape()[1]).., ..])
		.assign(&inputs_embeds.slice(s![.., image_token_position.., ..]));

	// Update attention_mask
	let mut new_attention_mask = Array2::ones((1, attention_mask.shape()[1] + visual_features.shape()[1]));
	new_attention_mask
		.slice_mut(s![.., ..image_token_position])
		.assign(&attention_mask.slice(s![.., ..image_token_position]));
	new_attention_mask
		.slice_mut(s![.., (image_token_position + visual_features.shape()[1])..])
		.assign(&attention_mask.slice(s![.., image_token_position..]));

	(combined_embeds, new_attention_mask)
}

/// see https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/phi3v.py
/// <|user|><|image_1|>{text}<|end|><|assistant|>
/// Includes the `<s>` token, which is typically used as the BOS (Beginning of Sequence) token by LlamaTokenizer
fn format_chat_template(img: &Option<DynamicImage>, txt: &str) -> String {
	match img {
		Some(_) => format!("<s><|user|>\n<|image_1|>\n{txt}<|end|>\n<|assistant|>\n", txt = txt),
		None => format!("<s><|user|>\n{txt}<|end|>\n<|assistant|>\n", txt = txt)
	}
}

pub async fn generate_text(
	tokenizer: &Tokenizer,
	vision_model: &Session,
	text_embedding_model: &Session,
	generation_model: &Session,
	image: &Option<DynamicImage>,
	text: &str
) -> Result<()> {
	let (inputs_embeds, mut attention_mask) = {
		let visual_features = get_image_embedding(vision_model, image)?;
		let prompt = format_chat_template(image, text);
		let encoding = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("Error encoding: {:?}", e))?;

		let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
		let input_ids: Array2<i64> = Array2::from_shape_vec((1, input_ids.len()), input_ids)?;
		let mut inputs_embeds: Array3<f32> = get_text_embedding(text_embedding_model, &input_ids)?;

		let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&mask| mask as i64).collect();
		let mut attention_mask: Array2<i64> = Array2::from_shape_vec((1, attention_mask.len()), attention_mask)?;

		if image.is_some() {
			// Find the position of the <|image_1|> token, which is after <|user|>
			let image_token_position = input_ids.iter().position(|&id| id == USER_TOKEN_ID).unwrap_or(0);
			(inputs_embeds, attention_mask) = merge_text_and_image_embeddings(&inputs_embeds, &attention_mask, &visual_features, image_token_position);
		};
		(inputs_embeds, attention_mask)
	};

	// Initialize past_key_values for the transformer model
	// This is used to store the attention mechanism's state across multiple inference steps
	// The structure is:
	// - 64 elements (32 layers, each with a key and value)
	// - Each element is a 4D array with dimensions:
	//   1. Batch size (1)
	//   2. Number of attention heads (32)
	//   3. Sequence length (0 initially, will grow with each token generated)
	//   4. Head size (96)
	let mut past_key_values: Vec<Array4<f32>> = vec![Array4::zeros((1, 32, 0, 96)); 64];
	let mut generated_tokens: Vec<i64> = Vec::new();
	let mut next_inputs_embeds = inputs_embeds.clone();
	// Loop until <|end|> token is generated or max length is reached
	for _ in 0..MAX_LENGTH {
		// Prepare model inputs
		let model_inputs = {
			let mut model_inputs = ort::inputs![
				"inputs_embeds" => next_inputs_embeds.clone(),
				"attention_mask" => attention_mask.clone(),
			]?;
			for i in 0..32 {
				model_inputs.push((format!("past_key_values.{}.key", i).into(), Tensor::from_array(past_key_values[i * 2].view())?.into()));
				model_inputs.push((format!("past_key_values.{}.value", i).into(), Tensor::from_array(past_key_values[i * 2 + 1].view())?.into()));
			}
			model_inputs
		};

		// Run the model
		let model_outputs = generation_model.run(model_inputs)?;
		// Get the logits for the last token. Logits are unnormalized log probabilities, with a shape of [1, 1, VOCAB_SIZE],
		// where VOCAB_SIZE is the total number of unique tokens in the model's vocabulary.
		//
		// The current implementation uses a simple greedy decoding strategy:
		// - We select the token with the highest probability (argmax) from the logits.
		// - This approach always chooses the most likely next token, which can lead to deterministic and potentially repetitive
		//   outputs.
		//
		// Note: More advanced sampling strategies (e.g., temperature scaling, top-k, top-p sampling) are not implemented in the
		// current version.
		//
		// The selected token ID will be in the range [0, VOCAB_SIZE - 1].
		let logits: ArrayView<f32, _> = model_outputs["logits"].try_extract_tensor::<f32>()?.into_dimensionality::<Ix3>()?;
		let next_token_id = logits
			.slice(s![0, -1, ..VOCAB_SIZE])
			.iter()
			.enumerate()
			.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
			.unwrap()
			.0 as i64;

		if next_token_id == EOS_TOKEN_ID {
			break;
		}

		generated_tokens.push(next_token_id);
		// Log the generated text
		let output_ids: Vec<u32> = generated_tokens.iter().map(|&id| id as u32).collect();
		let generated_text = tokenizer.decode(&output_ids, false).unwrap();
		tracing::info!("Generated text: {}", generated_text);

		// Update current_embeds, attention_mask, and past_key_values for the next iteration
		let new_token_id = Array2::from_elem((1, 1), next_token_id);
		next_inputs_embeds = get_text_embedding(text_embedding_model, &new_token_id)?;
		attention_mask = Array2::ones((1, attention_mask.shape()[1] + 1));
		for i in 0..32 {
			past_key_values[i * 2] = model_outputs[format!("present.{}.key", i)]
				.try_extract_tensor::<f32>()?
				.into_dimensionality::<Ix4>()?
				.to_owned();
			past_key_values[i * 2 + 1] = model_outputs[format!("present.{}.value", i)]
				.try_extract_tensor::<f32>()?
				.into_dimensionality::<Ix4>()?
				.to_owned();
		}
	}

	Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
	tracing_subscriber::fmt().init(); // set up default subscriber with log level `INFO`

	let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
	let tokenizer = Tokenizer::from_file(data_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("Error loading tokenizer: {:?}", e))?;
	let vision_model = Session::builder()?.commit_from_file(data_dir.join(VISION_MODEL_NAME))?;
	let text_embedding_model = Session::builder()?.commit_from_file(data_dir.join(TEXT_EMBEDDING_MODEL_NAME))?;
	let generation_model = Session::builder()?.commit_from_file(data_dir.join(GENERATION_MODEL_NAME))?;

	// Generate text from text
	let image: Option<DynamicImage> = None;
	let text = "Who are you?".to_string();
	generate_text(&tokenizer, &vision_model, &text_embedding_model, &generation_model, &image, &text).await?;

	// Generate text from image and text
	let image: Option<DynamicImage> = Some(image::open(data_dir.join("example.jpg"))?);
	let text = "What is shown in this image?".to_string();
	generate_text(&tokenizer, &vision_model, &text_embedding_model, &generation_model, &image, &text).await?;

	Ok(())
}
