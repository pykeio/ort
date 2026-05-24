use std::slice;

use ort::{
	session::{RunOptions, Session},
	value::{Tensor, TensorRef}
};
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

mod detection;
mod resize;
mod ssd_anchors;

use crate::{
	detection::{Detection, NonMaximumSuppression, Point},
	resize::{LetterboxedResizer, RGBA8ToF32},
	ssd_anchors::{Anchor, calculate_ssd_anchors}
};

#[wasm_bindgen]
pub struct HandDetector {
	session: Session,
	resized_buffer: Box<[f32]>,
	anchors: Box<[Anchor]>,
	resizer: Option<LetterboxedResizer<RGBA8ToF32>>,
	nms: NonMaximumSuppression,
	candidates: Vec<Detection>,
	last_image_size: (u32, u32),
	min_confidence: f32,
	max_hands: usize,
	_run_options: RunOptions
}

const INPUT_SIZE: usize = 192;
const INPUT_SIZE_F32: f32 = INPUT_SIZE as f32;

#[wasm_bindgen]
impl HandDetector {
	pub async fn create(url: &str, max_hands: usize, min_confidence: f32, min_suppression_threshold: f32) -> Result<HandDetector, JsError> {
		Ok(Self {
			session: Session::builder()?.commit_from_url(url).await?,
			resized_buffer: vec![0.0_f32; INPUT_SIZE * INPUT_SIZE * 3].into_boxed_slice(),
			last_image_size: (0, 0),
			anchors: calculate_ssd_anchors(INPUT_SIZE as _, INPUT_SIZE as _, 0.1484375, 0.75, 4, vec![8, 16, 16, 16]),
			resizer: None,
			nms: NonMaximumSuppression::new(max_hands, min_suppression_threshold),
			candidates: Vec::new(),
			min_confidence,
			max_hands,
			_run_options: RunOptions::new()?
		})
	}

	pub async fn predict_from_canvas(&mut self, canvas: &HtmlCanvasElement, ctx: &CanvasRenderingContext2d) -> Result<JsValue, JsValue> {
		let (width, height) = (canvas.width(), canvas.height());
		let image_data = ctx.get_image_data(0.0, 0.0, width as _, height as _)?.data();
		self.predict(&image_data, (width, height))
			.await
			.map_err(|x| JsError::new(&x.to_string()).into())
			.and_then(|x| serde_wasm_bindgen::to_value(&x).map_err(|e| e.into()))
	}

	async fn predict(&mut self, image: &[u8], image_size: (u32, u32)) -> anyhow::Result<&[Detection]> {
		if image_size != self.last_image_size {
			self.resized_buffer = vec![0.0f32; INPUT_SIZE * INPUT_SIZE * 3].into_boxed_slice();
			self.resizer = Some(LetterboxedResizer::new(image_size.0 as _, image_size.1 as _, INPUT_SIZE, INPUT_SIZE, RGBA8ToF32::ZERO_TO_1));
			self.last_image_size = image_size;
		}

		let Some(resizer) = self.resizer.as_mut() else {
			unreachable!("no resizer");
		};

		self.candidates.clear();

		resizer.resize(unsafe { slice::from_raw_parts(image.as_ptr().cast(), image.len() / 4) }, unsafe {
			slice::from_raw_parts_mut(self.resized_buffer.as_mut_ptr().cast(), self.resized_buffer.len() / 3)
		});

		// Typically, `resized_buffer` would be a `Tensor` we created via `Tensor::new` rather than a normal Rust `Vec`.
		// However, use of `Tensor::new` is heavily discouraged with `ort-web`, since it creates the tensor in ONNX Runtime's
		// WASM context, requiring a useless copy of empty data before it can be used. `TensorRef::from_array_view` is the most
		// efficient solution.
		let input = TensorRef::from_array_view(([1, INPUT_SIZE, INPUT_SIZE, 3], &*self.resized_buffer))?;

		// Run the model! Note that `ort-web` only supports `run_async`.
		// `run_async` requires a `RunOptions`, but we don't use it. We keep a dummy one in the struct to avoid
		// needlessly reallocating on every `predict` call.
		let mut outputs = self.session.run_async(ort::inputs![input], &self._run_options).await?;

		// The outputs' data lies in ONNX Runtime's WASM context and is inaccessible to us. We need to synchronize them to get
		// their data. For more information on synchronization, see https://ort.pyke.io/backends/web#synchronization.
		//
		// We do use all of the model's outputs, so we can use the handy `sync_outputs` helper to synchronize all outputs at
		// once.
		ort_web::sync_outputs(&mut outputs).await?;

		// If the model outputs more tensors than you actually use, it's best to manually synchronize only the outputs you need
		// with `ort_web::TensorExt::sync()`. That would look like:
		// ```
		// use ort_web::{SyncDirection, TensorExt};
		// let classificators: Tensor<f32> = outputs.remove("Identity_1").unwrap().downcast().unwrap();
		// classificators.sync(SyncDirection::Rust).await?;
		// ```

		let classificators: Tensor<f32> = outputs.remove("Identity_1").unwrap().downcast().unwrap();
		let classificators = classificators.extract_tensor().1;
		let mut regressors: Tensor<f32> = outputs.remove("Identity").unwrap().downcast().unwrap();
		for (idx, candidate) in regressors.extract_tensor_mut().1.chunks_exact(18).enumerate() {
			let score = 1.0 / (1.0 + f32::exp(-classificators[idx])); // sigmoid
			if score < self.min_confidence {
				continue;
			}

			let anchor = &self.anchors[idx];
			let mut x = candidate[0] / INPUT_SIZE_F32 * anchor.width + anchor.center.x;
			let mut y = candidate[1] / INPUT_SIZE_F32 * anchor.height + anchor.center.y;

			// Undo letterboxing
			let rel_x_shift = resizer.x_shift as f32 / INPUT_SIZE_F32;
			let rel_y_shift = resizer.y_shift as f32 / INPUT_SIZE_F32;
			x = (x - rel_x_shift) / (1.0 - rel_x_shift - rel_x_shift);
			y = (y - rel_y_shift) / (1.0 - rel_y_shift - rel_y_shift);

			let w = (candidate[2] / INPUT_SIZE_F32) * anchor.width;
			let h = (candidate[3] / INPUT_SIZE_F32) * anchor.height;

			let detection = Detection::new(score, Point::new(x, y), w, h);
			self.candidates.push(detection);
		}

		self.candidates.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
		self.nms.process_inplace(&mut self.candidates);

		for detection in &mut self.candidates {
			// Detections are centered on the palm; shift boxes up by 50% to center near MCP joints, then expand by 2.6x to include
			// fingers.
			detection.to_roi(image_size, -0.5, 2.6);
		}

		Ok(&self.candidates[..self.max_hands.min(self.candidates.len())])
	}
}

#[wasm_bindgen]
pub async fn setup_ort() -> Result<(), JsError> {
	// Initialize `ort-web` without any hardware-accelerated EPs.
	// WebGPU seems to not like this model, at least on Firefox.
	ort::set_api(ort_web::api(ort_web::FEATURE_NONE).await?);

	// Set up a panic hook so panics are logged to the browser console.
	std::panic::set_hook(Box::new(console_error_panic_hook::hook));

	Ok(())
}
