//! This file is a Rust implementation of the image processing code for Phi-3-vision-128k-instruct model.
//! The original Python version can be found at:
//! https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/image_processing_phi3_v.py
//!
//! The image transformation is configured as Phi3ImageTransform in the processor config:
//! https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/blob/main/cpu-int4-rtn-block-32-acc-level-4/processor_config.json
//!
//! This Rust implementation aims to provide similar functionality for preprocessing images
//! to be used with the Phi-3 vision model, adapting the original Python code to Rust.
use anyhow::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer};
use ndarray::{Array2, Array4, Array5, Axis, s};

/// see https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/blob/main/cpu-int4-rtn-block-32-acc-level-4/processor_config.json
/// NOTE: The default setting in processor_config.json is num_crops = 16,
/// but this is too slow for practical use. We use 1 here for better performance.
pub const NUM_CROPS: usize = 1;
pub const _NUM_IMG_TOKENS: usize = 144;

const OPENAI_CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const OPENAI_CLIP_STD: [f32; 3] = [0.26862954, 0.2613026, 0.2757771];

pub struct Phi3VImageProcessor {
	num_crops: usize,
	image_mean: Vec<f32>,
	image_std: Vec<f32>,
	do_convert_rgb: bool
}

impl Phi3VImageProcessor {
	pub fn new() -> Self {
		Self {
			num_crops: NUM_CROPS,
			image_mean: OPENAI_CLIP_MEAN.to_vec(),
			image_std: OPENAI_CLIP_STD.to_vec(),
			do_convert_rgb: true
		}
	}

	pub fn _calc_num_image_tokens(&self, image: &DynamicImage) -> usize {
		let transformed = self.hd_transform(image);
		let (width, height) = transformed.dimensions();
		self.calc_num_image_tokens_from_image_size(width, height)
	}

	pub fn calc_num_image_tokens_from_image_size(&self, width: u32, height: u32) -> usize {
		let (new_width, new_height) = self.calc_hd_transform_size(width, height);
		((new_height / 336 * new_width / 336 + 1) * 144 + 1 + (new_height / 336 + 1) * 12) as usize
	}

	pub fn preprocess(&self, image: &DynamicImage) -> Result<BatchFeature> {
		let rgb_image = if self.do_convert_rgb { image.to_rgb8() } else { image.to_rgb8() };
		let rgb_image = DynamicImage::ImageRgb8(rgb_image);

		let transformed = self.hd_transform(&rgb_image);
		let (width, height) = transformed.dimensions();
		let shapes = vec![height as i64, width as i64];
		let image_sizes = Array2::from_shape_vec((1, 2), shapes)?;

		let num_img_tokens = self.calc_num_image_tokens_from_image_size(width, height);

		let normalized = self.normalize_image(&transformed);
		let global_image = self.create_global_image(&normalized);
		let local_patches = self.create_local_patches(&normalized);

		let mut all_patches = vec![global_image];
		all_patches.extend(local_patches);

		let padded_images = self.pad_to_max_num_crops_tensor(&all_patches, self.num_crops + 1);
		let pixel_values = padded_images.insert_axis(Axis(0));

		Ok(BatchFeature {
			pixel_values,
			image_sizes,
			num_img_tokens: vec![num_img_tokens as i64]
		})
	}

	fn hd_transform(&self, image: &DynamicImage) -> DynamicImage {
		let (width, height) = image.dimensions();
		let mut transposed = false;
		let (width, height) = if width < height {
			transposed = true;
			(height, width)
		} else {
			(width, height)
		};

		let ratio = width as f32 / height as f32;
		let mut scale = 1;
		while (scale as f32 * (scale as f32 / ratio).ceil()) <= self.num_crops as f32 {
			scale += 1;
		}
		scale -= 1;

		let new_width = scale * 336;
		let new_height = (new_width as f32 / ratio) as u32;

		let resized = image.resize_exact(new_width, new_height, image::imageops::FilterType::Lanczos3);
		let padded = self.padding_336(&resized);

		if transposed { padded.rotate90() } else { padded }
	}

	fn padding_336(&self, image: &DynamicImage) -> DynamicImage {
		let (width, height) = image.dimensions();
		let tar = ((height as f32 / 336.0).ceil() * 336.0) as u32;
		let top_padding = (tar - height) / 2;
		let mut padded = ImageBuffer::from_pixel(width, tar, image::Rgba([255, 255, 255, 255]));
		image::imageops::overlay(&mut padded, image, 0, top_padding as i64);
		DynamicImage::ImageRgba8(padded)
	}

	fn calc_hd_transform_size(&self, width: u32, height: u32) -> (u32, u32) {
		let (width, height) = if width < height { (height, width) } else { (width, height) };

		let ratio = width as f32 / height as f32;
		let mut scale = 1;
		while (scale as f32 * (scale as f32 / ratio).ceil()) <= self.num_crops as f32 {
			scale += 1;
		}
		scale -= 1;

		let new_width = scale * 336;
		let new_height = (new_width as f32 / ratio) as u32;

		self.calc_padded_size(new_width, new_height)
	}

	fn calc_padded_size(&self, width: u32, height: u32) -> (u32, u32) {
		let target_height = ((height as f32 / 336.0).ceil() * 336.0) as u32;
		(width, target_height)
	}

	fn normalize_image(&self, image: &DynamicImage) -> Array4<f32> {
		let (width, height) = image.dimensions();
		let mut normalized = Array4::<f32>::zeros((1, 3, height as usize, width as usize));

		for (x, y, pixel) in image.pixels() {
			for c in 0..3 {
				normalized[[0, c, y as usize, x as usize]] = (pixel[c] as f32 / 255.0 - self.image_mean[c]) / self.image_std[c];
			}
		}

		normalized
	}

	fn create_global_image(&self, _image: &Array4<f32>) -> Array4<f32> {
		Array4::<f32>::zeros((1, 3, 336, 336))
	}

	fn create_local_patches(&self, image: &Array4<f32>) -> Vec<Array4<f32>> {
		let (_, _, height, width) = image.dim();
		let mut patches = Vec::new();

		for h in (0..height).step_by(336) {
			for w in (0..width).step_by(336) {
				let patch = image
					.slice(s![.., .., h..std::cmp::min(h + 336, height), w..std::cmp::min(w + 336, width)])
					.to_owned();
				patches.push(patch);
			}
		}

		patches
	}

	fn pad_to_max_num_crops_tensor(&self, patches: &[Array4<f32>], max_crops: usize) -> Array4<f32> {
		let (_, channels, height, width) = patches[0].dim();
		let mut padded = Array4::<f32>::zeros((max_crops, channels, height, width));

		for (i, patch) in patches.iter().enumerate() {
			if i >= max_crops {
				break;
			}
			// Remove the extra dimension when assigning
			padded.slice_mut(s![i, .., .., ..]).assign(&patch.slice(s![0, .., .., ..]));
		}

		padded
	}
}

pub struct BatchFeature {
	pub pixel_values: Array5<f32>,
	pub image_sizes: Array2<i64>,
	pub num_img_tokens: Vec<i64>
}
