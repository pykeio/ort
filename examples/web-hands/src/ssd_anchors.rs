use crate::detection::Point;

#[derive(Debug, Clone)]
pub struct Anchor {
	pub center: Point,
	pub height: f32,
	pub width: f32
}

#[inline]
fn stride_scale(min_scale: f32, max_scale: f32, stride_index: usize, num_strides: usize) -> f32 {
	if num_strides == 1 {
		(min_scale + max_scale) * 0.5
	} else {
		min_scale + (max_scale - min_scale) * (stride_index as f32) / ((num_strides as f32) - 1.0)
	}
}

pub fn calculate_ssd_anchors(
	input_size_width: u32,
	input_size_height: u32,
	min_scale: f32,
	max_scale: f32,
	num_layers: usize,
	strides: Vec<i32>
) -> Box<[Anchor]> {
	let mut anchors = Vec::new();

	let mut layer_id = 0;
	while layer_id < num_layers {
		let mut scales = Vec::new();

		let mut last_same_stride_layer = layer_id;
		while last_same_stride_layer < strides.len() && strides[last_same_stride_layer] == strides[layer_id] {
			let scale = stride_scale(min_scale, max_scale, last_same_stride_layer, strides.len());
			scales.push(scale);

			let scale_next = if last_same_stride_layer == strides.len() - 1 {
				1.0
			} else {
				stride_scale(min_scale, max_scale, last_same_stride_layer + 1, strides.len())
			};
			scales.push((scale * scale_next).sqrt());

			last_same_stride_layer += 1;
		}

		let stride = strides[layer_id];
		let feature_map_height = (input_size_height as f32 / stride as f32).ceil() as u32;
		let feature_map_width = (input_size_width as f32 / stride as f32).ceil() as u32;
		for y in 0..feature_map_height {
			for x in 0..feature_map_width {
				for _scale in &scales {
					anchors.push(Anchor {
						center: Point {
							x: (x as f32 + 0.5) / feature_map_width as f32,
							y: (y as f32 + 0.5) / feature_map_height as f32
						},
						width: 1.0,
						height: 1.0
					});
				}
			}
		}

		layer_id = last_same_stride_layer;
	}

	anchors.into_boxed_slice()
}
