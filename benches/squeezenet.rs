use std::{path::Path, sync::Arc};

use glassbench::{pretend_used, Bench};
use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use ndarray::{s, Array4};
use ort::{GraphOptimizationLevel, Session};

fn load_squeezenet_data() -> ort::Result<(Session, Array4<f32>)> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	ort::init().with_name("integration_test").commit()?;

	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/squeezenet.onnx")
		.expect("Could not download model from file");

	let input0_shape: &Vec<i64> = session.inputs[0].input_type.tensor_dimensions().expect("input0 to be a tensor type");

	let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(IMAGE_TO_LOAD))
		.unwrap()
		.resize(input0_shape[2] as u32, input0_shape[3] as u32, FilterType::Nearest)
		.to_rgb8();

	let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();
		(channels[c] as f32) / 255.0
	});

	let mean = [0.485, 0.456, 0.406];
	let std = [0.229, 0.224, 0.225];
	for c in 0..3 {
		let mut channel_array = array.slice_mut(s![0, c, .., ..]);
		channel_array -= mean[c];
		channel_array /= std[c];
	}

	Ok((session, array))
}

fn bench_squeezenet(bench: &mut Bench) {
	let (session, data) = load_squeezenet_data().unwrap();
	bench.task("ArrayView", |task| {
		task.iter(|| {
			pretend_used(session.run(ort::inputs![data.view()].unwrap()).unwrap());
		})
	});

	let raw = Arc::new(data.as_standard_layout().as_slice().unwrap().to_owned().into_boxed_slice());
	let shape: Vec<i64> = data.shape().iter().map(|c| *c as _).collect();
	bench.task("Raw data", |task| {
		task.iter(|| {
			pretend_used(session.run(ort::inputs![(shape.clone(), Arc::clone(&raw))].unwrap()).unwrap());
		})
	});
}

glassbench::glassbench!("SqueezeNet", bench_squeezenet,);
