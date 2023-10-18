use std::{
	fs,
	io::{self, BufRead, BufReader},
	path::Path,
	time::Duration
};

use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use ndarray::s;
use ort::{
	download::vision::ImageClassification, inputs, Environment, GraphOptimizationLevel, LoggingLevel, NdArrayExtensions, OrtDownloadError, OrtOwnedTensor,
	OrtResult, SessionBuilder
};
use test_log::test;

#[test]
fn squeezenet_mushroom() -> OrtResult<()> {
	const IMAGE_TO_LOAD: &str = "mushroom.png";

	let environment = Environment::builder()
		.with_name("integration_test")
		.with_log_level(LoggingLevel::Warning)
		.build()?
		.into_arc();

	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_downloaded(ImageClassification::SqueezeNet)
		.expect("Could not download model from file");

	let metadata = session.metadata()?;
	assert_eq!(metadata.name()?, "main");
	assert_eq!(metadata.producer()?, "");

	let class_labels = get_imagenet_labels()?;

	let input0_shape: Vec<usize> = session.inputs[0].map(|d| d.unwrap()).collect();
	let output0_shape: Vec<usize> = session.outputs[0].map(|d| d.unwrap()).collect();

	assert_eq!(input0_shape, [1, 3, 224, 224]);
	assert_eq!(output0_shape, [1, 1000]);

	// Load image and resize to model's shape, converting to RGB format
	let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(IMAGE_TO_LOAD))
		.unwrap()
		.resize(input0_shape[2] as u32, input0_shape[3] as u32, FilterType::Nearest)
		.to_rgb8();

	// Python:
	// # image[y, x, RGB]
	// # x==0 --> left
	// # y==0 --> top

	// See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
	// for pre-processing image.
	// WARNING: Note order of declaration of arguments: (_,c,j,i)
	let mut array = ndarray::CowArray::from(
		ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
			let pixel = image_buffer.get_pixel(i as u32, j as u32);
			let channels = pixel.channels();

			// range [0, 255] -> range [0, 1]
			(channels[c] as f32) / 255.0
		})
		.into_dyn()
	);

	// Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
	let mean = [0.485, 0.456, 0.406];
	let std = [0.229, 0.224, 0.225];
	for c in 0..3 {
		let mut channel_array = array.slice_mut(s![0, c, .., ..]);
		channel_array -= mean[c];
		channel_array /= std[c];
	}

	// Perform the inference
	let outputs = session.run(inputs![&array]?)?;

	// Downloaded model does not have a softmax as final layer; call softmax on second axis
	// and iterate on resulting probabilities, creating an index to later access labels.
	let output: OrtOwnedTensor<_> = outputs[0].extract_tensor()?;
	let mut probabilities: Vec<(usize, f32)> = output.view().softmax(ndarray::Axis(1)).iter().copied().enumerate().collect::<Vec<_>>();
	// Sort probabilities so highest is at beginning of vector.
	probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

	assert_eq!(class_labels[probabilities[0].0], "n07734744 mushroom", "Expecting class for {} to be a mushroom", IMAGE_TO_LOAD);
	assert_eq!(probabilities[0].0, 947, "Expecting class for {} to be a mushroom (index 947 in labels file)", IMAGE_TO_LOAD);

	Ok(())
}

fn get_imagenet_labels() -> Result<Vec<String>, OrtDownloadError> {
	// Download the ImageNet class labels, matching SqueezeNet's classes.
	let labels_path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("synset.txt");
	if !labels_path.exists() {
		let url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt";
		println!("Downloading {:?} to {:?}...", url, labels_path);
		let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()
            .map_err(Box::new)
            .map_err(OrtDownloadError::FetchError)?;

		assert!(resp.has("Content-Length"));
		let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
		println!("Downloading {} bytes...", len);

		let mut reader = resp.into_reader();
		let f = fs::File::create(&labels_path).unwrap();
		let mut writer = io::BufWriter::new(f);

		let bytes_io_count = io::copy(&mut reader, &mut writer).unwrap();
		assert_eq!(bytes_io_count, len as u64);
	}

	let file = BufReader::new(fs::File::open(labels_path).unwrap());
	file.lines().map(|line| line.map_err(OrtDownloadError::IoError)).collect()
}
