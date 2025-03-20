#![no_main]

// Below YoloV8 implementation partly copied from the "yolov8" example.
#[derive(Debug, Clone, Copy)]
struct BoundingBox {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32
}

#[rustfmt::skip]
static YOLOV8_CLASS_LABELS: [&str; 80] = [
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[no_mangle]
pub extern "C" fn alloc(size: usize) -> *mut std::os::raw::c_void {
	unsafe {
		let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<u8>()).expect("Cannot create memory layout.");
		return std::alloc::alloc(layout) as *mut std::os::raw::c_void;
	}
}

#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut std::os::raw::c_void, size: usize) {
	unsafe {
		let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<u8>()).expect("Cannot create memory layout.");
		std::alloc::dealloc(ptr as *mut u8, layout);
	}
}

#[no_mangle]
pub extern "C" fn detect_objects(ptr: *const u8, width: u32, height: u32) {
	ort::init()
		.with_global_thread_pool(ort::environment::GlobalThreadPoolOptions::default())
		.commit()
		.expect("Cannot initialize ort.");

	let mut builder = ort::session::Session::builder()
		.expect("Cannot create Session builder.")
		.with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
		.expect("Cannot optimize graph.")
		.with_parallel_execution(true)
		.expect("Cannot activate parallel execution.")
		.with_intra_threads(2)
		.expect("Cannot set intra thread count.")
		.with_inter_threads(1)
		.expect("Cannot set inter thread count.");

	let use_webgpu = true; // TODO: Make `use_webgpu` a parameter of `detect_objects`? Or say in README to change it here.
	if use_webgpu {
		use ort::execution_providers::ExecutionProvider;
		let ep = ort::execution_providers::WebGPUExecutionProvider::default();
		if ep.is_available().expect("Cannot check for availability of WebGPU ep.") {
			ep.register(&mut builder).expect("Cannot register WebGPU ep.");
		} else {
			println!("WebGPU ep is not available.");
		}
	}

	let mut session = builder
		.commit_from_memory(include_bytes!("../yolov8m.onnx"))
		.expect("Cannot commit model.");

	let image_data = unsafe { std::slice::from_raw_parts(ptr, (width * height * 4) as usize).to_vec() }; // Copy via .to_vec might be not necessary as memory lives long enough.
	let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(width, height, image_data).expect("Cannot parse input image.");
	let image640 = image::imageops::resize(&image, 640, 640, image::imageops::FilterType::CatmullRom);
	let tensor =
		ort::value::Tensor::from_array(ndarray::Array4::from_shape_fn((1, 3, 640, 640), |(_, c, y, x)| image640[(x as u32, y as u32)][c] as f32 / 255_f32))
			.unwrap();

	let outputs: ort::session::SessionOutputs = session.run(ort::inputs!["images" => tensor]).unwrap();
	let output = outputs["output0"].try_extract_array::<f32>().unwrap().t().into_owned();

	let mut boxes = Vec::new();
	let output = output.slice(ndarray::s![.., .., 0]);
	for row in output.axis_iter(ndarray::Axis(0)) {
		let row: Vec<_> = row.iter().copied().collect();
		let (class_id, prob) = row
			.iter()
			// skip bounding box coordinates
			.skip(4)
			.enumerate()
			.map(|(index, value)| (index, *value))
			.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			.unwrap();
		if prob < 0.5 {
			continue;
		}
		let label = YOLOV8_CLASS_LABELS[class_id];
		let xc = row[0] / 640. * (width as f32);
		let yc = row[1] / 640. * (height as f32);
		let w = row[2] / 640. * (width as f32);
		let h = row[3] / 640. * (height as f32);
		boxes.push((
			BoundingBox {
				x1: xc - w / 2.,
				y1: yc - h / 2.,
				x2: xc + w / 2.,
				y2: yc + h / 2.
			},
			label,
			prob
		));
	}

	// Just print the results to the console for now.
	println!("{:#?}", boxes);
}
