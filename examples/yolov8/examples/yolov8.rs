#![allow(clippy::manual_retain)]

use std::path::Path;

use image::{GenericImageView, imageops::FilterType};
use ndarray::{Array, Axis, s};
use ort::{
	execution_providers::CUDAExecutionProvider,
	inputs,
	session::{Session, SessionOutputs}
};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{AsImageView, WindowOptions, event};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	(box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}

const YOLOV8M_URL: &str = "https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/yolov8m.onnx";

#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

#[show_image::main]
fn main() -> ort::Result<()> {
	tracing_subscriber::fmt::init();

	ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let original_img = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("baseball.jpg")).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(640, 640, FilterType::CatmullRom);
	let mut input = Array::zeros((1, 3, 640, 640));
	for pixel in img.pixels() {
		let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = (r as f32) / 255.;
		input[[0, 1, y, x]] = (g as f32) / 255.;
		input[[0, 2, y, x]] = (b as f32) / 255.;
	}

	let model = Session::builder()?.commit_from_url(YOLOV8M_URL)?;

	// Run YOLOv8 inference
	let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
	let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

	let mut boxes = Vec::new();
	let output = output.slice(s![.., .., 0]);
	for row in output.axis_iter(Axis(0)) {
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
		let xc = row[0] / 640. * (img_width as f32);
		let yc = row[1] / 640. * (img_height as f32);
		let w = row[2] / 640. * (img_width as f32);
		let h = row[3] / 640. * (img_height as f32);
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

	boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
	let mut result = Vec::new();

	while !boxes.is_empty() {
		result.push(boxes[0]);
		boxes = boxes
			.iter()
			.filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < 0.7)
			.copied()
			.collect();
	}

	let mut dt = DrawTarget::new(img_width as _, img_height as _);

	for (bbox, label, _confidence) in result {
		let mut pb = PathBuilder::new();
		pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
		let path = pb.finish();
		let color = match label {
			"baseball bat" => SolidSource { r: 0x00, g: 0x10, b: 0x80, a: 0x80 },
			"baseball glove" => SolidSource { r: 0x20, g: 0x80, b: 0x40, a: 0x80 },
			_ => SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 }
		};
		dt.stroke(
			&path,
			&Source::Solid(color),
			&StrokeStyle {
				join: LineJoin::Round,
				width: 4.,
				..StrokeStyle::default()
			},
			&DrawOptions::new()
		);
	}

	let overlay: show_image::Image = dt.into();

	let window = show_image::context()
		.run_function_wait(move |context| -> Result<_, String> {
			let mut window = context
				.create_window("ort + YOLOv8", WindowOptions {
					size: Some([img_width, img_height]),
					..WindowOptions::default()
				})
				.map_err(|e| e.to_string())?;
			window.set_image("baseball", &original_img.as_image_view().map_err(|e| e.to_string())?);
			window.set_overlay("yolo", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
			Ok(window.proxy())
		})
		.unwrap();

	for event in window.event_channel().unwrap() {
		if let event::WindowEvent::KeyboardInput(event) = event {
			if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
				break;
			}
		}
	}

	Ok(())
}
