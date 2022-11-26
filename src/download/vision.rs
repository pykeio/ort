use super::ModelUrl;

pub mod body_face_gesture_analysis;
pub mod domain_based_image_classification;
pub mod image_classification;
pub mod image_manipulation;
pub mod object_detection_image_segmentation;

pub use body_face_gesture_analysis::BodyFaceGestureAnalysis;
pub use domain_based_image_classification::DomainBasedImageClassification;
pub use image_classification::ImageClassification;
pub use image_manipulation::ImageManipulation;
pub use object_detection_image_segmentation::ObjectDetectionImageSegmentation;

#[derive(Debug, Clone)]
pub enum Vision {
	BodyFaceGestureAnalysis(BodyFaceGestureAnalysis),
	DomainBasedImageClassification(DomainBasedImageClassification),
	ImageClassification(ImageClassification),
	ImageManipulation(ImageManipulation),
	ObjectDetectionImageSegmentation(ObjectDetectionImageSegmentation)
}

impl ModelUrl for Vision {
	fn fetch_url(&self) -> &'static str {
		match self {
			Vision::DomainBasedImageClassification(v) => v.fetch_url(),
			Vision::ImageClassification(v) => v.fetch_url(),
			Vision::ImageManipulation(v) => v.fetch_url(),
			Vision::ObjectDetectionImageSegmentation(v) => v.fetch_url(),
			Vision::BodyFaceGestureAnalysis(v) => v.fetch_url()
		}
	}
}
