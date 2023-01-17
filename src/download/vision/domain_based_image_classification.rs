use crate::download::ModelUrl;

#[derive(Debug, Clone)]
pub enum DomainBasedImageClassification {
	/// Handwritten digit prediction using CNN.
	Mnist
}

impl ModelUrl for DomainBasedImageClassification {
	fn fetch_url(&self) -> &'static str {
		match self {
			DomainBasedImageClassification::Mnist => "https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-8.onnx"
		}
	}
}
