//! Models for domain-based image classification.

use crate::download::ModelUrl;

/// Models for domain-based image classification.
#[derive(Debug, Clone)]
pub enum DomainBasedImageClassification {
	/// Handwritten digit prediction using CNN.
	Mnist
}

impl ModelUrl for DomainBasedImageClassification {
	fn model_url(&self) -> &'static str {
		match self {
			DomainBasedImageClassification::Mnist => {
				"https://github.com/onnx/models/raw/5faef4c33eba0395177850e1e31c4a6a9e634c82/vision/classification/mnist/model/mnist-8.onnx"
			}
		}
	}
}
