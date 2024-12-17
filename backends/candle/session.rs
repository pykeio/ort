use std::{collections::HashMap, path::Path};

use candle_core::Tensor;
use candle_onnx::onnx::ModelProto;
use prost::Message;

#[derive(Default, Clone)]
pub struct SessionOptions;

pub struct Session {
	pub model: ModelProto
}

impl Session {
	pub fn from_buffer(_options: &SessionOptions, data: &[u8]) -> Result<Session, prost::DecodeError> {
		let model = ModelProto::decode(data)?;
		Ok(Session { model })
	}

	pub fn run(&self, inputs: HashMap<String, Tensor>) -> candle_core::Result<HashMap<String, Tensor>> {
		candle_onnx::simple_eval(&self.model, inputs)
	}
}
