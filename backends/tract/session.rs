use std::{collections::HashMap, path::Path};

use tract_onnx::prelude::{Framework, Graph, InferenceModel, InferenceModelExt, OutletId, Tensor, TractResult, TypedFact, TypedOp};

use crate::{Environment, error::Error};

type OptimizedGraph = Graph<TypedFact, Box<dyn TypedOp>>;

#[derive(Default, Clone)]
pub struct SessionOptions {
	pub perform_optimizations: bool
}

pub struct Session {
	pub outlet_labels: HashMap<OutletId, String>,
	pub original_graph: OptimizedGraph
}

impl Session {
	pub fn from_buffer(env: &Environment, options: &SessionOptions, mut data: &[u8]) -> TractResult<Session> {
		let model = env.onnx.model_for_read(&mut data)?;
		let outlet_labels = model.outlet_labels.clone();
		let graph = if options.perform_optimizations { model.into_optimized()? } else { model.into_typed()? };
		Ok(Session { outlet_labels, original_graph: graph })
	}

	pub fn run(&self, inputs: HashMap<String, Tensor>) -> TractResult<HashMap<String, Tensor>> {
		unimplemented!()
	}
}
