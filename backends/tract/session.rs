use std::{
	collections::{HashMap, hash_map::Entry},
	hash::{BuildHasher, DefaultHasher, Hasher},
	sync::Arc
};

use parking_lot::Mutex;
use tract_onnx::{
	pb::ValueInfoProto,
	prelude::{Framework, Graph, InferenceModelExt, IntoTensor, SimplePlan, Tensor, TractResult, TypedFact, TypedOp}
};

use crate::Environment;

type OptimizedGraph = Graph<TypedFact, Box<dyn TypedOp>>;
type RunnableGraph = SimplePlan<TypedFact, Box<dyn TypedOp>, OptimizedGraph>;

#[derive(Default, Clone)]
pub struct SessionOptions {
	pub perform_optimizations: bool
}

pub struct SessionLockedInner {
	original_graph: Arc<OptimizedGraph>,
	graphs: HashMap<u64, RunnableGraph, PassthroughHashBuilder>
}

impl SessionLockedInner {
	pub fn new(original_graph: Arc<OptimizedGraph>) -> Self {
		Self {
			original_graph,
			graphs: HashMap::with_hasher(PassthroughHashBuilder)
		}
	}

	pub fn get_graph(&mut self, inputs: &[(String, Tensor)]) -> TractResult<&mut RunnableGraph> {
		let input_mark = Session::hash_inputs(inputs);
		match self.graphs.entry(input_mark) {
			Entry::Vacant(entry) => Ok(entry.insert(
				OptimizedGraph::clone(&*self.original_graph)
					.with_input_names(inputs.iter().map(|(n, _)| n))?
					.into_runnable()?
			)),
			Entry::Occupied(entry) => Ok(entry.into_mut())
		}
	}
}

pub struct Session {
	pub inputs: Vec<ValueInfoProto>,
	pub outputs: Vec<ValueInfoProto>,
	pub original_graph: Arc<OptimizedGraph>,
	locked_inner: Mutex<SessionLockedInner>
}

impl Session {
	pub fn from_buffer(env: &Environment, options: &SessionOptions, mut data: &[u8]) -> TractResult<Session> {
		let proto_model = env.onnx.proto_model_for_read(&mut data)?;
		let inputs = proto_model.graph.as_ref().map(|graph| graph.input.clone()).unwrap_or_default();
		let outputs = proto_model.graph.as_ref().map(|graph| graph.output.clone()).unwrap_or_default();

		let model = env.onnx.model_for_proto_model(&proto_model)?;
		let graph = Arc::new(if options.perform_optimizations { model.into_optimized()? } else { model.into_typed()? });
		Ok(Session {
			inputs,
			outputs,
			original_graph: Arc::clone(&graph),
			locked_inner: Mutex::new(SessionLockedInner::new(graph))
		})
	}

	fn hash_inputs(inputs: &[(String, Tensor)]) -> u64 {
		let mut hasher = DefaultHasher::new();
		for (name, _) in inputs {
			hasher.write_u64(name.len() as _);
			hasher.write(name.as_bytes());
			hasher.write_u8(0);
		}
		hasher.finish()
	}

	pub fn run(&self, inputs: Vec<(String, Tensor)>) -> TractResult<Vec<(String, Tensor)>> {
		let mut inner = self.locked_inner.lock();
		let graph = inner.get_graph(&inputs)?;
		let outputs = graph.run(inputs.into_iter().map(|(_, v)| tract_onnx::prelude::TValue::from(v)).collect())?;
		Ok(outputs
			.into_iter()
			.enumerate()
			.map(|(i, v)| (self.outputs[i].name.clone(), v.into_tensor()))
			.collect())
	}
}

struct PassthroughHasher(u64);

impl Hasher for PassthroughHasher {
	fn write(&mut self, _: &[u8]) {
		unreachable!()
	}

	fn write_u64(&mut self, i: u64) {
		self.0 = i;
	}

	fn finish(&self) -> u64 {
		self.0
	}
}

struct PassthroughHashBuilder;
impl BuildHasher for PassthroughHashBuilder {
	type Hasher = PassthroughHasher;

	fn build_hasher(&self) -> Self::Hasher {
		PassthroughHasher(0)
	}
}
