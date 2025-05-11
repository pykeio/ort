use super::{Graph, Model, Node, Opset, Outlet};
use crate::{
	Result,
	editor::ONNX_DOMAIN,
	session::builder::SessionBuilder,
	tensor::{Shape, SymbolicDimensions, TensorElementType},
	value::ValueType
};

#[test]
fn test_model_editor() -> Result<()> {
	let mut graph = Graph::new()?;
	graph.set_inputs([Outlet::new(
		"input",
		&ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([]),
			dimension_symbols: SymbolicDimensions::empty(0)
		}
	)?])?;
	graph.set_outputs([Outlet::new(
		"output",
		&ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([]),
			dimension_symbols: SymbolicDimensions::empty(0)
		}
	)?])?;
	graph.add_node(Node::new("Identity", ONNX_DOMAIN, "identity", ["input"], ["output"], [])?)?;

	let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22)?])?;
	model.add_graph(graph)?;

	let session = model.into_session(SessionBuilder::new()?)?;
	println!("{:?}", session.inputs);

	Ok(())
}
