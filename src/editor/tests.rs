use super::{Graph, Model, Node, Opset, Outlet};
use crate::{
	Result,
	editor::ONNX_DOMAIN,
	inputs,
	memory::Allocator,
	session::builder::SessionBuilder,
	tensor::{Shape, SymbolicDimensions, TensorElementType},
	value::{Tensor, ValueType}
};

#[test]
fn test_identity_graph() -> Result<()> {
	let mut graph = Graph::new()?;
	graph.set_inputs([Outlet::new(
		"input",
		ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([]),
			dimension_symbols: SymbolicDimensions::empty(0)
		}
	)])?;
	graph.set_outputs([Outlet::new(
		"output",
		ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([]),
			dimension_symbols: SymbolicDimensions::empty(0)
		}
	)])?;
	graph.add_node(Node::new("Identity", ONNX_DOMAIN, "identity", ["input"], ["output"], [])?)?;

	let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22)?])?;
	model.add_graph(graph)?;

	let mut session = model.into_session(SessionBuilder::new()?)?;
	let output = session
		.run(inputs![Tensor::<f32>::from_array((Shape::new([5]), vec![1.0f32; 5]))?])?
		.remove("output")
		.expect("");
	assert_eq!(output.try_extract_tensor::<f32>()?.1, [1., 1., 1., 1., 1.]);

	Ok(())
}

#[test]
fn test_mul_graph() -> Result<()> {
	let mut graph = Graph::new()?;
	graph.set_inputs([Outlet::new(
		"input",
		ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([5]),
			dimension_symbols: SymbolicDimensions::empty(1)
		}
	)])?;
	graph.set_outputs([Outlet::new(
		"output",
		ValueType::Tensor {
			ty: TensorElementType::Float32,
			shape: Shape::new([5]),
			dimension_symbols: SymbolicDimensions::empty(1)
		}
	)])?;
	graph.add_node(Node::new("Mul", ONNX_DOMAIN, "mul", ["input", "weight"], ["output"], [])?)?;
	let mut weight = Tensor::<f32>::new(&Allocator::default(), [5i64])?;
	{
		let (_, weight) = weight.extract_tensor_mut();
		weight[0] = 1.;
		weight[1] = 2.;
		weight[2] = 3.;
		weight[3] = 4.;
		weight[4] = 5.;
	}
	graph.add_initializer("weight", weight, false)?;

	let mut model = Model::new([Opset::new(ONNX_DOMAIN, 22)?])?;
	model.add_graph(graph)?;

	let mut session = model.into_session(SessionBuilder::new()?)?;
	let output = session
		.run(inputs![Tensor::<f32>::from_array((Shape::new([5]), vec![2.0f32; 5]))?])?
		.remove("output")
		.expect("");
	assert_eq!(output.try_extract_tensor::<f32>()?.1, [2., 4., 6., 8., 10.]);

	Ok(())
}
