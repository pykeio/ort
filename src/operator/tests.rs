use crate::{
	Result,
	operator::{
		Operator, OperatorDomain,
		io::{OperatorInput, OperatorOutput},
		kernel::{Kernel, KernelAttributes, KernelContext}
	},
	session::Session,
	tensor::TensorElementType,
	value::Tensor
};

struct CustomOpOne;

impl Operator for CustomOpOne {
	fn name(&self) -> &str {
		"CustomOpOne"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32), OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Float32)]
	}

	fn create_kernel(&self, _: &KernelAttributes) -> Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.ok_or_else(|| crate::Error::new("missing input"))?;
			let y = ctx.input(1)?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (x_shape, x) = x.try_extract_tensor::<f32>()?;
			let (y_shape, y) = y.try_extract_tensor::<f32>()?;

			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_tensor_mut::<f32>()?;
			for i in 0..y_shape.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0) as usize {
				if i % 2 == 0 {
					z_ref[i] = x[i];
				} else {
					z_ref[i] = y[i];
				}
			}
			Ok(())
		}))
	}
}

struct CustomOpTwo;

impl Operator for CustomOpTwo {
	fn name(&self) -> &str {
		"CustomOpTwo"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Int32)]
	}

	fn create_kernel(&self, _: &KernelAttributes) -> crate::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (x_shape, x) = x.try_extract_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_tensor_mut::<i32>()?;
			for i in 0..x_shape.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0) as usize {
				z_ref[i] = (x[i] * i as f32) as i32;
			}
			Ok(())
		}))
	}
}

#[test]
fn test_custom_ops() -> crate::Result<()> {
	let model = std::fs::read("tests/data/custom_op_test.onnx").expect("");
	let mut session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.commit_from_memory(&model)?;

	let allocator = session.allocator();
	let mut value1 = Tensor::<f32>::new(allocator, [3_usize, 5])?;
	{
		let (_, data) = value1.extract_tensor_mut();
		for datum in data {
			*datum = 0.;
		}
	}
	let mut value2 = Tensor::<f32>::new(allocator, [3_usize, 5])?;
	{
		let (_, data) = value2.extract_tensor_mut();
		for datum in data {
			*datum = 1.;
		}
	}
	let values = session.run(crate::inputs![&value1, &value2])?;
	assert_eq!(values[0].try_extract_tensor::<i32>()?.1, [0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0]);

	Ok(())
}
