use ort::{
	operator::{BoxedKernel, ComputeContext, Kernel, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput},
	session::Session,
	value::{Tensor, TensorElementType}
};

struct CustomOpOne;

impl Operator for CustomOpOne {
	type Kernel<'attr> = BoxedKernel<'attr>;

	fn name(&self) -> &str {
		"CustomOpOne"
	}

	fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
		[OperatorInput::required(TensorElementType::Float32), OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
		[OperatorOutput::required(TensorElementType::Float32)]
	}

	fn create_kernel<'attr>(&self, _: &KernelContext<'attr>) -> ort::Result<Self::Kernel<'attr>> {
		Ok(Box::new(|ctx: &ComputeContext| {
			let x = ctx.input(0)?.unwrap();
			let y = ctx.input(1)?.unwrap();
			let (x_shape, x) = x.try_extract_tensor::<f32>()?;
			let (y_shape, y) = y.try_extract_tensor::<f32>()?;

			let mut z = ctx.output(0, x_shape.to_vec())?.unwrap();
			let (_, z_ref) = z.try_extract_tensor_mut::<f32>()?;
			for i in 0..y_shape.iter().copied().reduce(|acc, e| acc * e).unwrap() as usize {
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
	type Kernel<'attr> = BoxedKernel<'attr>;

	const INPLACES: &[(u32, u32)] = &[(0, 0)];

	fn name(&self) -> &str {
		"CustomOpTwo"
	}

	fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
		[OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
		[OperatorOutput::required(TensorElementType::Int32)]
	}

	fn create_kernel<'attr>(&self, _: &KernelContext<'attr>) -> ort::Result<Self::Kernel<'attr>> {
		Ok(Box::new(|ctx: &ComputeContext| {
			let x = ctx.input(0)?.unwrap();
			let (x_shape, x) = x.try_extract_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.unwrap();
			let (_, z_ref) = z.try_extract_tensor_mut::<i32>()?;
			for i in 0..x_shape.iter().copied().reduce(|acc, e| acc * e).unwrap() as usize {
				z_ref[i] = (x[i] * i as f32) as i32;
			}
			Ok(())
		}))
	}
}

fn main() -> ort::Result<()> {
	let mut session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.commit_from_file("tests/data/custom_op_test.onnx")?;

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
	let values = session.run(ort::inputs![&value1, &value2])?;
	println!("{:?}", values[0].try_extract_array::<i32>()?);

	Ok(())
}
