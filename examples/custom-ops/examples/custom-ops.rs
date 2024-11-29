use ndarray::Array2;
use ort::{
	operator::{
		Operator, OperatorDomain,
		io::{OperatorInput, OperatorOutput},
		kernel::{Kernel, KernelAttributes, KernelContext}
	},
	session::Session,
	tensor::TensorElementType
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

	fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.unwrap();
			let y = ctx.input(1)?.unwrap();
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let (y_shape, y) = y.try_extract_raw_tensor::<f32>()?;

			let mut z = ctx.output(0, x_shape.to_vec())?.unwrap();
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<f32>()?;
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
	fn name(&self) -> &str {
		"CustomOpTwo"
	}
	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Int32)]
	}

	fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.unwrap();
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.unwrap();
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<i32>()?;
			for i in 0..x_shape.iter().copied().reduce(|acc, e| acc * e).unwrap() as usize {
				z_ref[i] = (x[i] * i as f32) as i32;
			}
			Ok(())
		}))
	}
}

fn main() -> ort::Result<()> {
	let session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.commit_from_file("tests/data/custom_op_test.onnx")?;

	let values = session.run(ort::inputs![Array2::<f32>::zeros((3, 5)), Array2::<f32>::ones((3, 5))]?)?;
	println!("{:?}", values[0].try_extract_tensor::<i32>()?);

	Ok(())
}
