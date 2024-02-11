use ndarray::Array2;
use ort::{Kernel, KernelAttributes, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput, Session, TensorElementType};

struct CustomOpOne;
struct CustomOpOneKernel;

impl Operator for CustomOpOne {
	type Kernel = CustomOpOneKernel;

	fn name() -> &'static str {
		"CustomOpOne"
	}

	fn create_kernel(_: KernelAttributes) -> ort::Result<Self::Kernel> {
		Ok(CustomOpOneKernel)
	}

	fn inputs() -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32), OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs() -> Vec<ort::OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Float32)]
	}
}

impl Kernel for CustomOpOneKernel {
	fn compute(&mut self, ctx: &mut KernelContext) -> ort::Result<()> {
		let x = ctx.input(0).unwrap();
		let y = ctx.input(1).unwrap();
		let (x_shape, x) = x.extract_raw_tensor::<f32>()?;
		let (y_shape, y) = y.extract_raw_tensor::<f32>()?;
		assert_eq!(&x_shape[..], &y_shape[..]);
		let mut z = ctx.output(0, x_shape).unwrap();
		let (_, data_ref) = z.extract_raw_tensor_mut::<f32>()?;
		for i in 0..y_shape.into_iter().reduce(|acc, e| acc * e).unwrap() as usize {
			data_ref[i] = x[i] + y[i];
		}
		Ok(())
	}
}

struct CustomOpTwo;
struct CustomOpTwoKernel;

impl Operator for CustomOpTwo {
	type Kernel = CustomOpTwoKernel;

	fn name() -> &'static str {
		"CustomOpTwo"
	}

	fn create_kernel(_: KernelAttributes) -> ort::Result<Self::Kernel> {
		Ok(CustomOpTwoKernel)
	}

	fn inputs() -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs() -> Vec<ort::OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Int32)]
	}
}

impl Kernel for CustomOpTwoKernel {
	fn compute(&mut self, ctx: &mut KernelContext) -> ort::Result<()> {
		let x = ctx.input(0).unwrap();
		let (x_shape, x) = x.extract_raw_tensor::<f32>()?;
		let mut z = ctx.output(0, x_shape.clone()).unwrap();
		let (_, data_ref) = z.extract_raw_tensor_mut::<i32>()?;
		for i in 0..x_shape.into_iter().reduce(|acc, e| acc * e).unwrap() as usize {
			data_ref[i] = x[i] as i32;
			println!("{}", x[i]);
		}
		Ok(())
	}
}

fn main() -> ort::Result<()> {
	let session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.with_model_from_file("custom_op_test.onnx")?;

	let values = session.run(ort::inputs![Array2::<f32>::zeros((3, 5)), Array2::<f32>::ones((3, 5))]?)?;
	println!("{:?}", values[0].extract_tensor::<i32>()?);

	Ok(())
}
