use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

use crate::{
	Result,
	logging::LogLevel,
	operator::{Kernel, KernelAttributes, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput},
	session::Session,
	value::{Tensor, TensorElementType}
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
			let logger = ctx.logger()?;
			crate::log!(logger, Warning @ "$sentinel1$");

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
			let logger = ctx.logger()?;
			crate::log!(logger, Verbose @ "$sentinel2$");

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
	let logged_values = Arc::new((AtomicBool::new(false), AtomicBool::new(false)));
	let model = std::fs::read("tests/data/custom_op_test.onnx").expect("");
	let mut session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.with_logger(Arc::new({
			let logged_values = logged_values.clone();
			move |_level: LogLevel, _category: &str, _id: &str, _code_location: &str, message: &str| match message {
				"$sentinel1$" => logged_values.0.store(true, Ordering::Release),
				"$sentinel2$" => logged_values.1.store(true, Ordering::Release),
				_ => {}
			}
		}))?
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

	assert!(logged_values.0.load(Ordering::Acquire));
	assert!(logged_values.1.load(Ordering::Acquire));

	Ok(())
}
