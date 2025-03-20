use std::sync::Arc;

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
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let (y_shape, y) = y.try_extract_raw_tensor::<f32>()?;

			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<f32>()?;
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
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<i32>()?;
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
	let mut value1 = Tensor::<f32>::new(allocator, [3, 5])?;
	{
		let (_, data) = value1.extract_raw_tensor_mut();
		for datum in data {
			*datum = 0.;
		}
	}
	let mut value2 = Tensor::<f32>::new(allocator, [3, 5])?;
	{
		let (_, data) = value2.extract_raw_tensor_mut();
		for datum in data {
			*datum = 1.;
		}
	}
	let values = session.run(crate::inputs![&value1, &value2])?;
	assert_eq!(values[0].try_extract_raw_tensor::<i32>()?.1, [0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0]);

	Ok(())
}

struct AttrTesterIntFloat;

impl Operator for AttrTesterIntFloat {
	fn name(&self) -> &str {
		"AttrTesterIntFloat"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Float32)]
	}

	fn infer_shape(&self, ctx: &mut super::ShapeInferenceContext) -> crate::Result<()> {
		assert!(matches!(ctx.attr("a_int"), Ok(1_i64)));
		assert!(matches!(ctx.attr("a_float"), Ok(2.0_f32)));
		assert!(matches!(ctx.attr::<Vec<i64>>("ints").as_deref(), Ok(&[3, 4, 5])));
		assert!(matches!(ctx.attr::<Vec<f32>>("floats").as_deref(), Ok(&[6., 7., 8.])));

		ctx.set_output(0, &ctx.inputs()[0])?;

		Ok(())
	}

	fn create_kernel(&self, _: &KernelAttributes) -> crate::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<f32>()?;
			for i in 0..x_shape.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0) as usize {
				z_ref[i] = x[i] * 2.;
			}
			Ok(())
		}))
	}
}

struct AttrTesterString;

impl Operator for AttrTesterString {
	fn name(&self) -> &str {
		"AttrTesterString"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::required(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::required(TensorElementType::Float32)]
	}

	fn infer_shape(&self, ctx: &mut super::ShapeInferenceContext) -> crate::Result<()> {
		assert!(matches!(ctx.attr::<String>("a_string").as_deref(), Ok("iamastring")));

		ctx.set_output(0, &ctx.inputs()[0])?;

		Ok(())
	}

	fn create_kernel(&self, _: &KernelAttributes) -> crate::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			let x = ctx.input(0)?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (x_shape, x) = x.try_extract_raw_tensor::<f32>()?;
			let mut z = ctx.output(0, x_shape.to_vec())?.ok_or_else(|| crate::Error::new("missing input"))?;
			let (_, z_ref) = z.try_extract_raw_tensor_mut::<f32>()?;
			for i in 0..x_shape.iter().copied().reduce(|acc, e| acc * e).unwrap_or(0) as usize {
				z_ref[i] = x[i] * 3.;
			}
			Ok(())
		}))
	}
}

#[test]
fn test_op_attrs() -> crate::Result<()> {
	let model = std::fs::read("tests/data/attr_tester.onnx").expect("");
	let mut session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(AttrTesterIntFloat)?.add(AttrTesterString)?)?
		.commit_from_memory(&model)?;

	let value1 = Tensor::from_array(([5], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]))?;

	let values = session.run(crate::inputs!["input_0" => &value1])?;
	assert_eq!(values[0].try_extract_raw_tensor::<f32>()?.1, [6.0, 12.0, 18.0, 24.0, 30.0]);

	Ok(())
}

struct CopyTensorArrayAllVariadic;

impl Operator for CopyTensorArrayAllVariadic {
	fn name(&self) -> &str {
		"CopyTensorArrayAllVariadic"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::variadic(1).homogenous(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::variadic(1).homogenous(TensorElementType::Float32)]
	}

	fn infer_shape(&self, ctx: &mut super::ShapeInferenceContext) -> crate::Result<()> {
		for (i, input) in ctx.inputs().into_iter().enumerate() {
			ctx.set_output(i, &input)?;
		}

		Ok(())
	}

	fn create_kernel(&self, _: &KernelAttributes) -> crate::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| copy_variadic(0, ctx)))
	}
}

fn copy_variadic(start: usize, ctx: &KernelContext) -> crate::Result<()> {
	for i in start..ctx.num_inputs()? {
		let input = ctx.input(i)?.ok_or_else(|| crate::Error::new("missing input"))?;
		let mut output = ctx
			.output(i, input.shape().clone())?
			.ok_or_else(|| crate::Error::new(format!("failed to allocate output {i}")))?;

		output
			.try_extract_raw_tensor_mut::<f32>()?
			.1
			.copy_from_slice(input.try_extract_raw_tensor()?.1);
	}
	Ok(())
}

struct CopyTensorArrayCombined;

impl Operator for CopyTensorArrayCombined {
	fn name(&self) -> &str {
		"CopyTensorArrayCombined"
	}

	fn inputs(&self) -> Vec<OperatorInput> {
		vec![OperatorInput::optional(TensorElementType::Float32), OperatorInput::variadic(0).homogenous(TensorElementType::Float32)]
	}

	fn outputs(&self) -> Vec<OperatorOutput> {
		vec![OperatorOutput::optional(TensorElementType::Float32), OperatorOutput::variadic(0).homogenous(TensorElementType::Float32)]
	}

	fn infer_shape(&self, ctx: &mut super::ShapeInferenceContext) -> crate::Result<()> {
		for (i, input) in ctx.inputs().into_iter().enumerate() {
			ctx.set_output(i, &input)?;
		}

		Ok(())
	}

	fn create_kernel(&self, _: &KernelAttributes) -> crate::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
			if let Ok(Some(input)) = ctx.input(0) {
				let mut output = ctx
					.output(0, input.shape().clone())?
					.ok_or_else(|| crate::Error::new("failed to allocate output 0"))?;

				output
					.try_extract_raw_tensor_mut::<f32>()?
					.1
					.copy_from_slice(input.try_extract_raw_tensor()?.1);
			}
			copy_variadic(1, ctx)
		}))
	}
}

#[test]
fn test_variadic_io() -> crate::Result<()> {
	let ops = Arc::new(
		OperatorDomain::new("test.customop")?
			.add(CopyTensorArrayAllVariadic)?
			.add(CopyTensorArrayCombined)?
	);

	let model = std::fs::read("tests/data/copy_2_inputs_2_outputs.onnx").expect("");
	let mut session = Session::builder()?.with_operators(Arc::clone(&ops))?.commit_from_memory(&model)?;

	let input0 = Tensor::from_array(([15], vec![1.1_f32, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2, 13.3, 14.4, 15.5]))?;
	let input1 = Tensor::from_array(([15], vec![15.5_f32, 14.4, 13.3, 12.2, 11.1, 10.0, 9.9, 8.8, 7.7, 6.6, 5.5, 4.4, 3.3, 2.2, 1.1]))?;

	let values = session.run(crate::inputs!["input_0" => &input0, "input_1" => &input1])?;
	assert_eq!(values[0].try_extract_raw_tensor::<f32>()?.1, input0.extract_raw_tensor().1);
	assert_eq!(values[1].try_extract_raw_tensor::<f32>()?.1, input1.extract_raw_tensor().1);

	let model = std::fs::read("tests/data/copy_3_inputs_3_outputs.onnx").expect("");
	let mut session = Session::builder()?.with_operators(Arc::clone(&ops))?.commit_from_memory(&model)?;

	let input2 = Tensor::from_array(([15], vec![6.6_f32, 7.7, 8.8, 9.9, 10.0, 1.1, 2.2, 3.3, 4.4, 5.5, 11.1, 12.2, 13.3, 14.4, 15.5]))?;
	let values = session.run(crate::inputs![
		"input_0" => &input0,
		"input_1" => &input1,
		"input_2" => &input2
	])?;
	assert_eq!(values[0].try_extract_raw_tensor::<f32>()?.1, input0.extract_raw_tensor().1);
	assert_eq!(values[1].try_extract_raw_tensor::<f32>()?.1, input1.extract_raw_tensor().1);
	assert_eq!(values[2].try_extract_raw_tensor::<f32>()?.1, input2.extract_raw_tensor().1);

	Ok(())
}
