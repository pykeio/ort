use ort::{
	adapter::Adapter,
	execution_providers::CPUExecutionProvider,
	memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
	operator::{
		Operator, OperatorDomain,
		io::{OperatorInput, OperatorOutput},
		kernel::{Kernel, KernelAttributes, KernelContext}
	},
	session::{RunOptions, Session},
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

	fn create_kernel(&self, _: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
		Ok(Box::new(|ctx: &KernelContext| {
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
	let _env = ort::init().with_execution_providers([CPUExecutionProvider::default().build()]).commit()?;

	let mut session = Session::builder()?
		.with_operators(OperatorDomain::new("test.customop")?.add(CustomOpOne)?.add(CustomOpTwo)?)?
		.commit_from_file("tests/data/custom_op_test.onnx")?;

	let allocator = Allocator::new(&session, MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?.clone())?;

	let mut value1 = Tensor::<f32>::new(&allocator, [3_usize, 5])?;
	{
		let (_, data) = value1.extract_tensor_mut();
		for datum in data {
			*datum = 0.;
		}
	}
	let mut value2 = Tensor::<f32>::new(&allocator, [3_usize, 5])?;
	{
		let (_, data) = value2.extract_tensor_mut();
		for datum in data {
			*datum = 1.;
		}
	}
	{
		let values = session.run(ort::inputs![&value1, &value2])?;
		let _ = values[0].try_extract_array::<i32>()?;
	}

	{
		let _ = session.run(ort::inputs!["HyperSuperUltraLongInputNameLikeAReallyLongNameSoLongInFactThatItDoesntFitOnTheStackAsSpecifiedInTheSTACK_CSTR_ARRAY_MAX_TOTAL_ConstantDefinedInUtilDotRsThisStringIsSoLongThatImStartingToRunOutOfThingsToSaySoIllJustPutZeros000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000hi0000000000000000000000000000000000000000000000000000000000000000000000000000000" => &value1]);
		let _ = session.run(ort::inputs![
			"input1" => &value1,
			"input2" => &value1,
			"input3" => &value1,
			"input4" => &value1,
			"input5" => &value1,
			"input6" => &value1,
			"input7" => &value1,
			"input8" => &value1,
			"input9" => &value1,
			"input10" => &value1,
			"input11" => &value1,
			"input12" => &value1,
			"input_more_than_STACK_CSTR_ARRAY_MAX_ELEMENTS" => &value1,
		]);
	}

	{
		let adapter = Adapter::from_file("tests/data/adapter.orl", None)?;
		let mut options = RunOptions::new()?;
		options.add_adapter(&adapter)?;

		drop(adapter);

		let _ = session.run_with_options(
			ort::inputs![
				"phony" => &value1
			],
			&options
		);
	}

	{
		let metadata = session.metadata()?;

		let _ = metadata.custom_keys();
		let _ = metadata.description();
		let _ = metadata.domain();
		let _ = metadata.graph_description();
		let _ = metadata.name();
		let _ = metadata.producer();
		let _ = metadata.version();
	}

	{
		let mut binding = session.create_binding()?;
		binding.bind_input("input_1", &value1)?;
		binding.bind_input("input_2", &value2)?;
		binding.bind_output_to_device("output", &Allocator::default().memory_info())?;
		let _ = session.run_binding(&binding)?;
	}

	Ok(())
}
