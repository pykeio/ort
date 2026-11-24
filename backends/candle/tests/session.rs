use std::{
	path::{Path, PathBuf},
	sync::OnceLock
};

use ort::{
	environment::Environment,
	memory::Allocator,
	session::{OutputSelector, RunOptions, Session},
	value::{Tensor, TensorElementType, TensorRef, ValueType}
};

fn test_data(name: &str) -> PathBuf {
	Path::new(env!("CARGO_MANIFEST_DIR"))
		.parent()
		.unwrap()
		.parent()
		.unwrap()
		.join("tests")
		.join("data")
		.join(name)
}

fn test_env() -> ort::Result<Environment> {
	static ENV: OnceLock<Environment> = OnceLock::new();
	unsafe { ort::set_api(ort_candle::api()) };
	Ok(ENV.get_or_init(|| ort::init().build().unwrap()).clone())
}

#[test]
fn test_metadata() -> ort::Result<()> {
	let env = test_env()?;
	let session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	let metadata = session.metadata()?;
	assert_eq!(metadata.producer().as_deref(), Some("generate-models.py"));
	assert_eq!(metadata.name().as_deref(), Some("mul_add"));
	assert_eq!(metadata.domain().as_deref(), Some("io.pyke.ort"));
	assert_eq!(metadata.version(), Some(42));

	Ok(())
}

#[test]
fn test_io_info() -> ort::Result<()> {
	let env = test_env()?;
	let session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	assert_eq!(session.inputs().len(), 1);
	assert_eq!(session.inputs()[0].name(), "x");
	assert_eq!(session.inputs()[0].dtype().tensor_type(), Some(TensorElementType::Float32));
	assert_eq!(**session.inputs()[0].dtype().tensor_shape().unwrap(), [-1, 3]);
	let ValueType::Tensor { dimension_symbols, .. } = session.inputs()[0].dtype() else {
		unreachable!()
	};
	assert_eq!(&**dimension_symbols, ["batch", ""]);

	assert_eq!(session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>(), ["y", "z"]);

	Ok(())
}

#[test]
fn test_dyn_shape() -> ort::Result<()> {
	let env = test_env()?;
	let mut session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	for batch in [1, 4] {
		let input = ndarray::Array2::<f32>::from_shape_fn((batch, 3), |(i, j)| (i * 3 + j) as f32);
		let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;

		assert_eq!(outputs["y"].try_extract_array::<f32>()?, input.mapv(|x| x * 2.0).into_dyn());
		assert_eq!(outputs["z"].try_extract_array::<f32>()?, (input + ndarray::arr1(&[1.0, 2.0, 3.0])).into_dyn());
	}

	Ok(())
}

#[test]
fn test_output_selector() -> ort::Result<()> {
	let env = test_env()?;
	let mut session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	let options = RunOptions::new()?.with_outputs(OutputSelector::no_default().with("z"));
	let outputs = session.run_with_options(ort::inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)?;
	assert_eq!(outputs.len(), 1);
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	Ok(())
}

#[test]
fn test_preallocated_outputs() -> ort::Result<()> {
	let env = test_env()?;
	let mut session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::<f32>::new(&Allocator::default(), [1_usize, 3])?));
	let outputs = session.run_with_options(ort::inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)?;
	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);
	drop(outputs);

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::from_array(([1, 3], vec![0.0_f32; 3]))?));
	for input in [[1.0_f32, 2.0, 3.0], [3.0, 2.0, 1.0]] {
		let outputs = session.run_with_options(ort::inputs![TensorRef::from_array_view(([1, 3], input.as_slice()))?], &options)?;
		assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, input.map(|x| x * 2.0));
	}

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::<f32>::new(&Allocator::default(), [2_usize, 3])?));
	let result = session.run_with_options(ort::inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options);
	assert!(result.is_err());

	Ok(())
}

#[test]
fn test_io_binding() -> ort::Result<()> {
	let env = test_env()?;
	let mut session = Session::builder(&env)?.commit_from_file(test_data("mul_add.onnx"))?;

	let mut binding = session.create_binding()?;
	binding.bind_input("x", Tensor::from_array(([2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]))?)?;
	binding.bind_output("y", Tensor::<f32>::new(&Allocator::default(), [2_usize, 3])?)?;
	binding.bind_output_to_device("z", Allocator::default().memory_info())?;

	for _ in 0..2 {
		let outputs = session.run_binding(&binding)?;
		assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
		assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);
	}

	Ok(())
}
