#![cfg(all(target_family = "wasm", target_os = "unknown"))]

use std::cell::RefCell;

use ort::{
	environment::Environment,
	inputs,
	memory::{AllocationDevice, Allocator, DeviceType},
	session::{OutputSelector, RunOptions, Session, builder::GraphOptimizationLevel},
	value::{DynTensor, Tensor, TensorElementType, TensorRefMut}
};
use ort_web::{SyncDirection, ValueExt};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

static MODEL_MUL_ADD: &[u8] = include_bytes!("../../../tests/data/mul_add.onnx");
static MODEL_OVERRIDABLE_INITIALIZER: &[u8] = include_bytes!("../../../tests/data/overridable_initializer.onnx");

thread_local! {
	static ENV: RefCell<Option<Environment>> = const { RefCell::new(None) };
}

async fn env() -> ort::Result<Environment> {
	ort::set_api(ort_web::api(ort_web::FEATURE_NONE).await.expect("failed to load runtime"));
	ENV.with_borrow_mut(|env| match env {
		Some(env) => Ok(env.clone()),
		None => {
			let new = ort::init().with_telemetry(false).build()?;
			*env = Some(new.clone());
			Ok(new)
		}
	})
}

async fn mul_add_session() -> ort::Result<Session> {
	Session::builder(&env().await?)?.commit_from_memory(MODEL_MUL_ADD).await
}

#[wasm_bindgen_test]
async fn test_input_output_info() -> ort::Result<()> {
	let session = mul_add_session().await?;

	assert_eq!(session.inputs().len(), 1);
	assert_eq!(session.inputs()[0].name(), "x");
	assert_eq!(session.inputs()[0].dtype().tensor_type(), Some(TensorElementType::Float32));
	assert_eq!(**session.inputs()[0].dtype().tensor_shape().unwrap(), [-1, 3]);
	let ort::value::ValueType::Tensor { dimension_symbols, .. } = session.inputs()[0].dtype() else {
		unreachable!()
	};
	assert_eq!(&**dimension_symbols, ["batch", ""]);
	assert_eq!(session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>(), ["y", "z"]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_run() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let options = RunOptions::new()?;
	let outputs = session
		.run_async(inputs![Tensor::from_array(([2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]))?], &options)
		.await?;

	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0, 5.0, 7.0, 9.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_sync_outputs() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let options = RunOptions::new()?;
	let mut outputs = session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await?;

	ort_web::sync_outputs(&mut outputs).await.unwrap();
	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	let mut y = outputs.remove("y").unwrap();
	y.sync(SyncDirection::Rust).await.unwrap();
	assert_eq!(y.try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_memory_growth() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let input_tensor = Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?;
	let lotsa_data = vec![1u8; 64 * 1024 * 1024];
	core::hint::black_box(&lotsa_data);

	let options = RunOptions::new()?;
	let outputs = session.run_async(inputs![input_tensor], &options).await?;
	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_new_input() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let mut input = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3])?;
	input.extract_tensor_mut().1.copy_from_slice(&[1.0, 2.0, 3.0]);

	let options = RunOptions::new()?;
	let outputs = session.run_async(inputs![input], &options).await?;
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_output_selector() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let options = RunOptions::new()?.with_outputs(OutputSelector::no_default().with("z"));
	let outputs = session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await?;
	assert_eq!(outputs.len(), 1);
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_preallocated_outputs() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::<f32>::new(&Allocator::default(), [1_usize, 3])?));
	let outputs = session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await?;
	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);
	assert_eq!(outputs["z"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0]);
	drop(outputs);

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::from_array(([1, 3], vec![0.0_f32; 3]))?));
	for input in [[1.0_f32, 2.0, 3.0], [3.0, 2.0, 1.0]] {
		let outputs = session
			.run_async(inputs![Tensor::from_array(([1, 3], input.to_vec()))?], &options)
			.await?;
		assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, input.map(|x| x * 2.0));
	}

	let options = RunOptions::new()?.with_outputs(OutputSelector::default().preallocate("y", Tensor::<f32>::new(&Allocator::default(), [2_usize, 3])?));
	let result = session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await;
	assert!(result.is_err());

	Ok(())
}

#[wasm_bindgen_test]
async fn test_run_options() -> ort::Result<()> {
	let mut session = mul_add_session().await?;

	let mut options = RunOptions::new()?;
	options.set_tag("xyzw")?;
	assert_eq!(options.tag()?, "xyzw");

	options.terminate()?;
	let result = session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await;
	assert!(result.is_err());
	drop(result);

	options.unterminate()?;
	session
		.run_async(inputs![Tensor::from_array(([1, 3], vec![1.0_f32, 2.0, 3.0]))?], &options)
		.await?;

	Ok(())
}

#[wasm_bindgen_test]
async fn test_session_options() -> ort::Result<()> {
	let env = env().await?;
	let mut session = Session::builder(&env)?
		.with_optimization_level(GraphOptimizationLevel::Disable)?
		.with_intra_threads(1)?
		.with_dimension_override("batch", 2)?
		.commit_from_memory(MODEL_MUL_ADD)
		.await?;

	let options = RunOptions::new()?;
	let outputs = session
		.run_async(inputs![Tensor::from_array(([2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]))?], &options)
		.await?;
	assert_eq!(outputs["y"].try_extract_tensor::<f32>()?.1, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_exotic_dtypes() -> ort::Result<()> {
	let env = env().await?;
	let mut session = Session::builder(&env)?.commit_from_memory(MODEL_OVERRIDABLE_INITIALIZER).await?;

	let options = RunOptions::new()?;
	let outputs = session
		.run_async(
			inputs![
				"Label" => Tensor::from_array(([1, 1], vec![true]))?,
				"F2" => Tensor::from_string_array(([1, 1], &["foobar"][..]))?
			],
			&options
		)
		.await?;
	assert_eq!(outputs["Label0"].try_extract_tensor::<bool>()?.1, [true]);
	assert_eq!(outputs["F20"].try_extract_strings()?.1, ["foobar"]);

	Ok(())
}

#[wasm_bindgen_test]
async fn test_tensors() -> ort::Result<()> {
	let _env = env().await?;

	let mut tensor = Tensor::<f32>::new(&Allocator::default(), [2_usize, 3])?;
	assert_eq!(tensor.extract_tensor().1, [0.0; 6]);
	tensor[[1, 1]] = 1.5;
	assert_eq!(tensor.extract_tensor().1, [0.0, 0.0, 0.0, 0.0, 1.5, 0.0]);
	assert_eq!(tensor.memory_info().allocation_device(), AllocationDevice::CPU);
	assert_eq!(tensor.memory_info().device_type(), DeviceType::CPU);

	let mut data = vec![0_i64; 4];
	{
		let mut tensor = TensorRefMut::from_array_view_mut(([4], &mut *data))?;
		tensor.extract_tensor_mut().1[2] = 7;
	}
	assert_eq!(data, [0, 0, 7, 0]);

	let strings = Tensor::from_string_array(([3], &["a", "", "雨が降っている！☔"][..]))?;
	assert_eq!(strings.try_extract_strings()?.1, ["a", "", "雨が降っている！☔"]);

	for ty in [TensorElementType::Float16, TensorElementType::Uint64, TensorElementType::Bool, TensorElementType::String] {
		let tensor = DynTensor::new(&Allocator::default(), ty, [4_usize])?;
		assert_eq!(*tensor.data_type(), ty);
	}

	let allocator = Allocator::default();
	assert_eq!(allocator.memory_info(), &allocator.memory_info().clone());

	Ok(())
}
