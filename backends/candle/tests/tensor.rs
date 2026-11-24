use ort::{
	memory::Allocator,
	value::{DynTensor, Tensor, TensorElementType, TensorRefMut}
};

#[test]
fn test_tensors() -> ort::Result<()> {
	unsafe { ort::set_api(ort_candle::api()) };

	let mut tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	let ptr = tensor.data_ptr_mut().cast::<i64>();
	unsafe {
		*ptr.add(3) = 42;
	};

	let (_, extracted) = tensor.extract_tensor();
	assert_eq!(&extracted, &[0, 1, 2, 42, 4]);

	Ok(())
}

#[test]
fn test_allocated_tensor() -> ort::Result<()> {
	unsafe { ort::set_api(ort_candle::api()) };

	let mut tensor = Tensor::<f32>::new(&Allocator::default(), [2_usize, 3])?;
	assert_eq!(**tensor.shape(), [2, 3]);
	assert_eq!(tensor.extract_tensor().1, [0.0; 6]);
	tensor.extract_tensor_mut().1[4] = 1.5;
	assert_eq!(tensor[[1, 1]], 1.5);
	tensor[[0, 2]] = -1.0;
	assert_eq!(tensor.extract_tensor().1, [0.0, 0.0, -1.0, 0.0, 1.5, 0.0]);

	for (ty, size) in [
		(TensorElementType::Bool, 1),
		(TensorElementType::Uint8, 1),
		(TensorElementType::Int16, 2),
		(TensorElementType::Int32, 4),
		(TensorElementType::Int64, 8),
		(TensorElementType::Float16, 2),
		(TensorElementType::Bfloat16, 2),
		(TensorElementType::Float64, 8)
	] {
		let tensor = DynTensor::new(&Allocator::default(), ty, [4_usize])?;
		assert_eq!(*tensor.data_type(), ty);
		assert_eq!(tensor.size(), Some(4 * size));
	}

	Ok(())
}

#[test]
fn test_borrowed_tensor_aliases_data() -> ort::Result<()> {
	unsafe { ort::set_api(ort_candle::api()) };

	let mut data = vec![0.0_f32; 4];
	{
		let mut tensor = TensorRefMut::from_array_view_mut(([4], &mut *data))?;
		tensor.extract_tensor_mut().1[2] = 7.0;
	}
	assert_eq!(data, [0.0, 0.0, 7.0, 0.0]);

	Ok(())
}

#[test]
fn test_unsupported_types() -> ort::Result<()> {
	unsafe { ort::set_api(ort_candle::api()) };

	assert!(DynTensor::new(&Allocator::default(), TensorElementType::String, [4_usize]).is_err());
	assert!(DynTensor::new(&Allocator::default(), TensorElementType::Int8, [4_usize]).is_err());

	Ok(())
}

#[test]
fn test_comically_large_tensor() {
	unsafe { ort::set_api(ort_candle::api()) };

	assert!(Tensor::<f32>::new(&Allocator::default(), [1_i64 << 62]).is_err());
	assert!(Tensor::<i64>::new(&Allocator::default(), [1_i64 << 31, 1_i64 << 30]).is_err());
}
