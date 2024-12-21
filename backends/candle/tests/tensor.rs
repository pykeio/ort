use ort::value::Tensor;

#[test]
fn test_tensors() -> ort::Result<()> {
	ort::set_api(ort_candle::api());

	let mut tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	let ptr = tensor.data_ptr_mut()?.cast::<i64>();
	unsafe {
		*ptr.add(3) = 42;
	};

	let (_, extracted) = tensor.extract_raw_tensor();
	assert_eq!(&extracted, &[0, 1, 2, 42, 4]);

	Ok(())
}
