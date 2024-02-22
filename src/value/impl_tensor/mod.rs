mod create;
mod extract;

pub use self::create::ToDimensions;

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use ndarray::{ArcArray1, Array1, CowArray};

	use crate::*;

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_value() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let value = Value::from_array(Array1::from_vec(v.clone()))?;
		assert!(value.is_tensor()?);
		assert_eq!(value.dtype()?.tensor_type(), Some(TensorElementType::Float32));
		assert_eq!(
			value.dtype()?,
			ValueType::Tensor {
				ty: TensorElementType::Float32,
				dimensions: vec![v.len() as i64]
			}
		);

		let (shape, data) = value.extract_raw_tensor::<f32>()?;
		assert_eq!(shape, vec![v.len() as i64]);
		assert_eq!(data, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc1 = ArcArray1::from_vec(v.clone());
		let mut arc2 = ArcArray1::clone(&arc1);
		let value = Value::from_array(&mut arc2)?;
		drop((arc1, arc2));

		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		let cow = CowArray::from(Array1::from_vec(v.clone()));
		let value = Value::from_array(&cow)?;
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		let owned = Array1::from_vec(v.clone());
		let value = Value::from_array(owned.view())?;
		drop(owned);
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];
		let value = Value::from_array((shape, Arc::clone(&arc)))?;
		drop(arc);
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_string_tensor_ndarray() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = Array1::from_vec(vec!["hello world".to_string(), "こんにちは世界".to_string()]);

		let value = Value::from_string_array(&allocator, v.view())?;
		let extracted = value.extract_tensor::<String>()?;
		assert_eq!(*extracted.view(), v.into_dyn().view());

		Ok(())
	}

	#[test]
	fn test_string_tensor_raw() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = vec!["hello world".to_string(), "こんにちは世界".to_string()];

		let value = Value::from_string_array(&allocator, (vec![v.len() as i64], v.clone().into_boxed_slice()))?;
		let (extracted_shape, extracted_view) = value.extract_raw_string_tensor()?;
		assert_eq!(extracted_shape, [v.len() as i64]);
		assert_eq!(extracted_view, v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_inputs() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let shape = [v.len()];
		let value_arc_box = Value::from_array((shape, Arc::new(v.clone().into_boxed_slice())))?;
		let value_box = Value::from_array((shape, v.clone().into_boxed_slice()))?;
		let value_vec = Value::from_array((shape, v.clone()))?;
		let value_slice = Value::from_array((shape, &v[..]))?;

		assert_eq!(value_arc_box.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_box.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_vec.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_slice.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}
}
