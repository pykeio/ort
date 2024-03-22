use std::{
	collections::HashMap,
	hash::Hash,
	ptr::{self, NonNull}
};

use super::ValueTypeMarker;
use crate::{memory::Allocator, ortsys, value::impl_tensor::DynTensor, Error, IntoTensorElementType, Result, Value, ValueType};

#[derive(Debug)]
pub struct MapValueType;
impl ValueTypeMarker for MapValueType {}
impl MapValueTypeMarker for MapValueType {}

pub trait MapValueTypeMarker: ValueTypeMarker {}

pub type Map = Value<MapValueType>;

impl<Type: MapValueTypeMarker + ?Sized> Value<Type> {
	pub fn try_extract_map<K: IntoTensorElementType + Clone + Hash + Eq, V: IntoTensorElementType + Clone>(&self, allocator: &Allocator) -> Result<HashMap<K, V>> {
		match self.dtype()? {
			ValueType::Map { key, value } => {
				let k_type = K::into_tensor_element_type();
				if k_type != key {
					return Err(Error::InvalidMapKeyType { expected: k_type, actual: key });
				}
				let v_type = V::into_tensor_element_type();
				if v_type != value {
					return Err(Error::InvalidMapValueType { expected: v_type, actual: value });
				}

				let mut key_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 0, allocator.ptr.as_ptr(), &mut key_tensor_ptr) -> Error::ExtractMap; nonNull(key_tensor_ptr)];
				let key_value: DynTensor = unsafe { Value::from_ptr(NonNull::new_unchecked(key_tensor_ptr), None) };
				let (key_tensor_shape, key_tensor) = key_value.try_extract_raw_tensor::<K>()?;

				let mut value_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 1, allocator.ptr.as_ptr(), &mut value_tensor_ptr) -> Error::ExtractMap; nonNull(value_tensor_ptr)];
				let value_value: DynTensor = unsafe { Value::from_ptr(NonNull::new_unchecked(value_tensor_ptr), None) };
				let (value_tensor_shape, value_tensor) = value_value.try_extract_raw_tensor::<V>()?;

				assert_eq!(key_tensor_shape.len(), 1);
				assert_eq!(value_tensor_shape.len(), 1);
				assert_eq!(key_tensor_shape[0], value_tensor_shape[0]);

				let mut vec = Vec::with_capacity(key_tensor_shape[0] as _);
				for i in 0..key_tensor_shape[0] as usize {
					vec.push((key_tensor[i].clone(), value_tensor[i].clone()));
				}
				Ok(vec.into_iter().collect())
			}
			t => Err(Error::NotMap(t))
		}
	}
}
