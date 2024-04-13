use std::{
	collections::HashMap,
	fmt::Debug,
	hash::Hash,
	marker::PhantomData,
	ptr::{self, NonNull}
};

use super::{ValueInner, ValueTypeMarker};
use crate::{
	memory::Allocator, ortsys, value::impl_tensor::DynTensor, DynValue, Error, IntoTensorElementType, Result, Tensor, Value, ValueRef, ValueRefMut, ValueType
};

pub trait MapValueTypeMarker: ValueTypeMarker {}

#[derive(Debug)]
pub struct DynMapValueType;
impl ValueTypeMarker for DynMapValueType {}
impl MapValueTypeMarker for DynMapValueType {}

#[derive(Debug)]
pub struct MapValueType<K: IntoTensorElementType + Clone + Hash + Eq, V: IntoTensorElementType + Debug>(PhantomData<(K, V)>);
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> ValueTypeMarker for MapValueType<K, V> {}
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> MapValueTypeMarker for MapValueType<K, V> {}

pub type DynMap = Value<DynMapValueType>;
pub type Map<K, V> = Value<MapValueType<K, V>>;

pub type DynMapRef<'v> = ValueRef<'v, DynMapValueType>;
pub type DynMapRefMut<'v> = ValueRefMut<'v, DynMapValueType>;
pub type MapRef<'v, K, V> = ValueRef<'v, MapValueType<K, V>>;
pub type MapRefMut<'v, K, V> = ValueRefMut<'v, MapValueType<K, V>>;

impl<Type: MapValueTypeMarker + ?Sized> Value<Type> {
	pub fn try_extract_map<K: IntoTensorElementType + Clone + Hash + Eq, V: IntoTensorElementType + Clone>(
		&self,
		allocator: &Allocator
	) -> Result<HashMap<K, V>> {
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

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq + 'static, V: IntoTensorElementType + Debug + Clone + 'static> Value<MapValueType<K, V>> {
	/// Creates a [`Map`] from an iterable emitting `K` and `V`.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::{Allocator, Map};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let mut map = HashMap::<i64, f32>::new();
	/// map.insert(0, 1.0);
	/// map.insert(1, 2.0);
	/// map.insert(2, 3.0);
	///
	/// let value = Map::new(map)?;
	///
	/// assert_eq!(*value.extract_map(&allocator).get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(data: impl IntoIterator<Item = (K, V)>) -> Result<Self> {
		let (keys, values): (Vec<K>, Vec<V>) = data.into_iter().unzip();
		Self::new_kv(Tensor::from_array((vec![keys.len()], keys))?, Tensor::from_array((vec![values.len()], values))?)
	}

	/// Creates a [`Map`] from two tensors of keys & values respectively.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::{Allocator, Map, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let keys = Tensor::<i64>::from_array(([4], vec![0, 1, 2, 3]))?;
	/// let values = Tensor::<f32>::from_array(([4], vec![1., 2., 3., 4.]))?;
	///
	/// let value = Map::new_kv(keys, values)?;
	///
	/// assert_eq!(*value.extract_map(&allocator).get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new_kv(keys: Tensor<K>, values: Tensor<V>) -> Result<Self> {
		let mut value_ptr = ptr::null_mut();
		let values: [DynValue; 2] = [keys.into_dyn(), values.into_dyn()];
		let value_ptrs: Vec<*const ort_sys::OrtValue> = values.iter().map(|c| c.ptr().cast_const()).collect();
		ortsys![
			unsafe CreateValue(value_ptrs.as_ptr(), 2, ort_sys::ONNXType::ONNX_TYPE_MAP, &mut value_ptr)
				-> Error::CreateMap;
			nonNull(value_ptr)
		];
		Ok(Value {
			inner: ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(values),
				_memory_info: None
			},
			_markers: PhantomData
		})
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug + Clone> Value<MapValueType<K, V>> {
	pub fn extract_map(&self, allocator: &Allocator) -> HashMap<K, V> {
		self.try_extract_map(allocator).expect("Failed to extract map")
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a type-erased [`DynMap`].
	#[inline]
	pub fn downcast(self) -> DynMap {
		unsafe { std::mem::transmute(self) }
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a reference to a type-erased [`DynMap`].
	#[inline]
	pub fn downcast_ref(&self) -> DynMapRef {
		DynMapRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a mutable reference to a type-erased [`DynMap`].
	#[inline]
	pub fn downcast_mut(&mut self) -> DynMapRefMut {
		DynMapRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}
}
