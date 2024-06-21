use std::{
	collections::HashMap,
	fmt::Debug,
	hash::Hash,
	marker::PhantomData,
	ptr::{self, NonNull},
	sync::Arc
};

use super::{ValueInner, ValueTypeMarker};
use crate::{
	memory::Allocator,
	ortsys,
	value::impl_tensor::{calculate_tensor_size, DynTensor},
	DynValue, Error, IntoTensorElementType, PrimitiveTensorElementType, Result, Tensor, TensorElementType, Value, ValueRef, ValueRefMut, ValueType
};

pub trait MapValueTypeMarker: ValueTypeMarker {
	crate::private_trait!();
}

#[derive(Debug)]
pub struct DynMapValueType;
impl ValueTypeMarker for DynMapValueType {
	crate::private_impl!();
}
impl MapValueTypeMarker for DynMapValueType {
	crate::private_impl!();
}

#[derive(Debug)]
pub struct MapValueType<K: IntoTensorElementType + Clone + Hash + Eq, V: IntoTensorElementType + Debug>(PhantomData<(K, V)>);
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> ValueTypeMarker for MapValueType<K, V> {
	crate::private_impl!();
}
impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug> MapValueTypeMarker for MapValueType<K, V> {
	crate::private_impl!();
}

pub type DynMap = Value<DynMapValueType>;
pub type Map<K, V> = Value<MapValueType<K, V>>;

pub type DynMapRef<'v> = ValueRef<'v, DynMapValueType>;
pub type DynMapRefMut<'v> = ValueRefMut<'v, DynMapValueType>;
pub type MapRef<'v, K, V> = ValueRef<'v, MapValueType<K, V>>;
pub type MapRefMut<'v, K, V> = ValueRefMut<'v, MapValueType<K, V>>;

impl<Type: MapValueTypeMarker + ?Sized> Value<Type> {
	pub fn try_extract_map<K: IntoTensorElementType + Clone + Hash + Eq, V: PrimitiveTensorElementType + Clone>(&self) -> Result<HashMap<K, V>> {
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

				let allocator = Allocator::default();

				let mut key_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 0, allocator.ptr.as_ptr(), &mut key_tensor_ptr) -> Error::ExtractMap; nonNull(key_tensor_ptr)];
				let key_value: DynTensor = unsafe { Value::from_ptr(NonNull::new_unchecked(key_tensor_ptr), None) };
				if K::into_tensor_element_type() != TensorElementType::String {
					let dtype = key_value.dtype()?;
					let (key_tensor_shape, key_tensor) = match dtype {
						ValueType::Tensor { ty, dimensions } => {
							let device = key_value.memory_info()?.allocation_device()?;
							if !device.is_cpu_accessible() {
								return Err(Error::TensorNotOnCpu(device.as_str()));
							}

							if ty == K::into_tensor_element_type() {
								let mut output_array_ptr: *mut K = ptr::null_mut();
								let output_array_ptr_ptr: *mut *mut K = &mut output_array_ptr;
								let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
								ortsys![unsafe GetTensorMutableData(key_tensor_ptr, output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

								let len = calculate_tensor_size(&dimensions);
								(dimensions, unsafe { std::slice::from_raw_parts(output_array_ptr, len) })
							} else {
								return Err(Error::DataTypeMismatch {
									actual: ty,
									requested: K::into_tensor_element_type()
								});
							}
						}
						_ => unreachable!()
					};

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
				} else {
					let (key_tensor_shape, key_tensor) = key_value.try_extract_raw_string_tensor()?;
					// SAFETY: `IntoTensorElementType` is a private trait, and we only map the `String` type to `TensorElementType::String`,
					// so at this point, `K` is **always** the `String` type, and this transmute really does nothing but please the type
					// checker.
					let key_tensor: Vec<K> = unsafe { std::mem::transmute(key_tensor) };

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
			}
			t => Err(Error::NotMap(t))
		}
	}
}

impl<K: PrimitiveTensorElementType + Debug + Clone + Hash + Eq + 'static, V: PrimitiveTensorElementType + Debug + Clone + 'static> Value<MapValueType<K, V>> {
	/// Creates a [`Map`] from an iterable emitting `K` and `V`.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::Map;
	/// # fn main() -> ort::Result<()> {
	/// let mut map = HashMap::<i64, f32>::new();
	/// map.insert(0, 1.0);
	/// map.insert(1, 2.0);
	/// map.insert(2, 3.0);
	///
	/// let value = Map::<i64, f32>::new(map)?;
	///
	/// assert_eq!(*value.extract_map().get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(data: impl IntoIterator<Item = (K, V)>) -> Result<Self> {
		let (keys, values): (Vec<K>, Vec<V>) = data.into_iter().unzip();
		Self::new_kv(Tensor::from_array((vec![keys.len()], keys))?, Tensor::from_array((vec![values.len()], values))?)
	}
}

impl<V: PrimitiveTensorElementType + Debug + Clone + 'static> Value<MapValueType<String, V>> {
	/// Creates a [`Map`] from an iterable emitting `K` and `V`.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::Map;
	/// # fn main() -> ort::Result<()> {
	/// let mut map = HashMap::<i64, f32>::new();
	/// map.insert(0, 1.0);
	/// map.insert(1, 2.0);
	/// map.insert(2, 3.0);
	///
	/// let value = Map::<i64, f32>::new(map)?;
	///
	/// assert_eq!(*value.extract_map().get(&0).unwrap(), 1.0);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn new(data: impl IntoIterator<Item = (String, V)>) -> Result<Self> {
		let (keys, values): (Vec<String>, Vec<V>) = data.into_iter().unzip();
		Self::new_kv(Tensor::from_string_array((vec![keys.len()], keys))?, Tensor::from_array((vec![values.len()], values))?)
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq + 'static, V: IntoTensorElementType + Debug + Clone + 'static> Value<MapValueType<K, V>> {
	/// Creates a [`Map`] from two tensors of keys & values respectively.
	///
	/// ```
	/// # use std::collections::HashMap;
	/// # use ort::{Map, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let keys = Tensor::<i64>::from_array(([4], vec![0, 1, 2, 3]))?;
	/// let values = Tensor::<f32>::from_array(([4], vec![1., 2., 3., 4.]))?;
	///
	/// let value = Map::new_kv(keys, values)?;
	///
	/// assert_eq!(*value.extract_map().get(&0).unwrap(), 1.0);
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
			inner: Arc::new(ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(values),
				_memory_info: None
			}),
			_markers: PhantomData
		})
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: PrimitiveTensorElementType + Debug + Clone> Value<MapValueType<K, V>> {
	pub fn extract_map(&self) -> HashMap<K, V> {
		self.try_extract_map().expect("Failed to extract map")
	}
}

impl<K: IntoTensorElementType + Debug + Clone + Hash + Eq, V: IntoTensorElementType + Debug + Clone> Value<MapValueType<K, V>> {
	/// Converts from a strongly-typed [`Map<K, V>`] to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast(self) -> DynMap {
		unsafe { std::mem::transmute(self) }
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a reference to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast_ref(&self) -> DynMapRef {
		DynMapRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Map<K, V>`] to a mutable reference to a type-erased [`DynMap`].
	#[inline]
	pub fn upcast_mut(&mut self) -> DynMapRefMut {
		DynMapRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}
}
