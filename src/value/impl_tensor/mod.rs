mod create;
mod extract;

use std::{
	fmt::Debug,
	marker::PhantomData,
	ops::{Index, IndexMut},
	ptr::NonNull
};

use super::{DowncastableTarget, Value, ValueInner, ValueTypeMarker};
use crate::{ortsys, DynValue, Error, IntoTensorElementType, MemoryInfo, Result, ValueRef, ValueRefMut, ValueType};

pub trait TensorValueTypeMarker: ValueTypeMarker {}

#[derive(Debug)]
pub struct DynTensorValueType;
impl ValueTypeMarker for DynTensorValueType {}
impl TensorValueTypeMarker for DynTensorValueType {}

#[derive(Debug)]
pub struct TensorValueType<T: IntoTensorElementType + Debug>(PhantomData<T>);
impl<T: IntoTensorElementType + Debug> ValueTypeMarker for TensorValueType<T> {}
impl<T: IntoTensorElementType + Debug> TensorValueTypeMarker for TensorValueType<T> {}

pub type DynTensor = Value<DynTensorValueType>;
pub type Tensor<T> = Value<TensorValueType<T>>;

pub type DynTensorRef<'v> = ValueRef<'v, DynTensorValueType>;
pub type DynTensorRefMut<'v> = ValueRefMut<'v, DynTensorValueType>;
pub type TensorRef<'v, T> = ValueRef<'v, TensorValueType<T>>;
pub type TensorRefMut<'v, T> = ValueRefMut<'v, TensorValueType<T>>;

impl DowncastableTarget for DynTensorValueType {
	fn can_downcast(dtype: &ValueType) -> bool {
		matches!(dtype, ValueType::Tensor { .. })
	}
}

impl<Type: TensorValueTypeMarker + ?Sized> Value<Type> {
	/// Returns a mutable pointer to the tensor's data.
	pub fn data_ptr_mut(&mut self) -> Result<*mut ort_sys::c_void> {
		let mut buffer_ptr: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr(), &mut buffer_ptr) -> Error::GetTensorMutableData; nonNull(buffer_ptr)];
		Ok(buffer_ptr)
	}

	/// Returns a pointer to the tensor's data.
	pub fn data_ptr(&self) -> Result<*const ort_sys::c_void> {
		let mut buffer_ptr: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr(), &mut buffer_ptr) -> Error::GetTensorMutableData; nonNull(buffer_ptr)];
		Ok(buffer_ptr)
	}

	/// Returns information about the device this tensor is allocated on.
	pub fn memory_info(&self) -> Result<MemoryInfo> {
		let mut memory_info_ptr: *const ort_sys::OrtMemoryInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorMemoryInfo(self.ptr(), &mut memory_info_ptr) -> Error::GetTensorMemoryInfo; nonNull(memory_info_ptr)];
		Ok(MemoryInfo::from_raw(unsafe { NonNull::new_unchecked(memory_info_ptr.cast_mut()) }, false))
	}
}

impl<T: IntoTensorElementType + Debug> Tensor<T> {
	/// Converts from a strongly-typed [`Tensor<T>`] to a type-erased [`DynTensor`].
	#[inline]
	pub fn upcast(self) -> DynTensor {
		unsafe { std::mem::transmute(self) }
	}

	/// Converts from a strongly-typed [`Tensor<T>`] to a reference to a type-erased [`DynTensor`].
	#[inline]
	pub fn upcast_ref(&self) -> DynTensorRef {
		DynTensorRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Tensor<T>`] to a mutable reference to a type-erased [`DynTensor`].
	#[inline]
	pub fn upcast_mut(&mut self) -> DynTensorRefMut {
		DynTensorRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &self.inner { _session.clone() } else { None }
			)
		})
	}
}

impl<T: IntoTensorElementType + Debug> DowncastableTarget for TensorValueType<T> {
	fn can_downcast(dtype: &ValueType) -> bool {
		match dtype {
			ValueType::Tensor { ty, .. } => *ty == T::into_tensor_element_type(),
			_ => false
		}
	}
}

impl<T: IntoTensorElementType + Debug> From<Value<TensorValueType<T>>> for DynValue {
	fn from(value: Value<TensorValueType<T>>) -> Self {
		value.into_dyn()
	}
}
impl From<Value<DynTensorValueType>> for DynValue {
	fn from(value: Value<DynTensorValueType>) -> Self {
		value.into_dyn()
	}
}

impl<T: IntoTensorElementType + Clone + Debug, const N: usize> Index<[i64; N]> for Tensor<T> {
	type Output = T;
	fn index(&self, index: [i64; N]) -> &Self::Output {
		let mut out: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr(), index.as_ptr(), N as _, &mut out).expect("Failed to index tensor")];
		unsafe { &*out.cast::<T>() }
	}
}
impl<T: IntoTensorElementType + Clone + Debug, const N: usize> IndexMut<[i64; N]> for Tensor<T> {
	fn index_mut(&mut self, index: [i64; N]) -> &mut Self::Output {
		let mut out: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr(), index.as_ptr(), N as _, &mut out).expect("Failed to index tensor")];
		unsafe { &mut *out.cast::<T>() }
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use ndarray::{ArcArray1, Array1, CowArray};

	use crate::{Allocator, DynTensor, TensorElementType, Value, ValueType};

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

		let (shape, data) = value.extract_raw_tensor();
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

		assert_eq!(value.extract_raw_tensor().1, &v);

		let cow = CowArray::from(Array1::from_vec(v.clone()));
		let value = Value::from_array(&cow)?;
		assert_eq!(value.extract_raw_tensor().1, &v);

		let owned = Array1::from_vec(v.clone());
		let value = Value::from_array(owned.view())?;
		drop(owned);
		assert_eq!(value.extract_raw_tensor().1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];
		let value = Value::from_array((shape, Arc::clone(&arc)))?;
		drop(arc);
		assert_eq!(value.try_extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_string_tensor_ndarray() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = Array1::from_vec(vec!["hello world".to_string(), "こんにちは世界".to_string()]);

		let value = DynTensor::from_string_array(&allocator, v.view())?;
		let extracted = value.try_extract_string_tensor()?;
		assert_eq!(extracted, v.into_dyn());

		Ok(())
	}

	#[test]
	fn test_string_tensor_raw() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = vec!["hello world".to_string(), "こんにちは世界".to_string()];

		let value = DynTensor::from_string_array(&allocator, (vec![v.len() as i64], v.clone().into_boxed_slice()))?;
		let (extracted_shape, extracted_view) = value.try_extract_raw_string_tensor()?;
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

		assert_eq!(value_arc_box.extract_raw_tensor().1, &v);
		assert_eq!(value_box.extract_raw_tensor().1, &v);
		assert_eq!(value_vec.extract_raw_tensor().1, &v);
		assert_eq!(value_slice.extract_raw_tensor().1, &v);

		Ok(())
	}
}
