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

pub trait TensorValueTypeMarker: ValueTypeMarker {
	crate::private_trait!();
}

#[derive(Debug)]
pub struct DynTensorValueType;
impl ValueTypeMarker for DynTensorValueType {
	crate::private_impl!();
}
impl TensorValueTypeMarker for DynTensorValueType {
	crate::private_impl!();
}

#[derive(Debug)]
pub struct TensorValueType<T: IntoTensorElementType + Debug>(PhantomData<T>);
impl<T: IntoTensorElementType + Debug> ValueTypeMarker for TensorValueType<T> {
	crate::private_impl!();
}
impl<T: IntoTensorElementType + Debug> TensorValueTypeMarker for TensorValueType<T> {
	crate::private_impl!();
}

/// A tensor [`Value`] whose data type is unknown.
pub type DynTensor = Value<DynTensorValueType>;
/// A strongly-typed tensor [`Value`].
pub type Tensor<T> = Value<TensorValueType<T>>;

/// A reference to a tensor [`Value`] whose data type is unknown.
pub type DynTensorRef<'v> = ValueRef<'v, DynTensorValueType>;
/// A mutable reference to a tensor [`Value`] whose data type is unknown.
pub type DynTensorRefMut<'v> = ValueRefMut<'v, DynTensorValueType>;
/// A reference to a strongly-typed tensor [`Value`].
pub type TensorRef<'v, T> = ValueRef<'v, TensorValueType<T>>;
/// A mutable reference to a strongly-typed tensor [`Value`].
pub type TensorRefMut<'v, T> = ValueRefMut<'v, TensorValueType<T>>;

impl DowncastableTarget for DynTensorValueType {
	fn can_downcast(dtype: &ValueType) -> bool {
		matches!(dtype, ValueType::Tensor { .. })
	}

	crate::private_impl!();
}

impl<Type: TensorValueTypeMarker + ?Sized> Value<Type> {
	/// Returns a mutable pointer to the tensor's data.
	///
	/// It's important to note that the resulting pointer may not point to CPU-accessible memory. In the case of a
	/// tensor created on a different EP device, e.g. via [`Tensor::new`], the pointer returned by this function may be
	/// a CUDA pointer, which would require a separate crate (like [`cudarc`](https://crates.io/crates/cudarc)) to access.
	/// Use [`Tensor::memory_info`] & [`MemoryInfo::allocation_device`] to check which device the data resides on before
	/// accessing it.
	///
	/// ```
	/// # use ort::{Allocator, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let mut tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	/// let ptr = tensor.data_ptr_mut()?.cast::<i64>();
	/// unsafe {
	/// 	*ptr.add(3) = 42;
	/// };
	///
	/// let (_, extracted) = tensor.extract_raw_tensor();
	/// assert_eq!(&extracted, &[0, 1, 2, 42, 4]);
	/// # Ok(())
	/// # }
	/// ```
	pub fn data_ptr_mut(&mut self) -> Result<*mut ort_sys::c_void> {
		let mut buffer_ptr: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr(), &mut buffer_ptr) -> Error::GetTensorMutableData; nonNull(buffer_ptr)];
		Ok(buffer_ptr)
	}

	/// Returns an immutable pointer to the tensor's underlying data.
	///
	/// It's important to note that the resulting pointer may not point to CPU-accessible memory. In the case of a
	/// tensor created on a different EP device, e.g. via [`Tensor::new`], the pointer returned by this function may be
	/// a CUDA pointer, which would require a separate crate (like [`cudarc`](https://crates.io/crates/cudarc)) to access.
	/// Use [`Tensor::memory_info`] & [`MemoryInfo::allocation_device`] to check which device the data resides on before
	/// accessing it.
	///
	/// ```
	/// # use ort::{Allocator, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	/// let ptr = tensor.data_ptr()?.cast::<i64>();
	/// assert_eq!(unsafe { *ptr.add(3) }, 3);
	/// # Ok(())
	/// # }
	/// ```
	pub fn data_ptr(&self) -> Result<*const ort_sys::c_void> {
		let mut buffer_ptr: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr(), &mut buffer_ptr) -> Error::GetTensorMutableData; nonNull(buffer_ptr)];
		Ok(buffer_ptr)
	}

	/// Returns information about the device this tensor is allocated on.
	///
	/// ```
	/// # use ort::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType, Session, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1, 3, 224, 224])?;
	/// // Tensors are allocated on CPU by default.
	/// assert_eq!(tensor.memory_info()?.allocation_device()?, AllocationDevice::CPU);
	///
	/// # if false {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let cuda_allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
	/// )?;
	/// let tensor = Tensor::<f32>::new(&cuda_allocator, [1, 3, 224, 224])?;
	/// assert_eq!(tensor.memory_info()?.allocation_device()?, AllocationDevice::CUDA);
	/// # }
	/// # Ok(())
	/// # }
	/// ```
	pub fn memory_info(&self) -> Result<MemoryInfo> {
		let mut memory_info_ptr: *const ort_sys::OrtMemoryInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorMemoryInfo(self.ptr(), &mut memory_info_ptr) -> Error::GetTensorMemoryInfo; nonNull(memory_info_ptr)];
		Ok(MemoryInfo::from_raw(unsafe { NonNull::new_unchecked(memory_info_ptr.cast_mut()) }, false))
	}
}

impl<T: IntoTensorElementType + Debug> Tensor<T> {
	/// Converts from a strongly-typed [`Tensor<T>`] to a type-erased [`DynTensor`].
	///
	/// ```
	/// # use ort::{Allocator, DynTensor, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1, 3, 224, 224])?;
	/// let tensor_dyn = tensor.upcast();
	/// assert!(tensor_dyn.try_extract_raw_tensor::<f32>().is_ok());
	/// assert!(tensor_dyn.try_extract_raw_tensor::<i64>().is_err());
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast(self) -> DynTensor {
		unsafe { std::mem::transmute(self) }
	}

	/// Creates a type-erased [`DynTensorRef`] from a strongly-typed [`Tensor<T>`].
	///
	/// ```
	/// # use ort::{Allocator, DynTensor, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1, 3, 224, 224])?;
	/// let tensor_dyn = tensor.upcast_ref();
	///
	/// let (_, original_extract) = tensor.extract_raw_tensor();
	/// let (_, ref_extract) = tensor_dyn.try_extract_raw_tensor::<f32>()?;
	/// assert_eq!(original_extract, ref_extract);
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast_ref(&self) -> DynTensorRef {
		DynTensorRef::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
			)
		})
	}

	/// Converts from a strongly-typed [`Tensor<T>`] to a mutable reference to a type-erased [`DynTensor`].
	///
	/// ```
	/// # use ort::{Allocator, DynTensor, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let mut tensor = Tensor::<i64>::from_array((vec![5], vec![1, 2, 3, 4, 5]))?;
	/// let mut tensor_dyn = tensor.upcast_mut();
	///
	/// let (_, mut_view) = tensor_dyn.try_extract_raw_tensor_mut::<i64>()?;
	/// mut_view[3] = 0;
	///
	/// let (_, original_view) = tensor.extract_raw_tensor();
	/// assert_eq!(original_view, &[1, 2, 3, 0, 5]);
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast_mut(&mut self) -> DynTensorRefMut {
		DynTensorRefMut::new(unsafe {
			Value::from_ptr_nodrop(
				NonNull::new_unchecked(self.ptr()),
				if let ValueInner::CppOwned { _session, .. } = &*self.inner { _session.clone() } else { None }
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

	crate::private_impl!();
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
		// Interestingly, the `TensorAt` API doesn't check if the tensor is on CPU, so we have to perform the check ourselves.
		if !self
			.memory_info()
			.expect("could not retrieve tensor memory info")
			.allocation_device()
			.expect("could not retrieve tensor allocation device")
			.is_cpu_accessible()
		{
			panic!("Cannot directly index a tensor which is not allocated on the CPU.");
		}

		let mut out: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr(), index.as_ptr(), N as _, &mut out).expect("Failed to index tensor")];
		unsafe { &*out.cast::<T>() }
	}
}
impl<T: IntoTensorElementType + Clone + Debug, const N: usize> IndexMut<[i64; N]> for Tensor<T> {
	fn index_mut(&mut self, index: [i64; N]) -> &mut Self::Output {
		if !self
			.memory_info()
			.expect("could not retrieve tensor memory info")
			.allocation_device()
			.expect("could not retrieve tensor allocation device")
			.is_cpu_accessible()
		{
			panic!("Cannot directly index a tensor which is not allocated on the CPU.");
		}

		let mut out: *mut ort_sys::c_void = std::ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr(), index.as_ptr(), N as _, &mut out).expect("Failed to index tensor")];
		unsafe { &mut *out.cast::<T>() }
	}
}

pub(crate) fn calculate_tensor_size(shape: &[i64]) -> usize {
	let mut size = 1usize;
	for dim in shape {
		if *dim < 0 {
			return 0;
		}
		size *= *dim as usize;
	}
	size
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;

	use ndarray::{ArcArray1, Array1, CowArray};

	use crate::{Tensor, TensorElementType, ValueType};

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_value() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let value = Tensor::from_array(Array1::from_vec(v.clone()))?;
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
		let value = Tensor::from_array(&mut arc2)?;
		drop((arc1, arc2));

		assert_eq!(value.extract_raw_tensor().1, &v);

		let cow = CowArray::from(Array1::from_vec(v.clone()));
		let value = Tensor::from_array(&cow)?;
		assert_eq!(value.extract_raw_tensor().1, &v);

		let owned = Array1::from_vec(v.clone());
		let value = Tensor::from_array(owned.view())?;
		drop(owned);
		assert_eq!(value.extract_raw_tensor().1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];
		let value = Tensor::from_array((shape, Arc::clone(&arc)))?;
		drop(arc);
		assert_eq!(value.try_extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_string_tensor_ndarray() -> crate::Result<()> {
		let v = Array1::from_vec(vec!["hello world".to_string(), "こんにちは世界".to_string()]);

		let value = Tensor::from_string_array(v.view())?;
		let extracted = value.try_extract_string_tensor()?;
		assert_eq!(extracted, v.into_dyn());

		Ok(())
	}

	#[test]
	fn test_string_tensor_raw() -> crate::Result<()> {
		let v = vec!["hello world".to_string(), "こんにちは世界".to_string()];

		let value = Tensor::from_string_array((vec![v.len() as i64], v.clone().into_boxed_slice()))?;
		let (extracted_shape, extracted_view) = value.try_extract_raw_string_tensor()?;
		assert_eq!(extracted_shape, [v.len() as i64]);
		assert_eq!(extracted_view, v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_inputs() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let shape = [v.len()];
		let value_arc_box = Tensor::from_array((shape, Arc::new(v.clone().into_boxed_slice())))?;
		let value_box = Tensor::from_array((shape, v.clone().into_boxed_slice()))?;
		let value_vec = Tensor::from_array((shape, v.clone()))?;
		let value_slice = Tensor::from_array((shape, &v[..]))?;

		assert_eq!(value_arc_box.extract_raw_tensor().1, &v);
		assert_eq!(value_box.extract_raw_tensor().1, &v);
		assert_eq!(value_vec.extract_raw_tensor().1, &v);
		assert_eq!(value_slice.extract_raw_tensor().1, &v);

		Ok(())
	}
}
