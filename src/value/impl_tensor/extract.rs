use std::{fmt::Debug, ptr, string::FromUtf8Error};

#[cfg(feature = "ndarray")]
use ndarray::IxDyn;

use super::TensorValueTypeMarker;
#[cfg(feature = "ndarray")]
use crate::tensor::{extract_primitive_array, extract_primitive_array_mut};
use crate::{
	ortsys, tensor::TensorElementType, value::impl_tensor::calculate_tensor_size, Error, PrimitiveTensorElementType, Result, Tensor, Value, ValueType
};

impl<Type: TensorValueTypeMarker + ?Sized> Value<Type> {
	/// Attempt to extract the underlying data of type `T` into a read-only [`ndarray::ArrayView`].
	///
	/// See also:
	/// - the mutable counterpart of this function, [`Tensor::try_extract_tensor_mut`].
	/// - the infallible counterpart, [`Tensor::extract_tensor`], for typed [`Tensor<T>`]s.
	/// - the alternative function for strings, [`Tensor::try_extract_string_tensor`].
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let value = Value::from_array(array.view())?;
	///
	/// let extracted = value.try_extract_tensor::<f32>()?;
	/// assert_eq!(array.into_dyn(), extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`crate::DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the
	///   infallible [`Tensor::extract_tensor`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	/// - The tensor's data is not allocated in CPU memory.
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_tensor<T: PrimitiveTensorElementType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == T::into_tensor_element_type() {
					Ok(extract_primitive_array(IxDyn(&dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>()), self.ptr())?)
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: T::into_tensor_element_type()
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the scalar from a tensor of type `T`.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// let value = Value::from_array(((), vec![3.14_f32]))?;
	///
	/// let extracted = value.try_extract_scalar::<f32>()?;
	/// assert_eq!(extracted, 3.14);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - The tensor is not 0-dimensional.
	/// - The provided type `T` does not match the tensor's element type.
	/// - This is a [`crate::DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the
	///   infallible [`Tensor::extract_tensor`] instead)*
	/// - The tensor's data is not allocated in CPU memory.
	pub fn try_extract_scalar<T: PrimitiveTensorElementType + Copy>(&self) -> Result<T> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if !dimensions.is_empty() {
					return Err(Error::TensorNot0Dimensional(dimensions.len()));
				}

				if ty == T::into_tensor_element_type() {
					let mut output_array_ptr: *mut T = ptr::null_mut();
					let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
					let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
					ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

					Ok(unsafe { *output_array_ptr })
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: T::into_tensor_element_type()
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the underlying data of type `T` into a mutable read-only [`ndarray::ArrayViewMut`].
	///
	/// See also the infallible counterpart, [`Tensor::extract_tensor_mut`], for typed [`Tensor<T>`]s.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let mut value = Value::from_array(array.view())?;
	///
	/// let mut extracted = value.try_extract_tensor_mut::<f32>()?;
	/// extracted[[0, 0, 0, 1]] = 0.0;
	///
	/// let mut array = array.into_dyn();
	/// assert_ne!(array, extracted);
	/// array[[0, 0, 0, 1]] = 0.0;
	/// assert_eq!(array, extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`crate::DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the
	///   infallible [`Tensor::extract_tensor_mut`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_tensor_mut<T: PrimitiveTensorElementType>(&mut self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == T::into_tensor_element_type() {
					Ok(extract_primitive_array_mut(IxDyn(&dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>()), self.ptr())?)
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: T::into_tensor_element_type()
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and an
	/// immutable view into its data.
	///
	/// See also:
	/// - the mutable counterpart of this function, [`Tensor::try_extract_raw_tensor_mut`].
	/// - the infallible counterpart, [`Tensor::extract_raw_tensor`], for typed [`Tensor<T>`]s.
	/// - the alternative function for strings, [`Tensor::try_extract_raw_string_tensor`].
	///
	/// ```
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let value = Value::from_array(([array.len()], array.clone().into_boxed_slice()))?;
	///
	/// let (extracted_shape, extracted_data) = value.try_extract_raw_tensor::<i64>()?;
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`crate::DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the
	///   infallible [`Tensor::extract_raw_tensor`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	pub fn try_extract_raw_tensor<T: PrimitiveTensorElementType>(&self) -> Result<(Vec<i64>, &[T])> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == T::into_tensor_element_type() {
					let mut output_array_ptr: *mut T = ptr::null_mut();
					let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
					let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
					ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

					let len = calculate_tensor_size(&dimensions);
					Ok((dimensions, unsafe { std::slice::from_raw_parts(output_array_ptr, len) }))
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: T::into_tensor_element_type()
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and a
	/// mutable view into its data.
	///
	/// See also the infallible counterpart, [`Tensor::extract_raw_tensor_mut`], for typed [`Tensor<T>`]s.
	///
	/// ```
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let mut value = Value::from_array(([array.len()], array.clone().into_boxed_slice()))?;
	///
	/// let (extracted_shape, extracted_data) = value.try_extract_raw_tensor_mut::<i64>()?;
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`crate::DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the
	///   infallible [`Tensor::extract_raw_tensor_mut`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	pub fn try_extract_raw_tensor_mut<T: PrimitiveTensorElementType>(&mut self) -> Result<(Vec<i64>, &mut [T])> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == T::into_tensor_element_type() {
					let mut output_array_ptr: *mut T = ptr::null_mut();
					let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
					let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
					ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

					let len = calculate_tensor_size(&dimensions);
					Ok((dimensions, unsafe { std::slice::from_raw_parts_mut(output_array_ptr, len) }))
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: T::into_tensor_element_type()
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the underlying data into a Rust `ndarray`.
	///
	/// ```
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array1::from_vec(vec!["hello", "world"]);
	/// let tensor = Tensor::from_string_array(array.clone())?;
	///
	/// let extracted = tensor.try_extract_string_tensor()?;
	/// assert_eq!(array.into_dyn(), extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_string_tensor(&self) -> Result<ndarray::ArrayD<String>> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == TensorElementType::String {
					let len = calculate_tensor_size(&dimensions);

					// Total length of string data, not including \0 suffix
					let mut total_length: ort_sys::size_t = 0;
					ortsys![unsafe GetStringTensorDataLength(self.ptr(), &mut total_length) -> Error::GetStringTensorDataLength];

					// In the JNI impl of this, tensor_element_len was included in addition to total_length,
					// but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
					// don't seem to be written to in practice either.
					// If the string data actually did go farther, it would panic below when using the offset
					// data to get slices for each string.
					let mut string_contents = vec![0u8; total_length as _];
					// one extra slot so that the total length can go in the last one, making all per-string
					// length calculations easy
					let mut offsets = vec![0; (len + 1) as _];

					ortsys![unsafe GetStringTensorContent(self.ptr(), string_contents.as_mut_ptr().cast(), total_length, offsets.as_mut_ptr(), len as _) -> Error::GetStringTensorContent];

					// final offset = overall length so that per-string length calculations work for the last string
					debug_assert_eq!(0, offsets[len]);
					offsets[len] = total_length;

					let strings = offsets
						// offsets has 1 extra offset past the end so that all windows work
						.windows(2)
						.map(|w| {
							let slice = &string_contents[w[0] as _..w[1] as _];
							String::from_utf8(slice.into())
						})
						.collect::<Result<Vec<String>, FromUtf8Error>>()
						.map_err(Error::StringFromUtf8Error)?;

					Ok(ndarray::Array::from_shape_vec(IxDyn(&dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>()), strings)
						.expect("Shape extracted from tensor didn't match tensor contents"))
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: TensorElementType::String
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Attempt to extract the underlying string data into a "raw" data tuple, consisting of the tensor's dimensions and
	/// an owned `Vec` of its data.
	///
	/// ```
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec!["hello", "world"];
	/// let tensor = Tensor::from_string_array(([array.len()], array.clone().into_boxed_slice()))?;
	///
	/// let (extracted_shape, extracted_data) = tensor.try_extract_raw_string_tensor()?;
	/// assert_eq!(extracted_data, array);
	/// assert_eq!(extracted_shape, [2]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn try_extract_raw_string_tensor(&self) -> Result<(Vec<i64>, Vec<String>)> {
		let dtype = self.dtype()?;
		match dtype {
			ValueType::Tensor { ty, dimensions } => {
				let device = self.memory_info()?.allocation_device()?;
				if !device.is_cpu_accessible() {
					return Err(Error::TensorNotOnCpu(device.as_str()));
				}

				if ty == TensorElementType::String {
					let len = calculate_tensor_size(&dimensions);

					// Total length of string data, not including \0 suffix
					let mut total_length: ort_sys::size_t = 0;
					ortsys![unsafe GetStringTensorDataLength(self.ptr(), &mut total_length) -> Error::GetStringTensorDataLength];

					// In the JNI impl of this, tensor_element_len was included in addition to total_length,
					// but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
					// don't seem to be written to in practice either.
					// If the string data actually did go farther, it would panic below when using the offset
					// data to get slices for each string.
					let mut string_contents = vec![0u8; total_length as _];
					// one extra slot so that the total length can go in the last one, making all per-string
					// length calculations easy
					let mut offsets = vec![0; (len + 1) as _];

					ortsys![unsafe GetStringTensorContent(self.ptr(), string_contents.as_mut_ptr().cast(), total_length, offsets.as_mut_ptr(), len as _) -> Error::GetStringTensorContent];

					// final offset = overall length so that per-string length calculations work for the last string
					debug_assert_eq!(0, offsets[len]);
					offsets[len] = total_length;

					let strings = offsets
						// offsets has 1 extra offset past the end so that all windows work
						.windows(2)
						.map(|w| {
							let slice = &string_contents[w[0] as _..w[1] as _];
							String::from_utf8(slice.into())
						})
						.collect::<Result<Vec<String>, FromUtf8Error>>()
						.map_err(Error::StringFromUtf8Error)?;

					Ok((dimensions, strings))
				} else {
					Err(Error::DataTypeMismatch {
						actual: ty,
						requested: TensorElementType::String
					})
				}
			}
			t => Err(Error::NotTensor(t))
		}
	}

	/// Returns the shape of the tensor.
	///
	/// ```
	/// # use ort::{Allocator, Sequence, Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let tensor = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
	///
	/// assert_eq!(tensor.shape()?, &[1, 128, 128, 3]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn shape(&self) -> Result<Vec<i64>> {
		let mut tensor_info_ptr: *mut ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorTypeAndShape(self.ptr(), &mut tensor_info_ptr) -> Error::GetTensorTypeAndShape];

		let res = {
			let mut num_dims = 0;
			ortsys![unsafe GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> Error::GetDimensionsCount];

			let mut node_dims: Vec<i64> = vec![0; num_dims as _];
			ortsys![unsafe GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> Error::GetDimensions];

			Ok(node_dims)
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}
}

impl<T: PrimitiveTensorElementType + Debug> Tensor<T> {
	/// Extracts the underlying data into a read-only [`ndarray::ArrayView`].
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let tensor = Tensor::from_array(array.view())?;
	///
	/// let extracted = tensor.extract_tensor();
	/// assert_eq!(array.into_dyn(), extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn extract_tensor(&self) -> ndarray::ArrayViewD<'_, T> {
		self.try_extract_tensor().expect("Failed to extract tensor")
	}

	/// Extracts the underlying data into a mutable [`ndarray::ArrayViewMut`].
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let mut tensor = Tensor::from_array(array.view())?;
	///
	/// let mut extracted = tensor.extract_tensor_mut();
	/// extracted[[0, 0, 0, 1]] = 0.0;
	///
	/// let mut array = array.into_dyn();
	/// assert_ne!(array, extracted);
	/// array[[0, 0, 0, 1]] = 0.0;
	/// assert_eq!(array, extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn extract_tensor_mut(&mut self) -> ndarray::ArrayViewMutD<'_, T> {
		self.try_extract_tensor_mut().expect("Failed to extract tensor")
	}

	/// Extracts the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and an immutable
	/// view into its data.
	///
	/// ```
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let tensor = Tensor::from_array(([array.len()], array.clone().into_boxed_slice()))?;
	///
	/// let (extracted_shape, extracted_data) = tensor.extract_raw_tensor();
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_tensor(&self) -> (Vec<i64>, &[T]) {
		self.try_extract_raw_tensor().expect("Failed to extract tensor")
	}

	/// Extracts the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and a mutable view
	/// into its data.
	///
	/// ```
	/// # use ort::{Session, Tensor, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let tensor = Tensor::from_array(([array.len()], array.clone().into_boxed_slice()))?;
	///
	/// let (extracted_shape, extracted_data) = tensor.extract_raw_tensor();
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_tensor_mut(&mut self) -> (Vec<i64>, &mut [T]) {
		self.try_extract_raw_tensor_mut().expect("Failed to extract tensor")
	}
}
