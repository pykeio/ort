use alloc::{
	format,
	string::{FromUtf8Error, String},
	vec,
	vec::Vec
};
use core::{ffi::c_void, fmt::Debug, ptr, slice};

use super::{Tensor, TensorValueTypeMarker};
use crate::{
	AsPointer,
	error::{Error, ErrorCode, Result},
	memory::MemoryInfo,
	ortsys,
	tensor::{PrimitiveTensorElementType, TensorElementType},
	util::element_count,
	value::{Value, ValueType}
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
	/// # use ort::value::TensorRef;
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let value = TensorRef::from_array_view(array.view())?.into_dyn();
	///
	/// let extracted = value.try_extract_tensor::<f32>()?;
	/// assert_eq!(array.view().into_dyn(), extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the infallible
	///   [`Tensor::extract_tensor`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	/// - The tensor's data is not allocated in CPU memory.
	///
	/// [`DynValue`]: crate::value::DynValue
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_tensor<T: PrimitiveTensorElementType>(&self) -> Result<ndarray::ArrayViewD<'_, T>> {
		use ndarray::IntoDimension;
		extract_tensor(self.ptr().cast_mut(), self.dtype(), self.memory_info(), T::into_tensor_element_type()).and_then(|(ptr, dimensions)| {
			let shape = dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>().into_dimension();
			Ok(unsafe { ndarray::ArrayView::from_shape_ptr(shape, data_ptr(ptr)?.cast::<T>()) })
		})
	}

	/// Attempt to extract the scalar from a tensor of type `T`.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let value = Tensor::from_array(((), vec![3.14_f32]))?.into_dyn();
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
	/// - This is a [`DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the infallible
	///   [`Tensor::extract_tensor`] instead)*
	/// - The tensor's data is not allocated in CPU memory.
	///
	/// [`DynValue`]: crate::value::DynValue
	pub fn try_extract_scalar<T: PrimitiveTensorElementType + Copy>(&self) -> Result<T> {
		extract_tensor(self.ptr().cast_mut(), self.dtype(), self.memory_info(), T::into_tensor_element_type()).and_then(|(ptr, dimensions)| {
			if !dimensions.is_empty() {
				return Err(Error::new_with_code(
					ErrorCode::InvalidArgument,
					format!("Cannot extract scalar {} from a tensor of dimensionality {}", T::into_tensor_element_type(), dimensions.len())
				));
			}

			Ok(unsafe { *data_ptr(ptr)?.cast::<T>() })
		})
	}

	/// Attempt to extract the underlying data of type `T` into a mutable read-only [`ndarray::ArrayViewMut`].
	///
	/// See also the infallible counterpart, [`Tensor::extract_tensor_mut`], for typed [`Tensor<T>`]s.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::value::TensorRefMut;
	/// # fn main() -> ort::Result<()> {
	/// let mut array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// {
	/// 	let mut value = TensorRefMut::from_array_view_mut(array.view_mut())?.into_dyn();
	/// 	let mut extracted = value.try_extract_tensor_mut::<f32>()?;
	/// 	extracted[[0, 0, 0, 1]] = 0.0;
	/// }
	///
	/// assert_eq!(array[[0, 0, 0, 1]], 0.0);
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// # Errors
	/// May return an error if:
	/// - This is a [`DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the infallible
	///   [`Tensor::extract_tensor_mut`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	///
	/// [`DynValue`]: crate::value::DynValue
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_tensor_mut<T: PrimitiveTensorElementType>(&mut self) -> Result<ndarray::ArrayViewMutD<'_, T>> {
		use ndarray::IntoDimension;
		extract_tensor(self.ptr_mut(), self.dtype(), self.memory_info(), T::into_tensor_element_type()).and_then(|(ptr, dimensions)| {
			let shape = dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>().into_dimension();
			Ok(unsafe { ndarray::ArrayViewMut::from_shape_ptr(shape, data_ptr(ptr)?.cast::<T>()) })
		})
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
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let value = Tensor::from_array(([array.len()], array.clone().into_boxed_slice()))?.into_dyn();
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
	/// - This is a [`DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the infallible
	///   [`Tensor::extract_raw_tensor`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	///
	/// [`DynValue`]: crate::value::DynValue
	pub fn try_extract_raw_tensor<T: PrimitiveTensorElementType>(&self) -> Result<(&[i64], &[T])> {
		extract_tensor(self.ptr().cast_mut(), self.dtype(), self.memory_info(), T::into_tensor_element_type())
			.and_then(|(ptr, dimensions)| Ok((dimensions, unsafe { slice::from_raw_parts(data_ptr(ptr)?.cast::<T>(), element_count(dimensions)) })))
	}

	/// Attempt to extract the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and a
	/// mutable view into its data.
	///
	/// See also the infallible counterpart, [`Tensor::extract_raw_tensor_mut`], for typed [`Tensor<T>`]s.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let mut value = Tensor::from_array(([array.len()], array.clone().into_boxed_slice()))?.into_dyn();
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
	/// - This is a [`DynValue`], and the value is not actually a tensor. *(for typed [`Tensor`]s, use the infallible
	///   [`Tensor::extract_raw_tensor_mut`] instead)*
	/// - The provided type `T` does not match the tensor's element type.
	///
	/// [`DynValue`]: crate::value::DynValue
	pub fn try_extract_raw_tensor_mut<T: PrimitiveTensorElementType>(&mut self) -> Result<(&[i64], &mut [T])> {
		extract_tensor(self.ptr_mut(), self.dtype(), self.memory_info(), T::into_tensor_element_type())
			.and_then(|(ptr, dimensions)| Ok((dimensions, unsafe { slice::from_raw_parts_mut(data_ptr(ptr)?.cast::<T>(), element_count(dimensions)) })))
	}

	/// Attempt to extract the underlying data into a Rust `ndarray`.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array1::from_vec(vec!["hello", "world"]);
	/// let tensor = Tensor::from_string_array(&array)?.into_dyn();
	///
	/// let extracted = tensor.try_extract_string_tensor()?;
	/// assert_eq!(array.into_dyn(), extracted);
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn try_extract_string_tensor(&self) -> Result<ndarray::ArrayD<String>> {
		use ndarray::IntoDimension;
		extract_tensor(self.ptr().cast_mut(), self.dtype(), self.memory_info(), TensorElementType::String).and_then(|(ptr, dimensions)| {
			let strings = extract_strings(ptr, dimensions)?;
			Ok(ndarray::Array::from_shape_vec(dimensions.iter().map(|&n| n as usize).collect::<Vec<_>>().into_dimension(), strings)
				.expect("Shape extracted from tensor didn't match tensor contents"))
		})
	}

	/// Attempt to extract the underlying string data into a "raw" data tuple, consisting of the tensor's dimensions and
	/// an owned `Vec` of its data.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let array = vec!["hello", "world"];
	/// let tensor = Tensor::from_string_array(([array.len()], &*array))?.into_dyn();
	///
	/// let (extracted_shape, extracted_data) = tensor.try_extract_raw_string_tensor()?;
	/// assert_eq!(extracted_data, array);
	/// assert_eq!(extracted_shape, [2]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn try_extract_raw_string_tensor(&self) -> Result<(&[i64], Vec<String>)> {
		extract_tensor(self.ptr().cast_mut(), self.dtype(), self.memory_info(), TensorElementType::String).and_then(|(ptr, dimensions)| {
			let strings = extract_strings(ptr, dimensions)?;
			Ok((dimensions, strings))
		})
	}

	/// Returns the shape of the tensor.
	///
	/// ```
	/// # use ort::{memory::Allocator, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let tensor = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
	///
	/// assert_eq!(tensor.shape(), [1, 128, 128, 3]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn shape(&self) -> &[i64] {
		match self.dtype() {
			ValueType::Tensor { dimensions, .. } => dimensions,
			_ => unreachable!()
		}
	}
}

fn extract_tensor<'t>(
	ptr: *mut ort_sys::OrtValue,
	dtype: &'t ValueType,
	memory_info: &MemoryInfo,
	expected_ty: TensorElementType
) -> Result<(*mut ort_sys::OrtValue, &'t [i64])> {
	match dtype {
		ValueType::Tensor { ty, dimensions, .. } => {
			if !memory_info.is_cpu_accessible() {
				return Err(Error::new(format!(
					"Cannot extract from value on device `{}`, which is not CPU accessible",
					memory_info.allocation_device().as_str()
				)));
			}

			if *ty == expected_ty {
				Ok((ptr, dimensions))
			} else {
				Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot extract Tensor<{}> from Tensor<{}>", expected_ty, ty)))
			}
		}
		t => Err(Error::new_with_code(ErrorCode::InvalidArgument, format!("Cannot extract a Tensor<{}> from {t}", expected_ty)))
	}
}

unsafe fn data_ptr(ptr: *mut ort_sys::OrtValue) -> Result<*mut c_void> {
	let mut output_array_ptr: *mut c_void = ptr::null_mut();
	ortsys![unsafe GetTensorMutableData(ptr, &mut output_array_ptr)?; nonNull(output_array_ptr)];
	Ok(output_array_ptr)
}

fn extract_strings(ptr: *mut ort_sys::OrtValue, dimensions: &[i64]) -> Result<Vec<String>> {
	let len = element_count(dimensions);

	// Total length of string data, not including \0 suffix
	let mut total_length = 0;
	ortsys![unsafe GetStringTensorDataLength(ptr, &mut total_length)?];

	// In the JNI impl of this, tensor_element_len was included in addition to total_length,
	// but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
	// don't seem to be written to in practice either.
	// If the string data actually did go farther, it would panic below when using the offset
	// data to get slices for each string.
	let mut string_contents = vec![0u8; total_length];
	// one extra slot so that the total length can go in the last one, making all per-string
	// length calculations easy
	let mut offsets = vec![0; len + 1];

	ortsys![unsafe GetStringTensorContent(ptr, string_contents.as_mut_ptr().cast(), total_length, offsets.as_mut_ptr(), len)?];

	// final offset = overall length so that per-string length calculations work for the last string
	debug_assert_eq!(0, offsets[len]);
	offsets[len] = total_length;

	let strings = offsets
		// offsets has 1 extra offset past the end so that all windows work
		.windows(2)
		.map(|w| {
			let slice = &string_contents[w[0]..w[1]];
			String::from_utf8(slice.into())
		})
		.collect::<Result<Vec<String>, FromUtf8Error>>()
		.map_err(Error::wrap)?;
	Ok(strings)
}

impl<T: PrimitiveTensorElementType + Debug> Tensor<T> {
	/// Extracts the underlying data into a read-only [`ndarray::ArrayView`].
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::value::TensorRef;
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let tensor = TensorRef::from_array_view(&array)?;
	///
	/// let extracted = tensor.extract_tensor();
	/// assert_eq!(array.view().into_dyn(), extracted);
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
	/// # use ort::value::TensorRefMut;
	/// # fn main() -> ort::Result<()> {
	/// let mut array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// {
	/// 	let mut tensor = TensorRefMut::from_array_view_mut(array.view_mut())?;
	/// 	let mut extracted = tensor.extract_tensor_mut();
	/// 	extracted[[0, 0, 0, 1]] = 0.0;
	/// }
	///
	/// assert_eq!(array[[0, 0, 0, 1]], 0.0);
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
	/// # use ort::value::TensorRef;
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let tensor = TensorRef::from_array_view(([array.len()], &*array))?;
	///
	/// let (extracted_shape, extracted_data) = tensor.extract_raw_tensor();
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_tensor(&self) -> (&[i64], &[T]) {
		self.try_extract_raw_tensor().expect("Failed to extract tensor")
	}

	/// Extracts the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and a mutable view
	/// into its data.
	///
	/// ```
	/// # use ort::value::TensorRefMut;
	/// # fn main() -> ort::Result<()> {
	/// let mut original_array = vec![1_i64, 2, 3, 4, 5];
	/// {
	/// 	let mut tensor = TensorRefMut::from_array_view_mut(([original_array.len()], &mut *original_array))?;
	/// 	let (extracted_shape, extracted_data) = tensor.extract_raw_tensor_mut();
	/// 	extracted_data[2] = 42;
	/// }
	/// assert_eq!(original_array, [1, 2, 42, 4, 5]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_tensor_mut(&mut self) -> (&[i64], &mut [T]) {
		self.try_extract_raw_tensor_mut().expect("Failed to extract tensor")
	}
}
