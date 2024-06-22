use std::{
	any::Any,
	ffi,
	fmt::Debug,
	marker::PhantomData,
	ptr::{self, NonNull},
	sync::Arc
};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, CowArray, Dimension};

use super::{DynTensor, Tensor};
use crate::{
	error::assert_non_null_pointer,
	memory::{Allocator, MemoryInfo},
	ortsys,
	tensor::{TensorElementType, Utf8Data},
	value::{impl_tensor::calculate_tensor_size, ValueInner},
	AllocationDevice, AllocatorType, DynValue, Error, MemoryType, PrimitiveTensorElementType, Result, TensorRefMut, Value
};

impl Tensor<String> {
	/// Construct a [`DynTensor`] from an array of strings.
	///
	/// Just like numeric tensors, string tensors can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`);
	/// - (with feature `ndarray`) a mutable/exclusive reference to an [`ndarray::ArcArray`] (`&mut ArcArray<T, D>`);
	/// - (with feature `ndarray`) an owned [`ndarray::Array`];
	/// - (with feature `ndarray`) a borrowed view of another array, as an [`ndarray::ArrayView`] (`ArrayView<'_, T,
	///   D>`);
	/// - a tuple of `(dimensions, data)` where:
	///   * `dimensions` is one of `Vec<I>`, `[I]` or `&[I]`, where `I` is `i64` or `usize`;
	///   * and `data` is one of `Vec<T>`, `Box<[T]>`, `Arc<Box<[T]>>`, or `&[T]`.
	///
	/// ```
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// // Create a string tensor from a raw data vector
	/// let data = vec!["hello", "world"];
	/// let value = Value::from_string_array(([data.len()], data.into_boxed_slice()))?;
	///
	/// // Create a string tensor from an `ndarray::Array`
	/// #[cfg(feature = "ndarray")]
	/// let value = Value::from_string_array(ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap())?;
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// Note that string data will *always* be copied, no matter what form the data is provided in.
	pub fn from_string_array<T: Utf8Data>(input: impl IntoValueTensor<Item = T>) -> Result<Tensor<String>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let (shape, data) = input.ref_parts()?;
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		// create tensor without data -- data is filled in later
		ortsys![
			unsafe CreateTensorAsOrtValue(Allocator::default().ptr.as_ptr(), shape_ptr, shape_len as _, TensorElementType::String.into(), &mut value_ptr)
				-> Error::CreateTensor;
			nonNull(value_ptr)
		];

		// create null-terminated copies of each string, as per `FillStringTensor` docs
		let null_terminated_copies: Vec<ffi::CString> = data
			.iter()
			.map(|elt| {
				let slice = elt.as_utf8_bytes();
				ffi::CString::new(slice)
			})
			.collect::<Result<Vec<_>, _>>()
			.map_err(Error::FfiStringNull)?;

		let string_pointers = null_terminated_copies.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();

		ortsys![unsafe FillStringTensor(value_ptr, string_pointers.as_ptr(), string_pointers.len() as _) -> Error::FillStringTensor];

		Ok(Value {
			inner: Arc::new(ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(()),
				_memory_info: None
			}),
			_markers: PhantomData
		})
	}
}

impl<T: PrimitiveTensorElementType + Debug> Tensor<T> {
	/// Construct a tensor in a given allocator with a given shape and datatype. The data contained in the
	/// value will be zero-allocated on the allocation device.
	///
	/// This can be used to create a tensor with data on a certain device. For example, to create a tensor with pinned
	/// (CPU) memory for use with CUDA:
	/// ```no_run
	/// # use ort::{Allocator, Session, Tensor, MemoryInfo, MemoryType, AllocationDevice, AllocatorType};
	/// # fn main() -> ort::Result<()> {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDAPinned, 0, AllocatorType::Device, MemoryType::CPUInput)?
	/// )?;
	///
	/// let mut img_input = Tensor::<f32>::new(&allocator, [1, 128, 128, 3])?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(allocator: &Allocator, shape: impl ToDimensions) -> Result<Tensor<T>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let shape = shape.to_dimensions(None)?;
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		ortsys![
			unsafe CreateTensorAsOrtValue(
				allocator.ptr.as_ptr(),
				shape_ptr,
				shape_len as _,
				T::into_tensor_element_type().into(),
				&mut value_ptr
			) -> Error::CreateTensorWithData;
			nonNull(value_ptr)
		];

		Ok(Value {
			inner: Arc::new(ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(()),
				_memory_info: None
			}),
			_markers: PhantomData
		})
	}

	/// Construct a tensor from an array of data.
	///
	/// Tensors can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`);
	/// - (with feature `ndarray`) a mutable/exclusive reference to an [`ndarray::ArcArray`] (`&mut ArcArray<T, D>`);
	/// - (with feature `ndarray`) an owned [`ndarray::Array`];
	/// - (with feature `ndarray`) a borrowed view of another array, as an [`ndarray::ArrayView`] (`ArrayView<'_, T,
	///   D>`);
	/// - a tuple of `(dimensions, data)` where:
	///   * `dimensions` is one of `Vec<I>`, `[I]` or `&[I]`, where `I` is `i64` or `usize`;
	///   * and `data` is one of `Vec<T>`, `Box<[T]>`, `Arc<Box<[T]>>`, or `&[T]`.
	///
	/// ```
	/// # use ort::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let tensor = Tensor::from_array(([1usize, 2, 3], vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice()))?;
	///
	/// // Create a tensor from an `ndarray::Array`
	/// #[cfg(feature = "ndarray")]
	/// let tensor = Tensor::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// Creating string tensors requires a separate method; see [`DynTensor::from_string_array`].
	///
	/// Note that data provided in an `ndarray` may be copied in some circumstances:
	/// - `&CowArray<'_, T, D>` will always be copied regardless of whether it is uniquely owned or borrowed.
	/// - `&mut ArcArray<T, D>` and `Array<T, D>` will be copied only if the data is not in a contiguous layout (which
	///   is the case after most reshape operations)
	/// - `ArrayView<'_, T, D>` will always be copied.
	///
	/// Raw data provided as a `Arc<Box<[T]>>`, `Box<[T]>`, or `Vec<T>` will never be copied. Raw data is expected to be
	/// in standard, contigous layout.
	pub fn from_array(input: impl IntoValueTensor<Item = T>) -> Result<Tensor<T>> {
		let memory_info = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Arena, MemoryType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
		let (shape, ptr, ptr_len, guard) = input.into_parts()?;
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		let tensor_values_ptr: *mut std::ffi::c_void = ptr.cast();
		assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

		ortsys![
			unsafe CreateTensorWithDataAsOrtValue(
				memory_info.ptr.as_ptr(),
				tensor_values_ptr,
				(ptr_len * std::mem::size_of::<T>()) as _,
				shape_ptr,
				shape_len as _,
				T::into_tensor_element_type().into(),
				&mut value_ptr
			) -> Error::CreateTensorWithData;
			nonNull(value_ptr)
		];

		Ok(Value {
			inner: Arc::new(ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: guard,
				_memory_info: Some(memory_info)
			}),
			_markers: PhantomData
		})
	}
}

impl<'a, T: PrimitiveTensorElementType + Debug> TensorRefMut<'a, T> {
	/// Create a mutable tensor view from a raw pointer and shape.
	///
	/// The length of data is determined by `T` and the given shape, so the given buffer must be at least
	/// `shape.iter().product() * std::mem::size_of::<T>()` bytes.
	///
	/// This function can be used to create data from raw device memory, e.g. to directly provide data to an execution
	/// provider. For instance, to create a tensor from a raw CUDA buffer using [`cudarc`](https://docs.rs/cudarc):
	/// ```ignore
	/// let device = CudaDevice::new(0)?;
	/// let device_data = device.htod_sync_copy(&input_data)?;
	///
	/// let tensor: TensorRefMut<'_, f32> = unsafe {
	/// 	TensorRefMut::from_raw(
	/// 		MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?,
	/// 		(*device_data.device_ptr() as usize as *mut ()).cast(),
	/// 		vec![1, 3, 512, 512]
	/// 	)?
	/// };
	/// ```
	///
	/// # Safety
	/// - The pointer must be valid for the device description provided by `MemoryInfo`.
	/// - The returned tensor must outlive the data described by the data pointer.
	pub unsafe fn from_raw(info: MemoryInfo, data: *mut ort_sys::c_void, shape: Vec<i64>) -> Result<TensorRefMut<'a, T>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		let data_len = shape.iter().product::<i64>() as usize * std::mem::size_of::<T>();

		ortsys![
			unsafe CreateTensorWithDataAsOrtValue(
				info.ptr.as_ptr(),
				data,
				data_len as _,
				shape_ptr,
				shape_len as _,
				T::into_tensor_element_type().into(),
				&mut value_ptr
			) -> Error::CreateTensorWithData;
			nonNull(value_ptr)
		];

		Ok(TensorRefMut::new(Value {
			inner: Arc::new(ValueInner::CppOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				drop: true,
				_session: None
			}),
			_markers: PhantomData
		}))
	}
}

pub trait IntoValueTensor {
	type Item;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])>;
	#[allow(clippy::type_complexity)]
	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)>;
}

pub trait ToDimensions {
	fn to_dimensions(&self, expected_size: Option<usize>) -> Result<Vec<i64>>;
}

macro_rules! impl_to_dimensions {
	(@inner) => {
		fn to_dimensions(&self, expected_size: Option<usize>) -> Result<Vec<i64>> {
			let v: Vec<i64> = self
				.iter()
				.enumerate()
				.map(|(i, c)| if *c >= 1 { Ok(*c as i64) } else { Err(Error::InvalidDimension(i)) })
				.collect::<Result<_>>()?;
			let sum = calculate_tensor_size(&v);
			if let Some(expected_size) = expected_size {
				if sum != expected_size {
					Err(Error::TensorShapeMismatch {
						input: v,
						total: sum,
						expected: expected_size
					})
				} else {
					Ok(v)
				}
			} else {
				Ok(v)
			}
		}
	};
	($(for $t:ty),+) => {
		$(impl ToDimensions for $t {
			impl_to_dimensions!(@inner);
		})+
	};
	(<N> $(for $t:ty),+) => {
		$(impl<const N: usize> ToDimensions for $t {
			impl_to_dimensions!(@inner);
		})+
	};
}

impl ToDimensions for () {
	fn to_dimensions(&self, expected_size: Option<usize>) -> Result<Vec<i64>> {
		match expected_size {
			Some(1) | None => Ok(vec![]),
			Some(x) => Err(Error::TensorShapeMismatch { input: vec![], total: 1, expected: x })
		}
	}
}
impl_to_dimensions!(for &[usize], for &[i32], for &[i64], for Vec<usize>, for Vec<i32>, for Vec<i64>);
impl_to_dimensions!(<N> for [usize; N], for [i32; N], for [i64; N]);

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for &'i CowArray<'v, T, D>
where
	'i: 'v
{
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().ok_or(Error::TensorDataNotContiguous)?;
		Ok((shape, data))
	}

	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		// This will result in a copy in either form of the CowArray
		let mut contiguous_array = self.as_standard_layout().into_owned();
		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
		let ptr = contiguous_array.as_mut_ptr();
		let ptr_len = contiguous_array.len();
		let guard = Box::new(contiguous_array);
		Ok((shape, ptr, ptr_len, guard))
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for &mut ArcArray<T, D> {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().ok_or(Error::TensorDataNotContiguous)?;
		Ok((shape, data))
	}

	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		if self.is_standard_layout() {
			// We can avoid the copy here and use the data as is
			let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
			let ptr = self.as_mut_ptr();
			let ptr_len = self.len();
			let guard = Box::new(self.clone());
			Ok((shape, ptr, ptr_len, guard))
		} else {
			// Need to do a copy here to get data in to standard layout
			let mut contiguous_array = self.as_standard_layout().into_owned();
			let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
			let ptr = contiguous_array.as_mut_ptr();
			let ptr_len: usize = contiguous_array.len();
			let guard = Box::new(contiguous_array);
			Ok((shape, ptr, ptr_len, guard))
		}
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for Array<T, D> {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().ok_or(Error::TensorDataNotContiguous)?;
		Ok((shape, data))
	}

	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		if self.is_standard_layout() {
			// We can avoid the copy here and use the data as is
			let mut guard = Box::new(self);
			let shape: Vec<i64> = guard.shape().iter().map(|d| *d as i64).collect();
			let ptr = guard.as_mut_ptr();
			let ptr_len = guard.len();
			Ok((shape, ptr, ptr_len, guard))
		} else {
			// Need to do a copy here to get data in to standard layout
			let mut contiguous_array = self.as_standard_layout().into_owned();
			let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
			let ptr = contiguous_array.as_mut_ptr();
			let ptr_len: usize = contiguous_array.len();
			let guard = Box::new(contiguous_array);
			Ok((shape, ptr, ptr_len, guard))
		}
	}
}

#[cfg(feature = "ndarray")]
impl<'v, T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for ArrayView<'v, T, D> {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().ok_or(Error::TensorDataNotContiguous)?;
		Ok((shape, data))
	}

	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		// This will result in a copy in either form of the ArrayView
		let mut contiguous_array = self.as_standard_layout().into_owned();
		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
		let ptr = contiguous_array.as_mut_ptr();
		let ptr_len = contiguous_array.len();
		let guard = Box::new(contiguous_array);
		Ok((shape, ptr, ptr_len, guard))
	}
}

impl<T: Clone + Debug + 'static, D: ToDimensions> IntoValueTensor for (D, &[T]) {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		Ok((shape, self.1))
	}

	fn into_parts(self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let mut data = self.1.to_vec();
		let ptr = data.as_mut_ptr();
		let ptr_len: usize = data.len();
		Ok((shape, ptr, ptr_len, Box::new(data)))
	}
}

impl<T: Clone + Debug + 'static, D: ToDimensions> IntoValueTensor for (D, Vec<T>) {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let data = &*self.1;
		Ok((shape, data))
	}

	fn into_parts(mut self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let ptr = self.1.as_mut_ptr();
		let ptr_len: usize = self.1.len();
		Ok((shape, ptr, ptr_len, Box::new(self.1)))
	}
}

impl<T: Clone + Debug + 'static, D: ToDimensions> IntoValueTensor for (D, Box<[T]>) {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let data = &*self.1;
		Ok((shape, data))
	}

	fn into_parts(mut self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let ptr = self.1.as_mut_ptr();
		let ptr_len: usize = self.1.len();
		Ok((shape, ptr, ptr_len, Box::new(self.1)))
	}
}

impl<T: Clone + Debug + 'static, D: ToDimensions> IntoValueTensor for (D, Arc<Box<[T]>>) {
	type Item = T;

	fn ref_parts(&self) -> Result<(Vec<i64>, &[Self::Item])> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let data = &*self.1;
		Ok((shape, data))
	}

	fn into_parts(mut self) -> Result<(Vec<i64>, *mut Self::Item, usize, Box<dyn Any>)> {
		let shape = self.0.to_dimensions(Some(self.1.len()))?;
		let ptr = std::sync::Arc::<std::boxed::Box<[T]>>::make_mut(&mut self.1).as_mut_ptr();
		let ptr_len: usize = self.1.len();
		let guard = Box::new(Arc::clone(&self.1));
		Ok((shape, ptr, ptr_len, guard))
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for Tensor<T>
where
	'i: 'v
{
	type Error = Error;
	fn try_from(arr: &'i CowArray<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for Tensor<T> {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for DynTensor
where
	'i: 'v
{
	type Error = Error;
	fn try_from(arr: &'i CowArray<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr).map(|c| c.upcast())
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for DynTensor {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr).map(|c| c.upcast())
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for DynValue
where
	'i: 'v
{
	type Error = Error;
	fn try_from(arr: &'i CowArray<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr).map(|c| c.into_dyn())
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: PrimitiveTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for DynValue {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Tensor::from_array(arr).map(|c| c.into_dyn())
	}
}

macro_rules! impl_try_from {
	(@T,I $($t:ty),+) => {
		$(
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, I: ToDimensions> TryFrom<$t> for Tensor<T> {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value)
				}
			}
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, I: ToDimensions> TryFrom<$t> for DynTensor {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value).map(|c| c.upcast())
				}
			}
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, I: ToDimensions> TryFrom<$t> for crate::DynValue {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value).map(|c| c.into_dyn())
				}
			}
		)+
	};
	(@T,D $($t:ty),+) => {
		$(
			#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, D: ndarray::Dimension + 'static> TryFrom<$t> for Tensor<T> {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value)
				}
			}
			#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, D: ndarray::Dimension + 'static> TryFrom<$t> for DynTensor {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value).map(|c| c.upcast())
				}
			}
			#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
			impl<T: PrimitiveTensorElementType + Debug + Clone + 'static, D: ndarray::Dimension + 'static> TryFrom<$t> for crate::DynValue {
				type Error = Error;
				fn try_from(value: $t) -> Result<Self, Self::Error> {
					Tensor::from_array(value).map(|c| c.into_dyn())
				}
			}
		)+
	};
}

#[cfg(feature = "ndarray")]
impl_try_from!(@T,D &mut ArcArray<T, D>, Array<T, D>);
impl_try_from!(@T,I (I, Arc<Box<[T]>>), (I, Vec<T>), (I, Box<[T]>), (I, &[T]));
