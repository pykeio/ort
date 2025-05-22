use alloc::{boxed::Box, ffi::CString, format, string::String, sync::Arc, vec::Vec};
use core::{
	any::Any,
	ffi::c_void,
	fmt::Debug,
	marker::PhantomData,
	mem::size_of,
	ptr::{self, NonNull}
};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, ArrayViewMut, CowArray, Dimension};

use super::{DynTensor, Tensor, TensorRef, TensorRefMut};
use crate::{
	AsPointer,
	error::{Error, ErrorCode, Result},
	memory::{Allocator, MemoryInfo},
	ortsys,
	tensor::{PrimitiveTensorElementType, Shape, SymbolicDimensions, TensorElementType, Utf8Data},
	value::{Value, ValueInner, ValueType}
};

impl Tensor<String> {
	/// Construct a [`Tensor`] from an array of strings.
	///
	/// String tensors can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`) or
	///   [`ndarray::Array`] (`&Array<T, D>`);
	/// - (with feature `ndarray`) an [`ndarray::ArcArray`] or [`ndarray::ArrayView`];
	/// - a tuple of `(shape, data)` where:
	///   * `shape` is one of `Vec<I>`, `[I; N]` or `&[I]`, where `I` is `i64` or `usize`, and
	///   * `data` is one of `&[T]`, `Arc<[T]>`, or `Arc<Box<[T]>>`.
	///
	/// ```
	/// # use ort::{session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// // Create a string tensor from a raw data vector
	/// let data = vec!["hello", "world"];
	/// let value = Tensor::from_string_array(([data.len()], &*data))?;
	///
	/// // Create a string tensor from an `ndarray::Array`
	/// #[cfg(feature = "ndarray")]
	/// let value = Tensor::from_string_array(&ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap())?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn from_string_array<T: Utf8Data>(input: impl TensorArrayData<T>) -> Result<Tensor<String>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let (shape, data, _guard) = input.ref_parts()?;
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		// create tensor without data -- data is filled in later
		ortsys![
			unsafe CreateTensorAsOrtValue(Allocator::default().ptr_mut(), shape_ptr, shape_len, TensorElementType::String.into(), &mut value_ptr)?;
			nonNull(value_ptr)
		];

		// create null-terminated copies of each string, as per `FillStringTensor` docs
		let null_terminated_copies: Vec<CString> = data
			.iter()
			.map(|elt| {
				let slice = elt.as_utf8_bytes();
				CString::new(slice)
			})
			.collect::<Result<Vec<_>, _>>()?;

		let string_pointers = null_terminated_copies.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();

		ortsys![unsafe FillStringTensor(value_ptr.as_ptr(), string_pointers.as_ptr(), string_pointers.len())?];

		Ok(Value {
			inner: Arc::new(ValueInner {
				ptr: value_ptr,
				dtype: ValueType::Tensor {
					ty: TensorElementType::String,
					shape,
					dimension_symbols: SymbolicDimensions::empty(shape_len)
				},
				memory_info: unsafe { MemoryInfo::from_value(value_ptr) },
				drop: true,
				_backing: None
			}),
			_markers: PhantomData
		})
	}
}

impl<T: PrimitiveTensorElementType + Debug> Tensor<T> {
	/// Construct a tensor via a given allocator with a given shape and datatype. The data in the tensor will be
	/// **uninitialized**.
	///
	/// This can be used to create a tensor with data on a certain device. For example, to create a tensor with pinned
	/// (CPU) memory for use with CUDA:
	/// ```no_run
	/// # use ort::{memory::{Allocator, MemoryInfo, MemoryType, AllocationDevice, AllocatorType}, session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
	/// )?;
	///
	/// let mut img_input = Tensor::<f32>::new(&allocator, [1_usize, 128, 128, 3])?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(allocator: &Allocator, shape: impl Into<Shape>) -> Result<Tensor<T>> {
		let tensor = DynTensor::new(allocator, T::into_tensor_element_type(), shape)?;
		Ok(unsafe { tensor.transmute_type() })
	}

	/// Construct an owned tensor from an array of data.
	///
	/// Owned tensors can be created from:
	/// - (with feature `ndarray`) an owned [`ndarray::Array`], or
	/// - a tuple of `(shape, data)` where:
	///   * `shape` is one of `Vec<I>`, `[I]` or `&[I]`, where `I` is `i64` or `usize`, and
	///   * `data` is one of `Vec<T>` or `Box<[T]>`.
	///
	/// ```
	/// # use ort::value::Tensor;
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
	/// When passing an [`ndarray::Array`], the array may be copied in order to convert it to a contiguous layout if it
	/// is not already. When creating a tensor from a `Vec` or boxed slice, the data is assumed to already be in
	/// contiguous layout.
	///
	/// Creating string tensors requires a separate method; see [`Tensor::from_string_array`].
	pub fn from_array(input: impl OwnedTensorArrayData<T>) -> Result<Tensor<T>> {
		let TensorArrayDataParts { shape, ptr, guard } = input.into_parts()?;
		tensor_from_array(MemoryInfo::default(), shape, ptr.as_ptr().cast(), size_of::<T>(), T::into_tensor_element_type(), guard)
			.map(|tensor| unsafe { tensor.transmute_type() })
	}
}

fn tensor_from_array(
	memory_info: MemoryInfo,
	shape: Shape,
	data: *mut c_void,
	element_size: usize,
	element_type: TensorElementType,
	guard: Option<Box<dyn Any>>
) -> Result<DynTensor> {
	let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

	ortsys![
		unsafe CreateTensorWithDataAsOrtValue(
			memory_info.ptr(),
			data,
			shape.num_elements() * element_size,
			shape.as_ptr(),
			shape.len(),
			element_type.into(),
			&mut value_ptr
		)?;
		nonNull(value_ptr)
	];

	Ok(DynTensor {
		inner: Arc::new(ValueInner {
			ptr: value_ptr,
			dtype: ValueType::Tensor {
				ty: element_type,
				dimension_symbols: SymbolicDimensions::empty(shape.len()),
				shape
			},
			drop: true,
			memory_info: Some(memory_info),
			_backing: guard
		}),
		_markers: PhantomData
	})
}

impl<'a, T: PrimitiveTensorElementType + Debug> TensorRef<'a, T> {
	/// Construct a tensor from borrowed data.
	///
	/// Borrowed tensors can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`) or
	///   [`ndarray::Array`] (`&Array<T, D>`);
	/// - (with feature `ndarray`) an [`ndarray::ArcArray`] or [`ndarray::ArrayView`];
	/// - a tuple of `(shape, data)` where:
	///   * `shape` is one of `Vec<I>`, `[I; N]` or `&[I]`, where `I` is `i64` or `usize`, and
	///   * `data` is one of `&[T]`, `Arc<[T]>`, or `Arc<Box<[T]>>`.
	///
	/// ```
	/// # use ort::value::TensorRef;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
	/// let tensor = TensorRef::from_array_view(([1usize, 2, 3], &*data))?;
	///
	/// // Create a tensor from an `ndarray::Array`
	/// # #[cfg(feature = "ndarray")]
	/// # {
	/// let array = ndarray::Array4::<f32>::zeros((1, 16, 16, 3));
	/// let tensor = TensorRef::from_array_view(array.view())?;
	/// # }
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// When passing an [`ndarray`] type, the data **must** have a contiguous memory layout, or else an error will be
	/// returned. See [`ndarray::ArrayBase::as_standard_layout`] to convert an array to a contiguous layout.
	pub fn from_array_view(input: impl TensorArrayData<T> + 'a) -> Result<TensorRef<'a, T>> {
		let (shape, data, guard) = input.ref_parts()?;
		tensor_from_array(MemoryInfo::default(), shape, data.as_ptr() as *mut _, size_of::<T>(), T::into_tensor_element_type(), guard).map(|tensor| {
			let mut tensor: TensorRef<'_, T> = TensorRef::new(unsafe { tensor.transmute_type() });
			tensor.upgradable = false;
			tensor
		})
	}
}

impl<'a, T: PrimitiveTensorElementType + Debug> TensorRefMut<'a, T> {
	/// Construct a mutable tensor view from borrowed data. Modifying data through this view will modify the
	/// underlying buffer as well.
	///
	/// Mutably borrowed tensors can be created from:
	/// - (with feature `ndarray`) an exclusive reference to an [`ndarray::Array`] (`&mut Array<T, D>`);
	/// - (with feature `ndarray`) an [`ndarray::ArrayViewMut`];
	/// - a tuple of `(shape, &mut [T])`, where `shape` is one of `Vec<I>`, `[I; N]` or `&[I]`, where `I` is `i64` or
	///   `usize`.
	///
	/// ```
	/// # use ort::value::TensorRefMut;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let mut data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
	/// let tensor = TensorRefMut::from_array_view_mut(([1usize, 2, 3], &mut *data))?;
	///
	/// // Create a tensor from an `ndarray::Array`
	/// # #[cfg(feature = "ndarray")]
	/// # {
	/// let mut array = ndarray::Array4::<f32>::zeros((1, 16, 16, 3));
	/// let tensor = TensorRefMut::from_array_view_mut(array.view_mut())?;
	/// # }
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// When passing an [`ndarray`] type, the data **must** have a contiguous memory layout, or else an error will be
	/// returned. See [`ndarray::ArrayBase::as_standard_layout`] to convert an array to a contiguous layout.
	pub fn from_array_view_mut(mut input: impl TensorArrayDataMut<T>) -> Result<TensorRefMut<'a, T>> {
		let (shape, data, guard) = input.ref_parts_mut()?;
		tensor_from_array(MemoryInfo::default(), shape, data.as_ptr() as *mut _, size_of::<T>(), T::into_tensor_element_type(), guard).map(|tensor| {
			let mut tensor: TensorRefMut<'_, T> = TensorRefMut::new(unsafe { tensor.transmute_type() });
			tensor.upgradable = false;
			tensor
		})
	}

	/// Create a mutable tensor view from a raw pointer and shape.
	///
	/// The length of data is determined by `T` and the given shape, so the given buffer must be at least
	/// `shape.num_elements() * size_of::<T>()` bytes.
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
	/// 		Shape::new([1, 3, 512, 512])
	/// 	)?
	/// };
	/// ```
	///
	/// # Safety
	/// - The pointer must be valid for the device description provided by `MemoryInfo`.
	/// - The returned tensor must outlive the data described by the data pointer.
	pub unsafe fn from_raw(info: MemoryInfo, data: *mut ort_sys::c_void, shape: Shape) -> Result<TensorRefMut<'a, T>> {
		tensor_from_array(info, shape, data, size_of::<T>(), T::into_tensor_element_type(), None).map(|tensor| {
			let mut tensor: TensorRefMut<'_, T> = TensorRefMut::new(unsafe { tensor.transmute_type() });
			tensor.upgradable = false;
			tensor
		})
	}
}

pub trait TensorArrayData<I> {
	#[allow(clippy::type_complexity)]
	fn ref_parts(&self) -> Result<(Shape, &[I], Option<Box<dyn Any>>)>;

	private_trait!();
}

pub trait TensorArrayDataMut<I>: TensorArrayData<I> {
	#[allow(clippy::type_complexity)]
	fn ref_parts_mut(&mut self) -> Result<(Shape, &mut [I], Option<Box<dyn Any>>)>;

	private_trait!();
}

pub trait OwnedTensorArrayData<I> {
	fn into_parts(self) -> Result<TensorArrayDataParts<I>>;

	private_trait!();
}

pub struct TensorArrayDataParts<I> {
	pub shape: Shape,
	pub ptr: NonNull<I>,
	pub guard: Option<Box<dyn Any>>
}

pub trait ToShape {
	fn to_shape(&self, expected_size: Option<usize>) -> Result<Shape>;
}

macro_rules! impl_to_shape {
	(@inner) => {
		fn to_shape(&self, expected_size: Option<usize>) -> Result<Shape> {
			let v = self
				.iter()
				.enumerate()
				.map(|(i, c)| {
					if *c >= 1 {
						Ok(*c as i64)
					} else {
						Err(Error::new_with_code(
							ErrorCode::InvalidArgument,
							format!("Invalid dimension #{}; all dimensions must be >= 1 when creating a tensor from raw data", i + 1)
						))
					}
				})
				.collect::<Result<Shape>>()?;
			if let Some(expected_size) = expected_size {
				if v.num_elements() != expected_size {
					Err(Error::new_with_code(
						ErrorCode::InvalidArgument,
						format!(
							"Cannot create a tensor from raw data; shape {:?} ({} elements) is larger than the length of the data provided ({} elements)",
							v,
							v.num_elements(),
							expected_size
						)
					))
				} else {
					Ok(v)
				}
			} else {
				Ok(v)
			}
		}
	};
	($(for $t:ty),+) => {
		$(impl ToShape for $t {
			impl_to_shape!(@inner);
		})+
	};
	(<N> $(for $t:ty),+) => {
		$(impl<const N: usize> ToShape for $t {
			impl_to_shape!(@inner);
		})+
	};
}

impl ToShape for () {
	fn to_shape(&self, expected_size: Option<usize>) -> Result<Shape> {
		match expected_size {
			Some(1) | None => Ok(Shape::default()),
			Some(_) => Err(Error::new_with_code(ErrorCode::InvalidArgument, "Expected data to have a length of exactly 1 for scalar shape"))
		}
	}
}

impl_to_shape!(for Shape, for &[usize], for &[i32], for &[i64], for Vec<usize>, for Vec<i32>, for Vec<i64>);
impl_to_shape!(<N> for [usize; N], for [i32; N], for [i64; N]);

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for &CowArray<'_, T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for ArcArray<T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, Some(Box::new(self.clone()))))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for &Array<T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for &mut Array<T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> OwnedTensorArrayData<T> for Array<T, D> {
	fn into_parts(self) -> Result<TensorArrayDataParts<T>> {
		if self.is_standard_layout() {
			// We can avoid the copy here and use the data as is
			let mut this = Box::new(self);
			let shape: Shape = this.shape().iter().map(|d| *d as i64).collect();
			// SAFETY: ndarrays internally store their pointer as NonNull
			let ptr = unsafe { NonNull::new_unchecked(this.as_mut_ptr()) };
			assert_eq!(this.len(), shape.num_elements());
			Ok(TensorArrayDataParts { shape, ptr, guard: Some(this) })
		} else {
			// Need to do a copy here to get data in to standard layout
			let mut contiguous_array = self.as_standard_layout().into_owned();
			let shape: Shape = contiguous_array.shape().iter().map(|d| *d as i64).collect();
			// SAFETY: ndarrays internally store their pointer as NonNull
			let ptr = unsafe { NonNull::new_unchecked(contiguous_array.as_mut_ptr()) };
			assert_eq!(contiguous_array.len(), shape.num_elements());
			Ok(TensorArrayDataParts {
				shape,
				ptr,
				guard: Some(Box::new(contiguous_array))
			})
		}
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for ArrayView<'_, T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayData<T> for ArrayViewMut<'_, T, D> {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayDataMut<T> for ArrayViewMut<'_, T, D> {
	fn ref_parts_mut(&mut self) -> Result<(Shape, &mut [T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice_mut()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> TensorArrayDataMut<T> for &mut Array<T, D> {
	fn ref_parts_mut(&mut self) -> Result<(Shape, &mut [T], Option<Box<dyn Any>>)> {
		let shape = self.shape().iter().map(|d| *d as i64).collect();
		let data = self
			.as_slice_mut()
			.ok_or_else(|| Error::new("Array has a non-contiguous layout and cannot be used to construct a Tensor"))?;
		Ok((shape, data, None))
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> TensorArrayData<T> for (D, &[T]) {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		Ok((shape, self.1, None))
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> TensorArrayData<T> for (D, &mut [T]) {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		Ok((shape, self.1, None))
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> TensorArrayDataMut<T> for (D, &mut [T]) {
	fn ref_parts_mut(&mut self) -> Result<(Shape, &mut [T], Option<Box<dyn Any>>)> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		Ok((shape, self.1, None))
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> OwnedTensorArrayData<T> for (D, Vec<T>) {
	fn into_parts(mut self) -> Result<TensorArrayDataParts<T>> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		// SAFETY: A `Vec` always has a non-null pointer.
		let ptr = unsafe { NonNull::new_unchecked(self.1.as_mut_ptr()) };
		assert_eq!(shape.num_elements(), self.1.len());
		Ok(TensorArrayDataParts {
			shape,
			ptr,
			guard: Some(Box::new(self.1))
		})
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> OwnedTensorArrayData<T> for (D, Box<[T]>) {
	fn into_parts(mut self) -> Result<TensorArrayDataParts<T>> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		// SAFETY: A `Box` always has a non-null pointer.
		let ptr = unsafe { NonNull::new_unchecked(self.1.as_mut_ptr()) };
		assert_eq!(shape.num_elements(), self.1.len());
		Ok(TensorArrayDataParts {
			shape,
			ptr,
			guard: Some(Box::new(self.1))
		})
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> TensorArrayData<T> for (D, Arc<[T]>) {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		let data = &*self.1;
		Ok((shape, data, Some(Box::new(self.1.clone()))))
	}

	private_impl!();
}

impl<T: Clone + 'static, D: ToShape> TensorArrayData<T> for (D, Arc<Box<[T]>>) {
	fn ref_parts(&self) -> Result<(Shape, &[T], Option<Box<dyn Any>>)> {
		let shape = self.0.to_shape(Some(self.1.len()))?;
		let data = &*self.1;
		Ok((shape, data, Some(Box::new(self.1.clone()))))
	}

	private_impl!();
}
