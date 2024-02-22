use std::{
	any::Any,
	ffi,
	fmt::Debug,
	ptr::{self, NonNull},
	sync::Arc
};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, CowArray, Dimension};

use crate::{
	error::assert_non_null_pointer,
	memory::{Allocator, MemoryInfo},
	ortsys,
	tensor::{IntoTensorElementType, TensorElementType, Utf8Data},
	value::ValueInner,
	AllocatorType, Error, MemoryType, Result, Value
};

impl Value {
	/// Construct a tensor [`Value`] from an array of data.
	///
	/// Tensor `Value`s can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`);
	/// - (with feature `ndarray`) a mutable/exclusive reference to an [`ndarray::ArcArray`] (`&mut ArcArray<T, D>`);
	/// - (with feature `ndarray`) an owned [`ndarray::Array`];
	/// - (with feature `ndarray`) a borrowed view of another array, as an [`ndarray::ArrayView`] (`ArrayView<'_, T,
	///   D>`);
	/// - a tuple of `(dimensions, data)` where `dimensions` is a `Vec<i64>` and `data` is an `Arc<Box<[T]>>` (referred
	///   to as "raw data").
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::Value;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let value =
	/// 	Value::from_array((vec![1, 2, 3], Arc::new(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0].into_boxed_slice())))?;
	///
	/// // Create a tensor from an `ndarray::Array`
	/// #[cfg(feature = "ndarray")]
	/// let value = Value::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// Creating string tensors requires a separate method; see [`Value::from_string_array`].
	///
	/// Note that data provided in an `ndarray` may be copied in some circumstances:
	/// - `&CowArray<'_, T, D>` will always be copied regardless of whether it is uniquely owned or borrowed.
	/// - `&mut ArcArray<T, D>` and `Array<T, D>` will be copied only if the data is not in a contiguous layout (which
	///   is the case after most reshape operations)
	/// - `ArrayView<'_, T, D>` will always be copied.
	///
	/// Raw data will never be copied. The data is expected to be in standard, contigous layout.
	pub fn from_array<T: IntoTensorElementType + Debug + Clone + 'static>(input: impl IntoValueTensor<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemoryType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let guard = match T::into_tensor_element_type() {
			TensorElementType::Float32
			| TensorElementType::Uint8
			| TensorElementType::Int8
			| TensorElementType::Uint16
			| TensorElementType::Int16
			| TensorElementType::Int32
			| TensorElementType::Int64
			| TensorElementType::Float64
			| TensorElementType::Uint32
			| TensorElementType::Uint64
			| TensorElementType::Bool => {
				// primitive data is already suitably laid out in memory; provide it to
				// onnxruntime as is
				let (shape, ptr, ptr_len, guard) = input.into_parts();
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

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
				guard
			}
			#[cfg(feature = "half")]
			TensorElementType::Bfloat16 | TensorElementType::Float16 => {
				// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
				let (shape, ptr, ptr_len, guard) = input.into_parts();
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

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
				guard
			}
			TensorElementType::String => unreachable!()
		};

		assert_non_null_pointer(value_ptr, "Value")?;

		Ok(Value {
			inner: ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: guard,
				_memory_info: memory_info
			}
		})
	}

	/// Construct a [`Value`] from an array of strings.
	///
	/// Just like numeric tensors, string tensor `Value`s can be created from:
	/// - (with feature `ndarray`) a shared reference to a [`ndarray::CowArray`] (`&CowArray<'_, T, D>`);
	/// - (with feature `ndarray`) a mutable/exclusive reference to an [`ndarray::ArcArray`] (`&mut ArcArray<T, D>`);
	/// - (with feature `ndarray`) an owned [`ndarray::Array`];
	/// - (with feature `ndarray`) a borrowed view of another array, as an [`ndarray::ArrayView`] (`ArrayView<'_, T,
	///   D>`);
	/// - a tuple of `(dimensions, data)` where `dimensions` is a `Vec<i64>` and `data` is an `Arc<Box<[T]>>` (referred
	///   to as "raw data").
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.with_model_from_file("tests/data/vectorizer.onnx")?;
	/// // You'll need to obtain an `Allocator` from a session in order to create string tensors.
	/// let allocator = session.allocator();
	///
	/// // Create a string tensor from a raw data vector
	/// let value = Value::from_string_array(allocator, (vec![2], Arc::new(vec!["hello", "world"].into_boxed_slice())))?;
	///
	/// // Create a string tensor from an `ndarray::Array`
	/// #[cfg(feature = "ndarray")]
	/// let value = Value::from_string_array(
	/// 	allocator,
	/// 	ndarray::Array::from_shape_vec((1,), vec!["document".to_owned()]).unwrap()
	/// )?;
	/// # 	Ok(())
	/// # }
	/// ```
	///
	/// Note that string data will always be copied, no matter what data is provided.
	pub fn from_string_array<T: Utf8Data + Debug + Clone + 'static>(allocator: &Allocator, input: impl IntoValueTensor<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemoryType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let (shape, data) = input.ref_parts();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		// create tensor without data -- data is filled in later
		ortsys![
			unsafe CreateTensorAsOrtValue(allocator.ptr.as_ptr(), shape_ptr, shape_len as _, TensorElementType::String.into(), &mut value_ptr)
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

		assert_non_null_pointer(value_ptr, "Value")?;

		Ok(Value {
			inner: ValueInner::RustOwned {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				_array: Box::new(()),
				_memory_info: memory_info
			}
		})
	}
}

pub trait IntoValueTensor {
	type Item;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]);
	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>);
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for &'i CowArray<'v, T, D>
where
	'i: 'v
{
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().expect("tensor should be contiguous");
		(shape, data)
	}

	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		// This will result in a copy in either form of the CowArray
		let mut contiguous_array = self.as_standard_layout().into_owned();
		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
		let ptr = contiguous_array.as_mut_ptr();
		let ptr_len = contiguous_array.len();
		let guard = Box::new(contiguous_array);
		(shape, ptr, ptr_len, guard)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for &mut ArcArray<T, D> {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().expect("tensor should be contiguous");
		(shape, data)
	}

	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		if self.is_standard_layout() {
			// We can avoid the copy here and use the data as is
			let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
			let ptr = self.as_mut_ptr();
			let ptr_len = self.len();
			let guard = Box::new(self.clone());
			(shape, ptr, ptr_len, guard)
		} else {
			// Need to do a copy here to get data in to standard layout
			let mut contiguous_array = self.as_standard_layout().into_owned();
			let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
			let ptr = contiguous_array.as_mut_ptr();
			let ptr_len: usize = contiguous_array.len();
			let guard = Box::new(contiguous_array);
			(shape, ptr, ptr_len, guard)
		}
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for Array<T, D> {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().expect("tensor should be contiguous");
		(shape, data)
	}

	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		if self.is_standard_layout() {
			// We can avoid the copy here and use the data as is
			let mut guard = Box::new(self);
			let shape: Vec<i64> = guard.shape().iter().map(|d| *d as i64).collect();
			let ptr = guard.as_mut_ptr();
			let ptr_len = guard.len();
			(shape, ptr, ptr_len, guard)
		} else {
			// Need to do a copy here to get data in to standard layout
			let mut contiguous_array = self.as_standard_layout().into_owned();
			let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
			let ptr = contiguous_array.as_mut_ptr();
			let ptr_len: usize = contiguous_array.len();
			let guard = Box::new(contiguous_array);
			(shape, ptr, ptr_len, guard)
		}
	}
}

#[cfg(feature = "ndarray")]
impl<'v, T: Clone + 'static, D: Dimension + 'static> IntoValueTensor for ArrayView<'v, T, D> {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape: Vec<i64> = self.shape().iter().map(|d| *d as i64).collect();
		let data = self.as_slice().expect("tensor should be contiguous");
		(shape, data)
	}

	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		// This will result in a copy in either form of the ArrayView
		let mut contiguous_array = self.as_standard_layout().into_owned();
		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
		let ptr = contiguous_array.as_mut_ptr();
		let ptr_len = contiguous_array.len();
		let guard = Box::new(contiguous_array);
		(shape, ptr, ptr_len, guard)
	}
}

impl<T: Clone + Debug + 'static> IntoValueTensor for (Vec<i64>, &[T]) {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape = self.0.clone();
		(shape, self.1)
	}

	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		let shape = self.0.clone();
		let mut data = self.1.to_vec();
		let ptr = data.as_mut_ptr();
		let ptr_len: usize = data.len();
		(shape, ptr, ptr_len, Box::new(data))
	}
}

impl<T: Clone + Debug + 'static> IntoValueTensor for (Vec<i64>, Vec<T>) {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape = self.0.clone();
		let data = &*self.1;
		(shape, data)
	}

	fn into_parts(mut self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		let shape = self.0.clone();
		let ptr = self.1.as_mut_ptr();
		let ptr_len: usize = self.1.len();
		(shape, ptr, ptr_len, Box::new(self.1))
	}
}

impl<T: Clone + Debug + 'static> IntoValueTensor for (Vec<i64>, Box<[T]>) {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape = self.0.clone();
		let data = &*self.1;
		(shape, data)
	}

	fn into_parts(mut self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		let shape = self.0.clone();
		let ptr = self.1.as_mut_ptr();
		let ptr_len: usize = self.1.len();
		(shape, ptr, ptr_len, Box::new(self.1))
	}
}

impl<T: Clone + Debug + 'static> IntoValueTensor for (Vec<i64>, Arc<Box<[T]>>) {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape = self.0.clone();
		let data = &*self.1;
		(shape, data)
	}

	fn into_parts(mut self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>) {
		let shape = self.0.clone();
		let ptr = std::sync::Arc::<std::boxed::Box<[T]>>::make_mut(&mut self.1).as_mut_ptr();
		let ptr_len: usize = self.1.len();
		let guard = Box::new(Arc::clone(&self.1));
		(shape, ptr, ptr_len, guard)
	}
}
