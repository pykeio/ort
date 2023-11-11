use std::{any::Any, ffi, fmt::Debug, ops::Deref, ptr, sync::Arc};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, CowArray, Dimension, IxDyn};

#[cfg(feature = "ndarray")]
use crate::tensor::Tensor;
use crate::{
	error::assert_non_null_pointer,
	memory::{Allocator, MemoryInfo},
	ortsys,
	session::SharedSessionInner,
	tensor::{ExtractTensorData, IntoTensorElementDataType, TensorElementDataType, Utf8Data},
	AllocatorType, Error, MemType, Result
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ValueType {
	Tensor { ty: TensorElementDataType, dimensions: Vec<i64> },
	Sequence(Box<ValueType>),
	Map { key: TensorElementDataType, value: TensorElementDataType }
}

impl ValueType {
	/// Returns the dimensions of this data type if it is a tensor, or `None` if it is a sequence or map.
	pub fn tensor_dimensions(&self) -> Option<&Vec<i64>> {
		match self {
			ValueType::Tensor { dimensions, .. } => Some(dimensions),
			_ => None
		}
	}
}

#[doc(hidden)]
#[derive(Debug)]
#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "fetch-models")))]
pub enum DynArrayRef<'v> {
	Float(CowArray<'v, f32, IxDyn>),
	#[cfg(feature = "half")]
	#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
	Float16(CowArray<'v, half::f16, IxDyn>),
	#[cfg(feature = "half")]
	#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
	Bfloat16(CowArray<'v, half::bf16, IxDyn>),
	Uint8(CowArray<'v, u8, IxDyn>),
	Int8(CowArray<'v, i8, IxDyn>),
	Uint16(CowArray<'v, u16, IxDyn>),
	Int16(CowArray<'v, i16, IxDyn>),
	Int32(CowArray<'v, i32, IxDyn>),
	Int64(CowArray<'v, i64, IxDyn>),
	Bool(CowArray<'v, bool, IxDyn>),
	Double(CowArray<'v, f64, IxDyn>),
	Uint32(CowArray<'v, u32, IxDyn>),
	Uint64(CowArray<'v, u64, IxDyn>),
	String(CowArray<'v, String, IxDyn>)
}

macro_rules! impl_convert_trait {
	($type_:ty, $variant:expr) => {
		#[cfg(feature = "ndarray")]
		#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
		impl<'v, D: Dimension> From<ArrayView<'v, $type_, D>> for DynArrayRef<'v> {
			fn from(array: ArrayView<'v, $type_, D>) -> DynArrayRef<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		#[cfg(feature = "ndarray")]
		#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
		impl<'v, D: Dimension> From<Array<$type_, D>> for DynArrayRef<'v> {
			fn from(array: Array<$type_, D>) -> DynArrayRef<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		#[cfg(feature = "ndarray")]
		#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
		impl<'v, D: Dimension> From<CowArray<'v, $type_, D>> for DynArrayRef<'v> {
			fn from(array: CowArray<'v, $type_, D>) -> DynArrayRef<'v> {
				$variant(array.into_dyn())
			}
		}
	};
}

impl_convert_trait!(f32, DynArrayRef::Float);
#[cfg(feature = "half")]
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
impl_convert_trait!(half::f16, DynArrayRef::Float16);
#[cfg(feature = "half")]
#[cfg_attr(docsrs, doc(cfg(feature = "half")))]
impl_convert_trait!(half::bf16, DynArrayRef::Bfloat16);
impl_convert_trait!(u8, DynArrayRef::Uint8);
impl_convert_trait!(i8, DynArrayRef::Int8);
impl_convert_trait!(u16, DynArrayRef::Uint16);
impl_convert_trait!(i16, DynArrayRef::Int16);
impl_convert_trait!(i32, DynArrayRef::Int32);
impl_convert_trait!(i64, DynArrayRef::Int64);
impl_convert_trait!(f64, DynArrayRef::Double);
impl_convert_trait!(u32, DynArrayRef::Uint32);
impl_convert_trait!(u64, DynArrayRef::Uint64);
impl_convert_trait!(bool, DynArrayRef::Bool);
impl_convert_trait!(String, DynArrayRef::String);

#[derive(Debug)]
pub(crate) enum ValueInner {
	RustOwned {
		ptr: *mut ort_sys::OrtValue,
		_array: Box<dyn Any>,
		_memory_info: MemoryInfo
	},
	CppOwned {
		ptr: *mut ort_sys::OrtValue,
		/// Hold [`SharedSessionInner`] to ensure that the value can stay alive after the main session is dropped.
		_session: Arc<SharedSessionInner>
	}
}

/// A [`Value`] contains data for inputs/outputs in ONNX Runtime graphs. [`Value`]s can hold a tensor, sequence (array),
/// or map.
#[derive(Debug)]
pub struct Value {
	inner: ValueInner
}

unsafe impl Send for Value {}

impl Value {
	/// Construct a [`Value`] from a C++ [`ort_sys::OrtValue`] pointer.
	///
	/// # Safety
	///
	/// - `ptr` must not be null.
	pub unsafe fn from_raw(ptr: *mut ort_sys::OrtValue, session: Arc<SharedSessionInner>) -> Value {
		Value {
			inner: ValueInner::CppOwned { ptr, _session: session }
		}
	}

	/// Attempt to extract the underlying data into a Rust `ndarray`.
	///
	/// The resulting array will be wrapped within a [`Tensor`].
	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn extract_tensor<T>(&self) -> Result<Tensor<'_, T>>
	where
		T: ExtractTensorData + Clone + Debug
	{
		let mut tensor_info_ptr: *mut ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorTypeAndShape(self.ptr(), &mut tensor_info_ptr) -> Error::GetTensorTypeAndShape];

		let res = {
			let mut num_dims = 0;
			ortsys![unsafe GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> Error::GetDimensionsCount];

			let mut node_dims: Vec<i64> = vec![0; num_dims as _];
			ortsys![unsafe GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> Error::GetDimensions];
			let shape = IxDyn(&node_dims.iter().map(|&n| n as usize).collect::<Vec<_>>());

			let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
			ortsys![unsafe GetTensorElementType(tensor_info_ptr, &mut type_sys) -> Error::GetTensorElementType];
			assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
			let data_type: TensorElementDataType = type_sys.into();
			if data_type != T::tensor_element_data_type() {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_data_type()
				})
			} else {
				// Note: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the ArrayView.
				assert_ne!(self.ptr(), ptr::null_mut());

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(self.ptr(), &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let mut len = 0;
				ortsys![unsafe GetTensorShapeElementCount(tensor_info_ptr, &mut len) -> Error::GetTensorShapeElementCount];

				let data = T::extract_tensor_array(shape, len as _, self.ptr())?;
				Ok(Tensor { data })
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}

	pub fn extract_raw_tensor<T>(&self) -> Result<(Vec<i64>, &[T])>
	where
		T: ExtractTensorData + Clone + Debug
	{
		let mut tensor_info_ptr: *mut ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorTypeAndShape(self.ptr(), &mut tensor_info_ptr) -> Error::GetTensorTypeAndShape];

		let res = {
			let mut num_dims = 0;
			ortsys![unsafe GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> Error::GetDimensionsCount];

			let mut node_dims: Vec<i64> = vec![0; num_dims as _];
			ortsys![unsafe GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> Error::GetDimensions];

			let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
			ortsys![unsafe GetTensorElementType(tensor_info_ptr, &mut type_sys) -> Error::GetTensorElementType];
			assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
			let data_type: TensorElementDataType = type_sys.into();
			if data_type != T::tensor_element_data_type() {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_data_type()
				})
			} else {
				// Note: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the slice.
				assert_ne!(self.ptr(), ptr::null_mut());

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(self.ptr(), &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let mut output_array_ptr: *mut T = ptr::null_mut();
				let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
				let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr as *mut *mut std::ffi::c_void;
				ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

				let mut len = 0;
				ortsys![unsafe GetTensorShapeElementCount(tensor_info_ptr, &mut len) -> Error::GetTensorShapeElementCount];

				Ok((node_dims, unsafe { std::slice::from_raw_parts(output_array_ptr, len) }))
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}
}

pub trait OrtInput {
	type Item;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]);
	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>);
}

impl Value {
	/// Construct a [`Value`] from a Rust-owned array.
	///
	/// `allocator` is required to be `Some` when converting a String tensor. See [`crate::Session::allocator`].
	pub fn from_array<T: IntoTensorElementDataType + Debug + Clone + 'static>(input: impl OrtInput<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		let value_ptr_ptr: *mut *mut ort_sys::OrtValue = &mut value_ptr;

		let guard = match T::into_tensor_element_data_type() {
			TensorElementDataType::Float32
			| TensorElementDataType::Uint8
			| TensorElementDataType::Int8
			| TensorElementDataType::Uint16
			| TensorElementDataType::Int16
			| TensorElementDataType::Int32
			| TensorElementDataType::Int64
			| TensorElementDataType::Float64
			| TensorElementDataType::Uint32
			| TensorElementDataType::Uint64
			| TensorElementDataType::Bool => {
				// primitive data is already suitably laid out in memory; provide it to
				// onnxruntime as is
				let (shape, ptr, ptr_len, guard) = input.into_parts();
				let shape_ptr: *const i64 = shape.as_ptr();
				let shape_len = shape.len();

				let tensor_values_ptr: *mut std::ffi::c_void = ptr as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(ptr_len * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::into_tensor_element_data_type().into(),
						value_ptr_ptr
					) -> Error::CreateTensorWithData;
					nonNull(value_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
				guard
			}
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 | TensorElementDataType::Float16 => {
				// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
				let (shape, ptr, ptr_len, guard) = input.into_parts();
				let shape_ptr: *const i64 = shape.as_ptr();
				let shape_len = shape.len();

				let tensor_values_ptr: *mut std::ffi::c_void = ptr as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(ptr_len * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::into_tensor_element_data_type().into(),
						value_ptr_ptr
					) -> Error::CreateTensorWithData;
					nonNull(value_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
				guard
			}
			TensorElementDataType::String => unreachable!()
		};

		assert_non_null_pointer(value_ptr, "Value")?;

		Ok(Value {
			inner: ValueInner::RustOwned {
				ptr: value_ptr,
				_array: guard,
				_memory_info: memory_info
			}
		})
	}

	/// Construct a [`Value`] from a Rust-owned [`CowArray`].
	///
	/// `allocator` is required to be `Some` when converting a String tensor. See [`crate::Session::allocator`].
	pub fn from_string_array<T: Utf8Data + Debug + Clone + 'static>(allocator: &Allocator, input: impl OrtInput<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		let value_ptr_ptr: *mut *mut ort_sys::OrtValue = &mut value_ptr;

		let (shape, data) = input.ref_parts();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		// create tensor without data -- data is filled in later
		ortsys![
			unsafe CreateTensorAsOrtValue(allocator.ptr, shape_ptr, shape_len as _, TensorElementDataType::String.into(), value_ptr_ptr)
				-> Error::CreateTensor
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
				ptr: value_ptr,
				_array: Box::new(()),
				_memory_info: memory_info
			}
		})
	}

	pub(crate) fn ptr(&self) -> *mut ort_sys::OrtValue {
		match &self.inner {
			ValueInner::CppOwned { ptr, .. } => *ptr,
			ValueInner::RustOwned { ptr, .. } => *ptr
		}
	}

	/// Returns `true` if this value is a tensor, or false if it is another type (sequence, map)
	pub fn is_tensor(&self) -> Result<bool> {
		let mut result = 0;
		ortsys![unsafe IsTensor(self.ptr(), &mut result) -> Error::GetTensorElementType];
		Ok(result == 1)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: Clone + 'static, D: Dimension + 'static> OrtInput for &'i CowArray<'v, T, D>
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
impl<T: Clone + 'static, D: Dimension + 'static> OrtInput for &mut ArcArray<T, D> {
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
impl<T: Clone + 'static, D: Dimension + 'static> OrtInput for Array<T, D> {
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
impl<'v, T: Clone + 'static, D: Dimension + 'static> OrtInput for ArrayView<'v, T, D> {
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

impl<T: Clone + 'static> OrtInput for (Vec<i64>, Arc<Box<[T]>>) {
	type Item = T;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]) {
		let shape = self.0.clone();
		let data = self.1.deref();
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

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'i, 'v, T: IntoTensorElementDataType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for Value
where
	'i: 'v
{
	type Error = Error;
	fn try_from(arr: &'i CowArray<'v, T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: IntoTensorElementDataType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&mut ArcArray<T, D>> for Value {
	type Error = Error;
	fn try_from(arr: &mut ArcArray<T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: IntoTensorElementDataType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<Array<T, D>> for Value {
	type Error = Error;
	fn try_from(arr: Array<T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: IntoTensorElementDataType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for Value {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

impl Drop for Value {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseValue(self.ptr())];
		match &mut self.inner {
			ValueInner::CppOwned { ptr, .. } => *ptr = ptr::null_mut(),
			ValueInner::RustOwned { ptr, .. } => *ptr = ptr::null_mut()
		}
	}
}
