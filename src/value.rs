use std::{any::Any, collections::HashMap, ffi, fmt::Debug, hash::Hash, marker::PhantomData, ops::Deref, ptr, sync::Arc};

#[cfg(feature = "ndarray")]
use ndarray::{ArcArray, Array, ArrayView, CowArray, Dimension, IxDyn};

#[cfg(feature = "ndarray")]
use crate::tensor::Tensor;
use crate::{
	error::{assert_non_null_pointer, status_to_result},
	memory::{Allocator, MemoryInfo},
	ortsys,
	session::SharedSessionInner,
	tensor::{ExtractTensorData, IntoTensorElementType, TensorElementType, Utf8Data},
	AllocatorType, Error, MemType, Result
};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ValueType {
	Tensor { ty: TensorElementType, dimensions: Vec<i64> },
	Sequence(Box<ValueType>),
	Map { key: TensorElementType, value: TensorElementType }
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
	},
	/// A version of `CppOwned` that does not belong to a session. Used exclusively in [`ValueRef`]s which are returned
	/// by `extract_sequence` and used temporarily in `extract_map`.
	///
	/// We forego holding onto an `Arc<SharedSessionInner>` here because:
	/// - a map value can be created independently of a session, and thus we wouldn't have anything to hold on to;
	/// - this is only ever used by `ValueRef`s, whos owner value (which *is* holding the session Arc) will outlive it.
	CppOwnedRef { ptr: *mut ort_sys::OrtValue }
}

/// A temporary version of [`Value`] with a lifetime specifier.
///
/// This is used exclusively by [`Value::extract_sequence`] to ensure the sequence value outlives its child elements.
#[derive(Debug)]
pub struct ValueRef<'v> {
	inner: Value,
	lifetime: PhantomData<&'v ()>
}

impl<'v> Deref for ValueRef<'v> {
	type Target = Value;

	fn deref(&self) -> &Self::Target {
		&self.inner
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

	unsafe fn from_raw_ref(ptr: *mut ort_sys::OrtValue) -> Value {
		Value {
			inner: ValueInner::CppOwnedRef { ptr }
		}
	}

	pub fn tensor_element_type(&self) -> Result<TensorElementType> {
		let mut tensor_info_ptr: *mut ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorTypeAndShape(self.ptr(), &mut tensor_info_ptr) -> Error::GetTensorTypeAndShape];

		let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
		ortsys![unsafe GetTensorElementType(tensor_info_ptr, &mut type_sys) -> Error::GetTensorElementType];
		assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

		Ok(type_sys.into())
	}

	pub fn dtype(&self) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTypeInfo(self.ptr(), &mut typeinfo_ptr) -> Error::GetTypeInfo; nonNull(typeinfo_ptr)];

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty)];
		status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToSequenceTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_sequence_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr)? }
			}
			_ => unreachable!()
		};

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		Ok(io_type)
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
			let data_type: TensorElementType = type_sys.into();
			if data_type != T::tensor_element_type() {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_type()
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
			let data_type: TensorElementType = type_sys.into();
			if data_type != T::tensor_element_type() {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_type()
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

				Ok((node_dims, unsafe { std::slice::from_raw_parts(output_array_ptr, len as _) }))
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}

	pub fn extract_sequence<'s>(&'s self, allocator: &Allocator) -> Result<Vec<ValueRef<'s>>> {
		match self.dtype()? {
			ValueType::Sequence(_) => {
				let mut len: ort_sys::size_t = 0;
				ortsys![unsafe GetValueCount(self.ptr(), &mut len) -> Error::ExtractSequence];

				let mut vec = Vec::with_capacity(len as usize);
				for i in 0..len {
					let mut value_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), i as _, allocator.ptr, &mut value_ptr) -> Error::ExtractSequence; nonNull(value_ptr)];

					vec.push(ValueRef {
						inner: unsafe { Value::from_raw_ref(value_ptr) },
						lifetime: PhantomData
					});
				}
				Ok(vec)
			}
			t => Err(Error::NotSequence(t))
		}
	}

	pub fn extract_map<K: ExtractTensorData + Clone + Hash + Eq, V: ExtractTensorData + Clone>(&self, allocator: &Allocator) -> Result<HashMap<K, V>> {
		match self.dtype()? {
			ValueType::Map { key, value } => {
				let k_type = K::tensor_element_type();
				if k_type != key {
					return Err(Error::InvalidMapKeyType { expected: k_type, actual: key });
				}
				let v_type = V::tensor_element_type();
				if v_type != value {
					return Err(Error::InvalidMapValueType { expected: v_type, actual: value });
				}

				let mut key_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 0, allocator.ptr, &mut key_tensor_ptr) -> Error::ExtractMap; nonNull(key_tensor_ptr)];
				let key_value = unsafe { Value::from_raw_ref(key_tensor_ptr) };
				let (key_tensor_shape, key_tensor) = key_value.extract_raw_tensor::<K>()?;

				let mut value_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 1, allocator.ptr, &mut value_tensor_ptr) -> Error::ExtractMap; nonNull(value_tensor_ptr)];
				let value_value = unsafe { Value::from_raw_ref(value_tensor_ptr) };
				let (value_tensor_shape, value_tensor) = value_value.extract_raw_tensor::<V>()?;

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

pub trait OrtInput {
	type Item;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]);
	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>);
}

impl Value {
	/// Construct a [`Value`] from a Rust-owned array.
	///
	/// `allocator` is required to be `Some` when converting a String tensor. See [`crate::Session::allocator`].
	pub fn from_array<T: IntoTensorElementType + Debug + Clone + 'static>(input: impl OrtInput<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;

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

				let tensor_values_ptr: *mut std::ffi::c_void = ptr as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
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

				let tensor_values_ptr: *mut std::ffi::c_void = ptr as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
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
				ptr: value_ptr,
				_array: guard,
				_memory_info: memory_info
			}
		})
	}

	/// Construct a [`Value`] from a Rust-owned array.
	pub fn from_string_array<T: Utf8Data + Debug + Clone + 'static>(allocator: &Allocator, input: impl OrtInput<Item = T>) -> Result<Value> {
		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let (shape, data) = input.ref_parts();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		// create tensor without data -- data is filled in later
		ortsys![
			unsafe CreateTensorAsOrtValue(allocator.ptr, shape_ptr, shape_len as _, TensorElementType::String.into(), &mut value_ptr)
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
				ptr: value_ptr,
				_array: Box::new(()),
				_memory_info: memory_info
			}
		})
	}

	pub(crate) fn ptr(&self) -> *mut ort_sys::OrtValue {
		match &self.inner {
			ValueInner::CppOwnedRef { ptr } => *ptr,
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

impl<T: Clone + Debug + 'static> OrtInput for (Vec<i64>, Arc<Box<[T]>>) {
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
impl<'i, 'v, T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&'i CowArray<'v, T, D>> for Value
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
impl<T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<&mut ArcArray<T, D>> for Value {
	type Error = Error;
	fn try_from(arr: &mut ArcArray<T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<Array<T, D>> for Value {
	type Error = Error;
	fn try_from(arr: Array<T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

#[cfg(feature = "ndarray")]
#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
impl<'v, T: IntoTensorElementType + Debug + Clone + 'static, D: Dimension + 'static> TryFrom<ArrayView<'v, T, D>> for Value {
	type Error = Error;
	fn try_from(arr: ArrayView<'v, T, D>) -> Result<Self, Self::Error> {
		Value::from_array(arr)
	}
}

impl<T: IntoTensorElementType + Debug + Clone + 'static> TryFrom<(Vec<i64>, Arc<Box<[T]>>)> for Value {
	type Error = Error;
	fn try_from(d: (Vec<i64>, Arc<Box<[T]>>)) -> Result<Self, Self::Error> {
		Value::from_array(d)
	}
}

impl Drop for Value {
	#[tracing::instrument]
	fn drop(&mut self) {
		let ptr = self.ptr();
		tracing::trace!(
			"dropping {} value at {ptr:p}",
			match &self.inner {
				ValueInner::RustOwned { .. } => "rust-owned",
				ValueInner::CppOwned { .. } | ValueInner::CppOwnedRef { .. } => "cpp-owned"
			}
		);
		ortsys![unsafe ReleaseValue(ptr)];
	}
}

pub(crate) unsafe fn extract_data_type_from_tensor_info(info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo) -> Result<ValueType> {
	let mut type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(info_ptr, &mut type_sys) -> Error::GetTensorElementType];
	assert_ne!(type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
	// This transmute should be safe since its value is read from GetTensorElementType, which we must trust
	let mut num_dims = 0;
	ortsys![GetDimensionsCount(info_ptr, &mut num_dims) -> Error::GetDimensionsCount];

	let mut node_dims: Vec<i64> = vec![0; num_dims as _];
	ortsys![GetDimensions(info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> Error::GetDimensions];

	Ok(ValueType::Tensor {
		ty: type_sys.into(),
		dimensions: node_dims
	})
}

pub(crate) unsafe fn extract_data_type_from_sequence_info(info_ptr: *const ort_sys::OrtSequenceTypeInfo) -> Result<ValueType> {
	let mut element_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
	ortsys![GetSequenceElementType(info_ptr, &mut element_type_info) -> Error::GetSequenceElementType];

	let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
	let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(element_type_info, &mut ty)];
	status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;

	match ty {
		ort_sys::ONNXType::ONNX_TYPE_TENSOR => {
			let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
			ortsys![unsafe CastTypeInfoToTensorInfo(element_type_info, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
			let ty = unsafe { extract_data_type_from_tensor_info(info_ptr)? };
			Ok(ValueType::Sequence(Box::new(ty)))
		}
		ort_sys::ONNXType::ONNX_TYPE_MAP => {
			let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
			ortsys![unsafe CastTypeInfoToMapTypeInfo(element_type_info, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
			let ty = unsafe { extract_data_type_from_map_info(info_ptr)? };
			Ok(ValueType::Sequence(Box::new(ty)))
		}
		_ => unreachable!()
	}
}

pub(crate) unsafe fn extract_data_type_from_map_info(info_ptr: *const ort_sys::OrtMapTypeInfo) -> Result<ValueType> {
	let mut key_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetMapKeyType(info_ptr, &mut key_type_sys) -> Error::GetMapKeyType];
	assert_ne!(key_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	let mut value_type_info: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();
	ortsys![GetMapValueType(info_ptr, &mut value_type_info) -> Error::GetMapValueType];
	let mut value_info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
	ortsys![unsafe CastTypeInfoToTensorInfo(value_type_info, &mut value_info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(value_info_ptr)];
	let mut value_type_sys = ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	ortsys![GetTensorElementType(value_info_ptr, &mut value_type_sys) -> Error::GetTensorElementType];
	assert_ne!(value_type_sys, ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);

	Ok(ValueType::Map {
		key: key_type_sys.into(),
		value: value_type_sys.into()
	})
}

#[cfg(test)]
mod tests {
	use ndarray::{ArcArray1, Array1, CowArray};

	use crate::*;

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_value() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let value = Value::from_array(Array1::from_vec(v.clone()))?;
		assert!(value.is_tensor()?);
		assert_eq!(value.tensor_element_type()?, TensorElementType::Float32);
		assert_eq!(
			value.dtype()?,
			ValueType::Tensor {
				ty: TensorElementType::Float32,
				dimensions: vec![v.len() as i64]
			}
		);

		let (shape, data) = value.extract_raw_tensor::<f32>()?;
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

		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		let cow = CowArray::from(Array1::from_vec(v.clone()));
		let value = Value::from_array(&cow)?;
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		let owned = Array1::from_vec(v.clone());
		let value = Value::from_array(owned.view())?;
		drop(owned);
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];
		let value = Value::from_array((shape, Arc::clone(&arc)))?;
		drop(arc);
		assert_eq!(value.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}
}
