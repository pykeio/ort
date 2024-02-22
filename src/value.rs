use std::{
	any::Any,
	collections::HashMap,
	ffi,
	fmt::Debug,
	hash::Hash,
	marker::PhantomData,
	ops::Deref,
	os::raw::c_char,
	ptr::{self, NonNull},
	string::FromUtf8Error,
	sync::Arc
};

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
	AllocatorType, Error, ExtractTensorDataView, MemoryType, Result
};

/// The type of a [`Value`], or a session input/output.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{Session, Value, ValueType, TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let session = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
/// // `ValueType`s can be obtained from session inputs/outputs:
/// let input = &session.inputs[0];
/// assert_eq!(
/// 	input.input_type,
/// 	ValueType::Tensor {
/// 		ty: TensorElementType::Float32,
/// 		// Our model has 3 dynamic dimensions, represented by -1
/// 		dimensions: vec![-1, -1, -1, 3]
/// 	}
/// );
///
/// // Or by `Value`s created in Rust or output by a session.
/// let value = Value::from_array((vec![5], Arc::new(vec![1_i64, 2, 3, 4, 5].into_boxed_slice())))?;
/// assert_eq!(
/// 	value.dtype()?,
/// 	ValueType::Tensor {
/// 		ty: TensorElementType::Int64,
/// 		dimensions: vec![5]
/// 	}
/// );
/// # 	Ok(())
/// # }
/// ```
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ValueType {
	/// Value is a tensor/multi-dimensional array.
	Tensor {
		/// Element type of the tensor.
		ty: TensorElementType,
		/// Dimensions of the tensor. If an exact dimension is not known (i.e. a dynamic dimension as part of an
		/// [`crate::Input`]/[`crate::Output`]), the dimension will be `-1`.
		///
		/// Actual tensor values, which have a known dimension, will always have positive (>1) dimensions.
		dimensions: Vec<i64>
	},
	/// A sequence (vector) of other `Value`s.
	///
	/// [Per ONNX spec](https://onnx.ai/onnx/intro/concepts.html#other-types), only sequences of tensors and maps are allowed.
	Sequence(Box<ValueType>),
	/// A map/dictionary from one element type to another.
	Map {
		/// The map key type. Allowed types are:
		/// - [`TensorElementType::Int8`]
		/// - [`TensorElementType::Int16`]
		/// - [`TensorElementType::Int32`]
		/// - [`TensorElementType::Int64`]
		/// - [`TensorElementType::Uint8`]
		/// - [`TensorElementType::Uint16`]
		/// - [`TensorElementType::Uint32`]
		/// - [`TensorElementType::Uint64`]
		/// - [`TensorElementType::String`]
		key: TensorElementType,
		/// The map value type.
		value: TensorElementType
	}
}

impl ValueType {
	/// Returns the dimensions of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let value = Value::from_array((vec![5], Arc::new(vec![1_i64, 2, 3, 4, 5].into_boxed_slice())))?;
	/// assert_eq!(value.dtype()?.tensor_dimensions(), Some(&vec![5]));
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn tensor_dimensions(&self) -> Option<&Vec<i64>> {
		match self {
			ValueType::Tensor { dimensions, .. } => Some(dimensions),
			_ => None
		}
	}

	/// Returns the element type of this value type if it is a tensor, or `None` if it is a sequence or map.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let value = Value::from_array((vec![5], Arc::new(vec![1_i64, 2, 3, 4, 5].into_boxed_slice())))?;
	/// assert_eq!(value.dtype()?.tensor_type(), Some(TensorElementType::Int64));
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn tensor_type(&self) -> Option<TensorElementType> {
		match self {
			ValueType::Tensor { ty, .. } => Some(*ty),
			_ => None
		}
	}

	/// Returns `true` if this value type is a tensor.
	#[inline]
	#[must_use]
	pub fn is_tensor(&self) -> bool {
		matches!(self, ValueType::Tensor { .. })
	}

	/// Returns `true` if this value type is a sequence.
	#[inline]
	#[must_use]
	pub fn is_sequence(&self) -> bool {
		matches!(self, ValueType::Sequence { .. })
	}

	/// Returns `true` if this value type is a map.
	#[inline]
	#[must_use]
	pub fn is_map(&self) -> bool {
		matches!(self, ValueType::Map { .. })
	}
}

#[derive(Debug)]
pub(crate) enum ValueInner {
	RustOwned {
		ptr: NonNull<ort_sys::OrtValue>,
		_array: Box<dyn Any>,
		_memory_info: MemoryInfo
	},
	CppOwned {
		ptr: NonNull<ort_sys::OrtValue>,
		/// Hold [`SharedSessionInner`] to ensure that the value can stay alive after the main session is dropped.
		_session: Arc<SharedSessionInner>
	},
	/// A version of `CppOwned` that does not belong to a session. Used exclusively in [`ValueRef`]s which are returned
	/// by `extract_sequence` and used temporarily in `extract_map`.
	///
	/// We forego holding onto an `Arc<SharedSessionInner>` here because:
	/// - a map value can be created independently of a session, and thus we wouldn't have anything to hold on to;
	/// - this is only ever used by `ValueRef`s, whos owner value (which *is* holding the session Arc) will outlive it.
	CppOwnedRef { ptr: NonNull<ort_sys::OrtValue> }
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

/// A [`Value`] contains data for inputs/outputs in ONNX Runtime graphs. [`Value`]s can hold a tensor, sequence
/// (array/vector), or map.
///
/// ## Creation
/// `Value`s can be created via methods like [`Value::from_array`], or as the output from running a [`crate::Session`].
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{Session, Value, ValueType, TensorElementType};
/// # fn main() -> ort::Result<()> {
/// # 	let upsample = Session::builder()?.with_model_from_file("tests/data/upsample.onnx")?;
/// // Create a value from a raw data vector
/// let value = Value::from_array((vec![1, 1, 1, 3], Arc::new(vec![1.0_f32, 2.0, 3.0].into_boxed_slice())))?;
///
/// // Create a value from an `ndarray::Array`
/// #[cfg(feature = "ndarray")]
/// let value = Value::from_array(ndarray::Array4::<f32>::zeros((1, 16, 16, 3)))?;
///
/// // Get a value from a session's output
/// let value = &upsample.run(ort::inputs![value]?)?[0];
/// # 	Ok(())
/// # }
/// ```
///
/// See [`Value::from_array`] for more details on what tensor values are accepted.
///
/// ## Usage
/// You can access the data in a `Value` by using the relevant `extract` methods: [`Value::extract_tensor`] &
/// [`Value::extract_raw_tensor`], [`Value::extract_sequence`], and [`Value::extract_map`].
#[derive(Debug)]
pub struct Value {
	inner: ValueInner
}

unsafe impl Send for Value {}

impl Value {
	/// Returns the data type of this [`Value`].
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
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = ndarray::Array4::<f32>::ones((1, 16, 16, 3));
	/// let value = Value::from_array(array.view())?;
	///
	/// let extracted = value.extract_tensor::<f32>()?;
	/// assert_eq!(array.view().into_dyn(), *extracted.view());
	/// # 	Ok(())
	/// # }
	/// ```
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
			if data_type == T::tensor_element_type() {
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
			} else {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_type()
				})
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}

	/// Attempt to extract the underlying data into a "raw" view tuple, consisting of the tensor's dimensions and a view
	/// into its data.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let value = Value::from_array((vec![5], Arc::new(array.clone().into_boxed_slice())))?;
	///
	/// let (extracted_shape, extracted_data) = value.extract_raw_tensor::<i64>()?;
	/// assert_eq!(extracted_data, &array);
	/// assert_eq!(extracted_shape, [5]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_tensor<T>(&self) -> Result<(Vec<i64>, &[T])>
	where
		T: ExtractTensorDataView + Clone + Debug
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
			if data_type == T::tensor_element_type() {
				// Note: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the slice.
				assert_ne!(self.ptr(), ptr::null_mut());

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(self.ptr(), &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let mut output_array_ptr: *mut T = ptr::null_mut();
				let output_array_ptr_ptr: *mut *mut T = &mut output_array_ptr;
				let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
				ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

				let mut len = 0;
				ortsys![unsafe GetTensorShapeElementCount(tensor_info_ptr, &mut len) -> Error::GetTensorShapeElementCount];

				Ok((node_dims, unsafe { std::slice::from_raw_parts(output_array_ptr, len as _) }))
			} else {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_type()
				})
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}

	/// Attempt to extract the underlying string data into a "raw" data tuple, consisting of the tensor's dimensions and
	/// an owned `Vec` of its data.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Allocator, Session, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let allocator = Allocator::default();
	/// let array = vec!["hello", "world"];
	/// let value = Value::from_string_array(&allocator, (vec![2], Arc::new(array.clone().into_boxed_slice())))?;
	///
	/// let (extracted_shape, extracted_data) = value.extract_raw_string_tensor()?;
	/// assert_eq!(extracted_data, array);
	/// assert_eq!(extracted_shape, [2]);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn extract_raw_string_tensor(&self) -> Result<(Vec<i64>, Vec<String>)> {
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
			if data_type == TensorElementType::String {
				// Note: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the slice.
				assert_ne!(self.ptr(), ptr::null_mut());

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(self.ptr(), &mut is_tensor) -> Error::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let mut output_array_ptr: *mut c_char = ptr::null_mut();
				let output_array_ptr_ptr: *mut *mut c_char = &mut output_array_ptr;
				let output_array_ptr_ptr_void: *mut *mut std::ffi::c_void = output_array_ptr_ptr.cast();
				ortsys![unsafe GetTensorMutableData(self.ptr(), output_array_ptr_ptr_void) -> Error::GetTensorMutableData; nonNull(output_array_ptr)];

				let mut len: ort_sys::size_t = 0;
				ortsys![unsafe GetTensorShapeElementCount(tensor_info_ptr, &mut len) -> Error::GetTensorShapeElementCount];
				// Total length of string data, not including \0 suffix
				let mut total_length = 0;
				ortsys![unsafe GetStringTensorDataLength(self.ptr(), &mut total_length) -> Error::GetStringTensorDataLength];

				// In the JNI impl of this, tensor_element_len was included in addition to total_length,
				// but that seems contrary to the docs of GetStringTensorDataLength, and those extra bytes
				// don't seem to be written to in practice either.
				// If the string data actually did go farther, it would panic below when using the offset
				// data to get slices for each string.
				let mut string_contents = vec![0u8; total_length as _];
				// one extra slot so that the total length can go in the last one, making all per-string
				// length calculations easy
				let mut offsets = vec![0; len as usize + 1];

				ortsys![unsafe GetStringTensorContent(self.ptr(), string_contents.as_mut_ptr().cast(), total_length as _, offsets.as_mut_ptr(), len as _) -> Error::GetStringTensorContent];

				// final offset = overall length so that per-string length calculations work for the last string
				debug_assert_eq!(0, offsets[len as usize]);
				offsets[len as usize] = total_length;

				let strings = offsets
					// offsets has 1 extra offset past the end so that all windows work
					.windows(2)
					.map(|w| {
						let slice = &string_contents[w[0] as _..w[1] as _];
						String::from_utf8(slice.into())
					})
					.collect::<Result<Vec<String>, FromUtf8Error>>()
					.map_err(Error::StringFromUtf8Error)?;

				Ok((node_dims, strings))
			} else {
				Err(Error::DataTypeMismatch {
					actual: data_type,
					requested: TensorElementType::String
				})
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
					ortsys![unsafe GetValue(self.ptr(), i as _, allocator.ptr.as_ptr(), &mut value_ptr) -> Error::ExtractSequence; nonNull(value_ptr)];

					vec.push(ValueRef {
						inner: unsafe { Value::from_ptr(NonNull::new_unchecked(value_ptr), None) },
						lifetime: PhantomData
					});
				}
				Ok(vec)
			}
			t => Err(Error::NotSequence(t))
		}
	}

	pub fn extract_map<K: ExtractTensorDataView + Clone + Hash + Eq, V: ExtractTensorDataView + Clone>(&self, allocator: &Allocator) -> Result<HashMap<K, V>> {
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
				ortsys![unsafe GetValue(self.ptr(), 0, allocator.ptr.as_ptr(), &mut key_tensor_ptr) -> Error::ExtractMap; nonNull(key_tensor_ptr)];
				let key_value = unsafe { Value::from_ptr(NonNull::new_unchecked(key_tensor_ptr), None) };
				let (key_tensor_shape, key_tensor) = key_value.extract_raw_tensor::<K>()?;

				let mut value_tensor_ptr = ptr::null_mut();
				ortsys![unsafe GetValue(self.ptr(), 1, allocator.ptr.as_ptr(), &mut value_tensor_ptr) -> Error::ExtractMap; nonNull(value_tensor_ptr)];
				let value_value = unsafe { Value::from_ptr(NonNull::new_unchecked(value_tensor_ptr), None) };
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

	/// Construct a [`Value`] from a C++ [`ort_sys::OrtValue`] pointer.
	///
	/// If the value belongs to a session (i.e. if it is returned from [`crate::Session::run`] or
	/// [`crate::IoBinding::run`]), you must provide the [`SharedSessionInner`] (acquired from
	/// [`crate::Session::inner`]). This ensures the session is not dropped until the value is.
	///
	/// # Safety
	///
	/// - `ptr` must be a valid pointer to an [`ort_sys::OrtValue`].
	/// - `session` must be `Some` for values returned from a session.
	#[must_use]
	pub unsafe fn from_ptr(ptr: NonNull<ort_sys::OrtValue>, session: Option<Arc<SharedSessionInner>>) -> Value {
		match session {
			Some(session) => Value {
				inner: ValueInner::CppOwned { ptr, _session: session }
			},
			None => Value {
				inner: ValueInner::CppOwnedRef { ptr }
			}
		}
	}
}

pub trait IntoValueTensor {
	type Item;

	fn ref_parts(&self) -> (Vec<i64>, &[Self::Item]);
	fn into_parts(self) -> (Vec<i64>, *mut Self::Item, usize, Box<dyn Any>);
}

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

	pub(crate) fn ptr(&self) -> *mut ort_sys::OrtValue {
		match &self.inner {
			ValueInner::CppOwnedRef { ptr } | ValueInner::CppOwned { ptr, .. } | ValueInner::RustOwned { ptr, .. } => ptr.as_ptr()
		}
	}

	/// Returns `true` if this value is a tensor, or `false` if it is another type (sequence, map).
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::Value;
	/// # fn main() -> ort::Result<()> {
	/// // Create a tensor from a raw data vector
	/// let tensor_value = Value::from_array((vec![3], Arc::new(vec![1.0_f32, 2.0, 3.0].into_boxed_slice())))?;
	/// assert!(tensor_value.is_tensor()?);
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn is_tensor(&self) -> Result<bool> {
		let mut result = 0;
		ortsys![unsafe IsTensor(self.ptr(), &mut result) -> Error::GetTensorElementType];
		Ok(result == 1)
	}
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
	use std::sync::Arc;

	use ndarray::{ArcArray1, Array1, CowArray};

	use crate::*;

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

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_string_tensor_ndarray() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = Array1::from_vec(vec!["hello world".to_string(), "こんにちは世界".to_string()]);

		let value = Value::from_string_array(&allocator, v.view())?;
		let extracted = value.extract_tensor::<String>()?;
		assert_eq!(*extracted.view(), v.into_dyn().view());

		Ok(())
	}

	#[test]
	fn test_string_tensor_raw() -> crate::Result<()> {
		let allocator = Allocator::default();
		let v = vec!["hello world".to_string(), "こんにちは世界".to_string()];

		let value = Value::from_string_array(&allocator, (vec![v.len() as i64], v.clone().into_boxed_slice()))?;
		let (extracted_shape, extracted_view) = value.extract_raw_string_tensor()?;
		assert_eq!(extracted_shape, [v.len() as i64]);
		assert_eq!(extracted_view, v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_inputs() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let shape = vec![v.len() as i64];
		let value_arc_box = Value::from_array((shape.clone(), Arc::new(v.clone().into_boxed_slice())))?;
		let value_box = Value::from_array((shape.clone(), v.clone().into_boxed_slice()))?;
		let value_vec = Value::from_array((shape.clone(), v.clone()))?;
		let value_slice = Value::from_array((shape, &v[..]))?;

		assert_eq!(value_arc_box.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_box.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_vec.extract_raw_tensor::<f32>()?.1, &v);
		assert_eq!(value_slice.extract_raw_tensor::<f32>()?.1, &v);

		Ok(())
	}
}
