use std::{collections::HashMap, ffi, fmt::Debug, marker::PhantomData, ptr};

use ndarray::{Array, ArrayView, CowArray, Dimension, IxDyn};

use crate::{
	error::assert_non_null_pointer,
	memory::MemoryInfo,
	ortsys, sys,
	tensor::{IntoTensorElementDataType, TensorElementDataType},
	OrtError, OrtResult
};

pub enum MapKey {
	Int8(i8),
	Int16(i16),
	Int32(i32),
	Int64(i64),
	Uint8(u8),
	Uint16(u16),
	Uint32(u32),
	Uint64(u64),
	String(String)
}

pub enum InputTensor<'v> {
	Float(CowArray<'v, f32, IxDyn>),
	#[cfg(feature = "half")]
	Float16(CowArray<'v, half::f16, IxDyn>),
	#[cfg(feature = "half")]
	Bfloat16(CowArray<'v, half::bf16, IxDyn>),
	Uint8(CowArray<'v, u8, IxDyn>),
	Int8(CowArray<'v, i8, IxDyn>),
	Uint16(CowArray<'v, u16, IxDyn>),
	Int16(CowArray<'v, i16, IxDyn>),
	Int32(CowArray<'v, i32, IxDyn>),
	Int64(CowArray<'v, i64, IxDyn>),
	Double(CowArray<'v, f64, IxDyn>),
	Uint32(CowArray<'v, u32, IxDyn>),
	Uint64(CowArray<'v, u64, IxDyn>),
	String(CowArray<'v, String, IxDyn>)
}

impl<'v> InputTensor<'v> {
	pub fn shape(&self) -> &[usize] {
		match self {
			InputTensor::Float(x) => x.shape(),
			#[cfg(feature = "half")]
			InputTensor::Float16(x) => x.shape(),
			#[cfg(feature = "half")]
			InputTensor::Bfloat16(x) => x.shape(),
			InputTensor::Uint8(x) => x.shape(),
			InputTensor::Int8(x) => x.shape(),
			InputTensor::Uint16(x) => x.shape(),
			InputTensor::Int16(x) => x.shape(),
			InputTensor::Int32(x) => x.shape(),
			InputTensor::Int64(x) => x.shape(),
			InputTensor::Double(x) => x.shape(),
			InputTensor::Uint32(x) => x.shape(),
			InputTensor::Uint64(x) => x.shape(),
			InputTensor::String(x) => x.shape()
		}
	}
}

macro_rules! impl_convert_trait {
	($type_:ty, $variant:expr) => {
		impl<'v, D: Dimension> From<ArrayView<'v, $type_, D>> for InputTensor<'v> {
			fn from(array: ArrayView<'v, $type_, D>) -> InputTensor<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		impl<'v, D: Dimension> From<Array<$type_, D>> for InputTensor<'v> {
			fn from(array: Array<$type_, D>) -> InputTensor<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		impl<'v, D: Dimension> From<CowArray<'v, $type_, D>> for InputTensor<'v> {
			fn from(array: CowArray<'v, $type_, D>) -> InputTensor<'v> {
				$variant(array.into_dyn())
			}
		}
	};
}

impl_convert_trait!(f32, InputTensor::Float);
#[cfg(feature = "half")]
impl_convert_trait!(half::f16, InputTensor::Float16);
#[cfg(feature = "half")]
impl_convert_trait!(half::bf16, InputTensor::Bfloat16);
impl_convert_trait!(u8, InputTensor::Uint8);
impl_convert_trait!(i8, InputTensor::Int8);
impl_convert_trait!(u16, InputTensor::Uint16);
impl_convert_trait!(i16, InputTensor::Int16);
impl_convert_trait!(i32, InputTensor::Int32);
impl_convert_trait!(i64, InputTensor::Int64);
impl_convert_trait!(f64, InputTensor::Double);
impl_convert_trait!(u32, InputTensor::Uint32);
impl_convert_trait!(u64, InputTensor::Uint64);
impl_convert_trait!(String, InputTensor::String);

pub enum InputValue<'v> {
	Tensor(InputTensor<'v>),
	Sequence(Vec<InputValue<'v>>),
	Map(HashMap<MapKey, InputValue<'v>>)
}

impl<'v, T: Into<InputTensor<'v>>> From<T> for InputValue<'v> {
	fn from(value: T) -> Self {
		Self::Tensor(value.into())
	}
}

impl<'v> InputValue<'v> {
	pub fn shape(&self) -> Vec<usize> {
		match self {
			InputValue::Tensor(tensor) => tensor.shape().to_vec(),
			InputValue::Sequence(seq) => vec![seq.len()],
			InputValue::Map(map) => vec![map.len()]
		}
	}
}

pub struct OrtRustOwnedValue<'m, 'v, T> {
	pub(crate) c_ptr: *mut sys::OrtValue,
	#[allow(unused)]
	owned_value: T,
	value_ref: PhantomData<&'v ()>,
	memory_info: PhantomData<&'m MemoryInfo>
}

impl<'m, 'v, T> OrtRustOwnedValue<'m, 'v, CowArray<'v, T, IxDyn>>
where
	//'m: 'v,
	T: IntoTensorElementDataType + Debug + Clone
{
	pub(crate) fn from_array<'i>(
		memory_info: &'m MemoryInfo,
		allocator_ptr: *mut sys::OrtAllocator,
		array: &'i CowArray<'v, T, IxDyn>
	) -> OrtResult<OrtRustOwnedValue<'m, 'v, CowArray<'v, T, IxDyn>>>
	where
		'v: 'm,
		'i: 'v
	{
		let mut contiguous_array: CowArray<'v, T, IxDyn> = array.as_standard_layout();

		let mut value_ptr: *mut sys::OrtValue = ptr::null_mut();
		let value_ptr_ptr: *mut *mut sys::OrtValue = &mut value_ptr;

		let shape: Vec<i64> = contiguous_array.shape().iter().map(|d| *d as i64).collect();
		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		match T::tensor_element_data_type() {
			TensorElementDataType::Float32
			| TensorElementDataType::Uint8
			| TensorElementDataType::Int8
			| TensorElementDataType::Uint16
			| TensorElementDataType::Int16
			| TensorElementDataType::Int32
			| TensorElementDataType::Int64
			| TensorElementDataType::Float64
			| TensorElementDataType::Uint32
			| TensorElementDataType::Uint64 => {
				// primitive data is already suitably laid out in memory; provide it to
				// onnxruntime as is
				let tensor_values_ptr: *mut std::ffi::c_void = contiguous_array.as_mut_ptr() as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(contiguous_array.len() * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::tensor_element_data_type().into(),
						value_ptr_ptr
					) -> OrtError::CreateTensorWithData;
					nonNull(value_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
			}
			#[cfg(feature = "half")]
			TensorElementDataType::Bfloat16 | TensorElementDataType::Float16 => {
				// f16 and bf16 are repr(transparent) to u16, so memory layout should be identical to onnxruntime
				let tensor_values_ptr: *mut std::ffi::c_void = contiguous_array.as_mut_ptr() as *mut std::ffi::c_void;
				assert_non_null_pointer(tensor_values_ptr, "TensorValues")?;

				ortsys![
					unsafe CreateTensorWithDataAsOrtValue(
						memory_info.ptr,
						tensor_values_ptr,
						(contiguous_array.len() * std::mem::size_of::<T>()) as _,
						shape_ptr,
						shape_len as _,
						T::tensor_element_data_type().into(),
						value_ptr_ptr
					) -> OrtError::CreateTensorWithData;
					nonNull(value_ptr)
				];

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(value_ptr, &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);
			}
			TensorElementDataType::String => {
				// create tensor without data -- data is filled in later
				ortsys![
					unsafe CreateTensorAsOrtValue(allocator_ptr, shape_ptr, shape_len as _, T::tensor_element_data_type().into(), value_ptr_ptr)
						-> OrtError::CreateTensor
				];

				// create null-terminated copies of each string, as per `FillStringTensor` docs
				let null_terminated_copies: Vec<ffi::CString> = contiguous_array
					.iter()
					.map(|elt| {
						let slice = elt.try_utf8_bytes().expect("String data type must provide utf8 bytes");
						ffi::CString::new(slice)
					})
					.collect::<std::result::Result<Vec<_>, _>>()
					.map_err(OrtError::FfiStringNull)?;

				let string_pointers = null_terminated_copies.iter().map(|cstring| cstring.as_ptr()).collect::<Vec<_>>();

				ortsys![unsafe FillStringTensor(value_ptr, string_pointers.as_ptr(), string_pointers.len() as _) -> OrtError::FillStringTensor];
			}
			_ => unimplemented!("Tensor element data type {:?} not yet implemented", T::tensor_element_data_type())
		}

		assert_non_null_pointer(value_ptr, "Value")?;

		Ok(OrtRustOwnedValue {
			c_ptr: value_ptr,
			owned_value: contiguous_array,
			value_ref: PhantomData,
			memory_info: PhantomData
		})
	}
}

pub enum InputOrtValue<'m, 'v> {
	FloatTensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, f32, IxDyn>>),
	#[cfg(feature = "half")]
	Float16Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, half::f16, IxDyn>>),
	#[cfg(feature = "half")]
	Bfloat16Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, half::bf16, IxDyn>>),
	Uint8Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, u8, IxDyn>>),
	Int8Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, i8, IxDyn>>),
	Uint16Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, u16, IxDyn>>),
	Int16Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, i16, IxDyn>>),
	Int32Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, i32, IxDyn>>),
	Int64Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, i64, IxDyn>>),
	DoubleTensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, f64, IxDyn>>),
	Uint32Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, u32, IxDyn>>),
	Uint64Tensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, u64, IxDyn>>),
	StringTensor(OrtRustOwnedValue<'m, 'v, CowArray<'v, String, IxDyn>>)
}

impl<'m, 'v> InputOrtValue<'m, 'v> {
	pub(crate) fn from_input_value<'i>(
		memory_info: &'m MemoryInfo,
		allocator_ptr: *mut sys::OrtAllocator,
		input_value: &'i InputValue<'v>
	) -> OrtResult<InputOrtValue<'m, 'v>>
	where
		'v: 'm,
		'i: 'v
	{
		match input_value {
			InputValue::Tensor(tensor) => match tensor {
				InputTensor::Float(array) => Ok(InputOrtValue::FloatTensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				#[cfg(feature = "half")]
				InputTensor::Float16(array) => Ok(InputOrtValue::Float16Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				#[cfg(feature = "half")]
				InputTensor::Bfloat16(array) => Ok(InputOrtValue::Bfloat16Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Uint8(array) => Ok(InputOrtValue::Uint8Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Int8(array) => Ok(InputOrtValue::Int8Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Uint16(array) => Ok(InputOrtValue::Uint16Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Int16(array) => Ok(InputOrtValue::Int16Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Int32(array) => Ok(InputOrtValue::Int32Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Int64(array) => Ok(InputOrtValue::Int64Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Double(array) => Ok(InputOrtValue::DoubleTensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Uint32(array) => Ok(InputOrtValue::Uint32Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::Uint64(array) => Ok(InputOrtValue::Uint64Tensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?)),
				InputTensor::String(array) => Ok(InputOrtValue::StringTensor(OrtRustOwnedValue::from_array(memory_info, allocator_ptr, array)?))
			},
			_ => unimplemented!()
		}
	}

	pub(crate) fn c_ptr(&self) -> *const sys::OrtValue {
		match self {
			InputOrtValue::FloatTensor(x) => x.c_ptr,
			#[cfg(feature = "half")]
			InputOrtValue::Float16Tensor(x) => x.c_ptr,
			#[cfg(feature = "half")]
			InputOrtValue::Bfloat16Tensor(x) => x.c_ptr,
			InputOrtValue::Uint8Tensor(x) => x.c_ptr,
			InputOrtValue::Int8Tensor(x) => x.c_ptr,
			InputOrtValue::Uint16Tensor(x) => x.c_ptr,
			InputOrtValue::Int16Tensor(x) => x.c_ptr,
			InputOrtValue::Int32Tensor(x) => x.c_ptr,
			InputOrtValue::Int64Tensor(x) => x.c_ptr,
			InputOrtValue::DoubleTensor(x) => x.c_ptr,
			InputOrtValue::Uint32Tensor(x) => x.c_ptr,
			InputOrtValue::Uint64Tensor(x) => x.c_ptr,
			InputOrtValue::StringTensor(x) => x.c_ptr
		}
	}
}
