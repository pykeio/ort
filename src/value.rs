use std::{ffi, fmt::Debug, ptr, sync::Arc};

use ndarray::{Array, ArrayView, CowArray, Dimension, IxDyn};

use crate::{
	error::assert_non_null_pointer,
	memory::MemoryInfo,
	ortsys,
	session::SessionPointerHolder,
	sys,
	tensor::{IntoTensorElementDataType, OrtOwnedTensor, TensorDataToType, TensorElementDataType},
	AllocatorType, MemType, OrtError, OrtResult
};

#[derive(Debug)]
pub enum DynArrayRef<'v> {
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
	Bool(CowArray<'v, bool, IxDyn>),
	Double(CowArray<'v, f64, IxDyn>),
	Uint32(CowArray<'v, u32, IxDyn>),
	Uint64(CowArray<'v, u64, IxDyn>),
	String(CowArray<'v, String, IxDyn>)
}

impl<'v> DynArrayRef<'v> {
	pub fn shape(&self) -> &[usize] {
		match self {
			DynArrayRef::Float(x) => x.shape(),
			#[cfg(feature = "half")]
			DynArrayRef::Float16(x) => x.shape(),
			#[cfg(feature = "half")]
			DynArrayRef::Bfloat16(x) => x.shape(),
			DynArrayRef::Uint8(x) => x.shape(),
			DynArrayRef::Int8(x) => x.shape(),
			DynArrayRef::Uint16(x) => x.shape(),
			DynArrayRef::Int16(x) => x.shape(),
			DynArrayRef::Int32(x) => x.shape(),
			DynArrayRef::Int64(x) => x.shape(),
			DynArrayRef::Bool(x) => x.shape(),
			DynArrayRef::Double(x) => x.shape(),
			DynArrayRef::Uint32(x) => x.shape(),
			DynArrayRef::Uint64(x) => x.shape(),
			DynArrayRef::String(x) => x.shape()
		}
	}
}

macro_rules! impl_convert_trait {
	($type_:ty, $variant:expr) => {
		impl<'v, D: Dimension> From<ArrayView<'v, $type_, D>> for DynArrayRef<'v> {
			fn from(array: ArrayView<'v, $type_, D>) -> DynArrayRef<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		impl<'v, D: Dimension> From<Array<$type_, D>> for DynArrayRef<'v> {
			fn from(array: Array<$type_, D>) -> DynArrayRef<'v> {
				$variant(CowArray::from(array.into_dyn()))
			}
		}
		impl<'v, D: Dimension> From<CowArray<'v, $type_, D>> for DynArrayRef<'v> {
			fn from(array: CowArray<'v, $type_, D>) -> DynArrayRef<'v> {
				$variant(array.into_dyn())
			}
		}
	};
}

impl_convert_trait!(f32, DynArrayRef::Float);
#[cfg(feature = "half")]
impl_convert_trait!(half::f16, DynArrayRef::Float16);
#[cfg(feature = "half")]
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
pub enum Value<'v> {
	RustOwned {
		ptr: *mut sys::OrtValue,
		array: DynArrayRef<'v>,
		memory_info: MemoryInfo
	},
	CppOwned {
		ptr: *mut sys::OrtValue,
		session: Arc<SessionPointerHolder>
	}
}

unsafe impl<'v> Send for Value<'v> {}
unsafe impl<'v> Sync for Value<'v> {}

impl Value<'static> {
	pub fn from_raw(ptr: *mut sys::OrtValue, session: Arc<SessionPointerHolder>) -> Value<'static> {
		Value::CppOwned { ptr, session }
	}
}

impl<'v> Value<'v> {
	#[allow(clippy::not_unsafe_ptr_arg_deref)]
	pub fn from_array<'i, T: IntoTensorElementDataType + Debug + Clone>(
		allocator_ptr: *mut sys::OrtAllocator,
		array: &'i CowArray<'v, T, IxDyn>
	) -> OrtResult<Value<'v>>
	where
		'i: 'v,
		DynArrayRef<'v>: From<CowArray<'v, T, IxDyn>>
	{
		let mut contiguous_array: CowArray<'v, T, IxDyn> = array.as_standard_layout();

		let memory_info = MemoryInfo::new_cpu(AllocatorType::Arena, MemType::Default)?;

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
			| TensorElementDataType::Uint64
			| TensorElementDataType::Bool => {
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
		}

		assert_non_null_pointer(value_ptr, "Value")?;

		Ok(Value::RustOwned {
			ptr: value_ptr,
			array: contiguous_array.into(),
			memory_info
		})
	}

	pub fn ptr(&self) -> *mut sys::OrtValue {
		match self {
			Self::CppOwned { ptr, .. } => *ptr,
			Self::RustOwned { ptr, .. } => *ptr
		}
	}

	pub fn is_tensor(&self) -> OrtResult<bool> {
		let mut result = 0;
		ortsys![unsafe IsTensor(self.ptr(), &mut result) -> OrtError::GetTensorElementType];
		Ok(result == 1)
	}

	pub fn try_extract<'t, T>(&self) -> OrtResult<OrtOwnedTensor<'t, T, IxDyn>>
	where
		T: TensorDataToType + Clone + Debug,
		'v: 't
	{
		let mut tensor_info_ptr: *mut sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
		ortsys![unsafe GetTensorTypeAndShape(self.ptr(), &mut tensor_info_ptr) -> OrtError::GetTensorTypeAndShape];

		let res = {
			let mut num_dims = 0;
			ortsys![unsafe GetDimensionsCount(tensor_info_ptr, &mut num_dims) -> OrtError::GetDimensionsCount];

			let mut node_dims: Vec<i64> = vec![0; num_dims as _];
			ortsys![unsafe GetDimensions(tensor_info_ptr, node_dims.as_mut_ptr(), num_dims as _) -> OrtError::GetDimensions];
			let shape = IxDyn(&node_dims.iter().map(|&n| n as usize).collect::<Vec<_>>());

			let mut type_sys = sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
			ortsys![unsafe GetTensorElementType(tensor_info_ptr, &mut type_sys) -> OrtError::GetTensorElementType];
			assert_ne!(type_sys, sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
			let data_type: TensorElementDataType = type_sys.into();
			if data_type != T::tensor_element_data_type() {
				Err(OrtError::DataTypeMismatch {
					actual: data_type,
					requested: T::tensor_element_data_type()
				})
			} else {
				// Note: Both tensor and array will point to the same data, nothing is copied.
				// As such, there is no need to free the pointer used to create the ArrayView.
				assert_ne!(self.ptr(), ptr::null_mut());

				let mut is_tensor = 0;
				ortsys![unsafe IsTensor(self.ptr(), &mut is_tensor) -> OrtError::FailedTensorCheck];
				assert_eq!(is_tensor, 1);

				let mut len = 0;
				ortsys![unsafe GetTensorShapeElementCount(tensor_info_ptr, &mut len) -> OrtError::GetTensorShapeElementCount];

				let data = T::extract_data(shape, len as _, self.ptr())?;
				Ok(OrtOwnedTensor { data })
			}
		};
		ortsys![unsafe ReleaseTensorTypeAndShapeInfo(tensor_info_ptr)];
		res
	}
}

impl<'v> Drop for Value<'v> {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseValue(self.ptr())];
		match self {
			Self::CppOwned { ptr, .. } => *ptr = ptr::null_mut(),
			Self::RustOwned { ptr, .. } => *ptr = ptr::null_mut()
		}
	}
}
