use std::{
	ffi::{CString, c_void},
	ptr::{self, NonNull},
	slice
};

use candle_core::{CpuStorage, Device, Storage};
use ort_sys::{ONNXTensorElementDataType, OrtErrorCode};

use crate::{
	Error,
	memory::{DEFAULT_CPU_MEMORY_INFO, MemoryInfo, alloc_host, free_host},
	util::{candle_error, convert_sys_to_dtype, element_size, num_elements}
};

pub(crate) enum TensorData {
	Host {
		ptr: *mut c_void,
		byte_len: usize,
		owned: bool,
		deleter: Option<NonNull<ort_sys::OrtAllocator>>
	},
	/// for non-CPU tensors
	Device(candle_core::Tensor)
}

pub(crate) struct Tensor {
	pub(crate) data: TensorData,
	pub(crate) dtype: ONNXTensorElementDataType,
	pub(crate) shape: Vec<i64>,
	pub(crate) memory_info: MemoryInfo
}

impl Tensor {
	pub(crate) fn new(dtype: ONNXTensorElementDataType, shape: Vec<i64>, memory_info: &MemoryInfo) -> Result<Self, Error> {
		let candle_dtype = convert_sys_to_dtype(dtype)?;
		let num_elements = num_elements(&shape)?;
		if memory_info.is_cpu() {
			let byte_len = num_elements
				.checked_mul(element_size(dtype))
				.ok_or_else(|| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("tensor shape {shape:?} is too large")))?;
			let ptr = alloc_host(byte_len, true);
			if ptr.is_null() {
				return Err(Error::new(OrtErrorCode::ORT_RUNTIME_EXCEPTION, "allocation failed"));
			}

			Ok(Self {
				data: TensorData::Host {
					ptr: ptr.cast(),
					byte_len,
					owned: true,
					deleter: None
				},
				dtype,
				shape,
				memory_info: DEFAULT_CPU_MEMORY_INFO.clone()
			})
		} else {
			let dims: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
			let tensor = candle_core::Tensor::zeros(dims, candle_dtype, &memory_info.device).map_err(candle_error)?;
			Ok(Self {
				data: TensorData::Device(tensor),
				dtype,
				shape,
				memory_info: memory_info.clone()
			})
		}
	}

	pub(crate) fn new_borrowed(
		dtype: ONNXTensorElementDataType,
		shape: Vec<i64>,
		ptr: *mut c_void,
		byte_len: usize,
		deleter: Option<NonNull<ort_sys::OrtAllocator>>
	) -> Result<Self, Error> {
		// just make sure its a valid type
		let _ = convert_sys_to_dtype(dtype)?;

		let expected_len = num_elements(&shape)?.checked_mul(element_size(dtype));
		if expected_len != Some(byte_len) {
			return Err(Error::new(
				OrtErrorCode::ORT_INVALID_ARGUMENT,
				format!("buffer of {byte_len} bytes does not match tensor shape {shape:?} of type {dtype:?}")
			));
		}

		Ok(Self {
			data: TensorData::Host { ptr, byte_len, owned: false, deleter },
			dtype,
			shape,
			memory_info: DEFAULT_CPU_MEMORY_INFO.clone()
		})
	}

	pub(crate) fn from_candle(tensor: candle_core::Tensor, dtype: ONNXTensorElementDataType) -> Result<Self, Error> {
		let shape = tensor.dims().iter().map(|x| *x as i64).collect();
		if tensor.device().is_cpu() {
			let mut out = Self::new(dtype, shape, &DEFAULT_CPU_MEMORY_INFO)?;
			out.copy_from(&tensor)?;
			Ok(out)
		} else {
			Ok(Self {
				memory_info: MemoryInfo::from_device(tensor.device()),
				data: TensorData::Device(tensor),
				dtype,
				shape
			})
		}
	}

	pub(crate) fn host_bytes(&self) -> Option<&[u8]> {
		match &self.data {
			TensorData::Host { byte_len: 0, .. } => Some(&[]),
			TensorData::Host { ptr, byte_len, .. } => Some(unsafe { slice::from_raw_parts(ptr.cast(), *byte_len) }),
			TensorData::Device(_) => None
		}
	}

	pub(crate) fn data_ptr(&self) -> Result<*mut c_void, Error> {
		match &self.data {
			TensorData::Host { ptr, .. } => Ok(*ptr),
			TensorData::Device(_) => Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, "GetTensorMutableData for non-CPU tensors not yet implemented"))
		}
	}

	pub(crate) fn to_candle(&self) -> Result<candle_core::Tensor, Error> {
		match &self.data {
			TensorData::Host { .. } => {
				let dims: Vec<usize> = self.shape.iter().map(|x| *x as usize).collect();
				candle_core::Tensor::from_raw_buffer(self.host_bytes().unwrap(), convert_sys_to_dtype(self.dtype)?, &dims, &Device::Cpu).map_err(candle_error)
			}
			TensorData::Device(tensor) => Ok(tensor.clone())
		}
	}

	pub(crate) fn copy_from(&mut self, src: &candle_core::Tensor) -> Result<(), Error> {
		if src.dtype() != convert_sys_to_dtype(self.dtype)?
			|| src.dims().len() != self.shape.len()
			|| src.dims().iter().zip(&self.shape).any(|(a, b)| *a as i64 != *b)
		{
			return Err(Error::new(
				OrtErrorCode::ORT_INVALID_ARGUMENT,
				format!(
					"shape mismatch when attempting to reuse tensor buffer; expected {:?} ({:?}), got {:?} ({:?})",
					self.shape,
					self.dtype,
					src.dims(),
					src.dtype()
				)
			));
		}

		match &mut self.data {
			TensorData::Host { ptr, byte_len, .. } => {
				let src = src.to_device(&Device::Cpu).and_then(|t| t.contiguous()).map_err(candle_error)?;
				let (storage, layout) = src.storage_and_layout();
				let Storage::Cpu(storage) = &*storage else {
					unreachable!()
				};

				let bytes = cpu_storage_bytes(storage)?;
				let elem_size = src.dtype().size_in_bytes();
				let Some((start, end)) = layout.contiguous_offsets() else {
					unreachable!()
				};
				let bytes = &bytes[start * elem_size..end * elem_size];
				debug_assert_eq!(bytes.len(), *byte_len);

				unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len()) };
			}
			TensorData::Device(tensor) => {
				*tensor = src.to_device(tensor.device()).and_then(|t| t.copy()).map_err(candle_error)?;
			}
		}
		Ok(())
	}

	pub(crate) fn copy(&self) -> Result<Tensor, Error> {
		match &self.data {
			TensorData::Host { .. } => {
				let out = Self::new(self.dtype, self.shape.clone(), &DEFAULT_CPU_MEMORY_INFO)?;
				let src = self.host_bytes().unwrap_or_default();
				unsafe { ptr::copy_nonoverlapping(src.as_ptr(), out.data_ptr()?.cast(), src.len()) };
				Ok(out)
			}
			TensorData::Device(tensor) => Ok(Self {
				data: TensorData::Device(tensor.copy().map_err(candle_error)?),
				dtype: self.dtype,
				shape: self.shape.clone(),
				memory_info: self.memory_info.clone()
			})
		}
	}
}

impl Drop for Tensor {
	fn drop(&mut self) {
		if let TensorData::Host { ptr, owned, deleter, .. } = self.data {
			if owned {
				unsafe { free_host(ptr.cast()) };
			} else if let Some(deleter) = deleter
				&& let Some(free) = unsafe { deleter.as_ref().Free }
			{
				unsafe { free(deleter.as_ptr(), ptr.cast()) };
			}
		}
	}
}

fn cpu_storage_bytes(storage: &CpuStorage) -> Result<&[u8], Error> {
	fn as_bytes<T>(v: &[T]) -> &[u8] {
		unsafe { slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
	}

	Ok(match storage {
		CpuStorage::U8(v) => v,
		CpuStorage::U32(v) => as_bytes(v),
		CpuStorage::I16(v) => as_bytes(v),
		CpuStorage::I32(v) => as_bytes(v),
		CpuStorage::I64(v) => as_bytes(v),
		CpuStorage::BF16(v) => as_bytes(v),
		CpuStorage::F16(v) => as_bytes(v),
		CpuStorage::F32(v) => as_bytes(v),
		CpuStorage::F64(v) => as_bytes(v),
		CpuStorage::F8E4M3(v) => as_bytes(v),
		_ => return Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, "unsupported tensor element type"))
	})
}

pub(crate) struct TypeInfo {
	pub(crate) dtype: ONNXTensorElementDataType,
	pub(crate) shape: Option<Vec<i64>>,
	pub(crate) symbolic_dims: Vec<CString>
}

impl TypeInfo {
	pub(crate) fn new(dtype: ONNXTensorElementDataType, shape: Vec<i64>) -> Self {
		Self {
			dtype,
			symbolic_dims: vec![CString::default(); shape.len()],
			shape: Some(shape)
		}
	}

	pub(crate) fn shape(&self) -> &[i64] {
		self.shape.as_deref().unwrap_or_default()
	}

	pub(crate) fn into_sys(self) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(self)) as *mut TypeInfo).cast()
	}

	pub(crate) unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		unsafe { Box::from_raw(ptr.cast::<TypeInfo>()) }
	}
}
