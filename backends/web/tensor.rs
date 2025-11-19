use alloc::{boxed::Box, vec::Vec};
use core::{ffi::c_void, slice};

use js_sys::Uint8Array;
use ort::{AsPointer, value::ValueTypeMarker};
use wasm_bindgen::{JsCast, JsValue};

use crate::{
	Error,
	binding::{self, DataType},
	memory::MemoryInfo,
	util::num_elements
};

pub const TENSOR_SENTINEL: [u8; 4] = [0xFC, 0x86, 0xA5, 0x39];

pub enum TensorData {
	/// Data is stored in WASM linear memory and can be immediately accessed.
	RustView { ptr: *mut c_void, byte_len: usize },
	/// Data is stored outside of WASM linear memory (i.e. session output, or a tensor created from anything other than
	/// a Rust slice) and would need to be retrieved if we try to extract this tensor.
	External { buffer: Option<Box<[u8]>> }
}

#[repr(C)]
pub struct Tensor {
	sentinel: [u8; 4],
	pub js: binding::Tensor,
	pub data: TensorData,
	pub memory_info: MemoryInfo
}

impl Tensor {
	pub unsafe fn from_ptr(dtype: binding::DataType, ptr: *mut c_void, byte_len: usize, dims: &[i32]) -> Result<Self, JsValue> {
		let tensor = binding::Tensor::new_from_buffer(dtype, unsafe { buffer_from_ptr(dtype, ptr, byte_len) }, dims)?;
		Ok(Self {
			sentinel: TENSOR_SENTINEL,
			memory_info: MemoryInfo { location: tensor.location() },
			js: tensor,
			data: TensorData::RustView { ptr, byte_len }
		})
	}

	pub fn from_tensor(tensor: binding::Tensor) -> Self {
		Self {
			sentinel: TENSOR_SENTINEL,
			memory_info: MemoryInfo { location: tensor.location() },
			js: tensor,
			data: TensorData::External { buffer: None }
		}
	}

	pub async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()> {
		match direction {
			SyncDirection::Rust => {
				let data = self.js.get_data().await?;

				// cast to some kind of typed array first, then convert to uint8array so we can properly copy
				let generic_typed_array = Uint8Array::unchecked_from_js(data);
				let bytes = Uint8Array::new_with_byte_offset_and_length(
					&generic_typed_array.buffer(),
					generic_typed_array.byte_offset(),
					generic_typed_array.byte_length()
				);
				match &mut self.data {
					TensorData::RustView { ptr, byte_len } => {
						bytes.copy_to(unsafe { core::slice::from_raw_parts_mut(ptr.cast(), *byte_len) });
					}
					TensorData::External { buffer } => {
						let buffer = match buffer {
							Some(buffer) => buffer,
							None => {
								*buffer = Some(vec![0; generic_typed_array.byte_length() as usize].into_boxed_slice());
								unsafe { buffer.as_mut().unwrap_unchecked() }
							}
						};
						bytes.copy_to(buffer);
					}
				}
			}
			SyncDirection::Runtime => {
				let Ok(generic_typed_array) = self.js.data().map(Uint8Array::unchecked_from_js) else {
					// we have a download function, but no upload...
					return Err(Error::new(
						"Cannot synchronize Rust data to a runtime tensor that is not on the CPU; modify the WebGPU/WebGL buffer directly."
					));
				};
				let bytes = Uint8Array::new_with_byte_offset_and_length(
					&generic_typed_array.buffer(),
					generic_typed_array.byte_offset(),
					generic_typed_array.byte_length()
				);
				bytes.copy_from(match &self.data {
					TensorData::RustView { ptr, byte_len } => unsafe { core::slice::from_raw_parts(ptr.cast(), *byte_len) },
					TensorData::External { buffer } => {
						let Some(buffer) = buffer else {
							return Ok(());
						};
						&*buffer
					}
				});
			}
		}
		Ok(())
	}
}

pub fn create_buffer(dtype: binding::DataType, shape: &[i32]) -> JsValue {
	let numel = num_elements(shape) as u32;
	match dtype {
		binding::DataType::Bool | binding::DataType::Uint8 => js_sys::Uint8Array::new_with_length(numel).into(),
		binding::DataType::Int8 => js_sys::Int8Array::new_with_length(numel).into(),
		binding::DataType::Uint16 => js_sys::Uint16Array::new_with_length(numel).into(),
		binding::DataType::Int16 => js_sys::Int16Array::new_with_length(numel).into(),
		binding::DataType::Uint32 => js_sys::Uint32Array::new_with_length(numel).into(),
		binding::DataType::Int32 => js_sys::Int32Array::new_with_length(numel).into(),
		binding::DataType::Uint64 => js_sys::BigUint64Array::new_with_length(numel).into(),
		binding::DataType::Int64 => js_sys::BigInt64Array::new_with_length(numel).into(),
		binding::DataType::Float32 => js_sys::Float32Array::new_with_length(numel).into(),
		binding::DataType::Float64 => js_sys::Float64Array::new_with_length(numel).into(),
		binding::DataType::Int4 | binding::DataType::Uint4 | binding::DataType::Float16 | binding::DataType::String => unimplemented!(),
		binding::DataType::__Invalid => unreachable!()
	}
}

pub unsafe fn buffer_from_ptr(dtype: binding::DataType, ptr: *mut c_void, byte_len: usize) -> JsValue {
	match dtype {
		binding::DataType::Bool | binding::DataType::Uint8 => unsafe { js_sys::Uint8Array::view(slice::from_raw_parts(ptr.cast(), byte_len)) }.into(),
		binding::DataType::Int8 => unsafe { js_sys::Int8Array::view(slice::from_raw_parts(ptr.cast(), byte_len)) }.into(),
		binding::DataType::Uint16 => unsafe { js_sys::Uint16Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 2)) }.into(),
		binding::DataType::Int16 => unsafe { js_sys::Int16Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 2)) }.into(),
		binding::DataType::Uint32 => unsafe { js_sys::Uint32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Int32 => unsafe { js_sys::Int32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Uint64 => unsafe { js_sys::BigUint64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::Int64 => unsafe { js_sys::BigInt64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::Float32 => unsafe { js_sys::Float32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Float64 => unsafe { js_sys::Float64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::Int4 | binding::DataType::Uint4 | binding::DataType::Float16 | binding::DataType::String => unimplemented!(),
		binding::DataType::__Invalid => unreachable!()
	}
}

pub fn dtype_to_onnx(dtype: binding::DataType) -> ort_sys::ONNXTensorElementDataType {
	match dtype {
		binding::DataType::String => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
		binding::DataType::Bool => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
		binding::DataType::Uint8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
		binding::DataType::Int8 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
		binding::DataType::Uint16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
		binding::DataType::Int16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
		binding::DataType::Uint32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
		binding::DataType::Int32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
		binding::DataType::Uint64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
		binding::DataType::Int64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		binding::DataType::Float16 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
		binding::DataType::Float32 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		binding::DataType::Float64 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
		binding::DataType::Int4 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
		binding::DataType::Uint4 => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
		binding::DataType::__Invalid => unreachable!()
	}
}

pub fn onnx_to_dtype(dtype: ort_sys::ONNXTensorElementDataType) -> Option<binding::DataType> {
	match dtype {
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING => Some(binding::DataType::String),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL => Some(binding::DataType::Bool),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 => Some(binding::DataType::Uint8),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 => Some(binding::DataType::Int8),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 => Some(binding::DataType::Uint16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 => Some(binding::DataType::Int16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 => Some(binding::DataType::Uint32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 => Some(binding::DataType::Int32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 => Some(binding::DataType::Uint64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => Some(binding::DataType::Int64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 => Some(binding::DataType::Float16),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => Some(binding::DataType::Float32),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE => Some(binding::DataType::Float64),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4 => Some(binding::DataType::Int4),
		ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 => Some(binding::DataType::Uint4),
		_ => None
	}
}

pub struct TypeInfo {
	pub dtype: ort_sys::ONNXTensorElementDataType,
	pub shape: Vec<i32>
}

impl TypeInfo {
	pub fn new_sys_from_tensor(tensor: &Tensor) -> *mut ort_sys::OrtTypeInfo {
		Self::new_sys(tensor.js.dtype(), tensor.js.dims())
	}

	pub fn new_sys_from_value_metadata(metadata: &binding::ValueMetadata) -> *mut ort_sys::OrtTypeInfo {
		Self::new_sys(
			metadata.r#type.unwrap(),
			metadata
				.shape
				.as_ref()
				.unwrap()
				.iter()
				.map(|el| match el {
					binding::ShapeElement::Value(v) => *v as i32,
					binding::ShapeElement::Named(_) => -1
				})
				.collect()
		)
	}

	pub fn new_sys(dtype: DataType, shape: Vec<i32>) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(Self { dtype: dtype_to_onnx(dtype), shape })) as *mut TypeInfo).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		unsafe { Box::from_raw(ptr.cast::<TypeInfo>()) }
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
	/// Synchronize tensor data from the device/runtime so that it is accessible to Rust code.
	Rust,
	/// Synchronize tensor data from Rust code so that it is accessible to the runtime.
	Runtime
}

pub trait ValueExt {
	crate::private_trait!();

	/// Synchronize data between Rust & the runtime.
	///
	/// See the [top-level documentation][crate] for more information on synchronization.
	#[allow(async_fn_in_trait)]
	async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()>;
}

impl<T: ValueTypeMarker> ValueExt for ort::value::Value<T> {
	crate::private_impl!();

	async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()> {
		let ptr = self.ptr_mut();
		// definitely safe regardless of what backend is used since it's highly improbable that a backend's tensor would be
		// smaller than 4 bytes (which is pointer size on wasm32)
		let sentinel: [u8; 4] = unsafe { core::ptr::read(ptr.cast()) };
		if sentinel != TENSOR_SENTINEL {
			return Err(Error::new("Cannot synchronize Value that was not created by ort-web"));
		}

		let tensor: &mut Tensor = unsafe { &mut *ptr.cast() };
		tensor.sync(direction).await
	}
}
