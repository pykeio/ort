use alloc::{boxed::Box, ffi::CString, vec::Vec};
use core::{ffi::c_void, ptr::NonNull, slice};

use js_sys::{Array, Uint8Array};
use ort::{AsPointer, value::ValueTypeMarker};
use ort_sys::{OrtErrorCode, stub::Error as SysError};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlImageElement, ImageBitmap, ImageData};

pub use crate::binding::{ImageFormat, ImageNorm, ImageNormOption, ImageTensorLayout};
use crate::{
	Error,
	binding::{self, DataType, ImageDataType},
	memory::{HostBuffer, MemoryInfo},
	util::{num_elements, value_to_string}
};

pub(crate) const TENSOR_SENTINEL: [u8; 4] = [0xFC, 0x86, 0xA5, 0x39];

pub(crate) enum TensorData {
	/// Data is stored in WASM linear memory and can be immediately accessed.
	RustView {
		ptr: *mut c_void,
		byte_len: usize,
		_owned: Option<HostBuffer>
	},
	/// Data is stored outside of WASM linear memory (i.e. session output, or a tensor created from anything other than
	/// a Rust slice) and would need to be retrieved if we try to extract this tensor.
	External { buffer: Option<HostBuffer> }
}

#[repr(C)]
pub(crate) struct Tensor {
	sentinel: [u8; 4],
	pub(crate) js: binding::Tensor,
	pub(crate) data: TensorData,
	pub(crate) memory_info: MemoryInfo,
	pub(crate) shape: Vec<i64>,
	dispose_on_drop: bool
}

impl Tensor {
	pub(crate) unsafe fn from_ptr(dtype: binding::DataType, ptr: *mut c_void, byte_len: usize, dims: &[i32]) -> Result<Self, JsValue> {
		if dims.iter().any(|d| *d < 0) || tensor_byte_size(dtype, num_elements(dims)) != Some(byte_len) {
			return Err(JsValue::from_str(&format!("buffer of {byte_len} bytes does not match tensor shape {dims:?} of type {dtype:?}")));
		}

		let tensor = binding::Tensor::new_from_buffer(dtype, unsafe { buffer_from_ptr(dtype, ptr, byte_len)? }, dims)?;
		Ok(Self {
			sentinel: TENSOR_SENTINEL,
			memory_info: MemoryInfo::from_location(tensor.location()),
			shape: tensor.dims(),
			js: tensor,
			data: TensorData::RustView { ptr, byte_len, _owned: None },
			dispose_on_drop: false
		})
	}

	pub(crate) fn new_owned(dtype: binding::DataType, dims: &[i32]) -> Result<Self, JsValue> {
		if dims.iter().any(|d| *d < 0) {
			return Err(JsValue::from_str(&format!("invalid shape {dims:?}")));
		}

		if dtype == binding::DataType::String {
			let data: Array = (0..num_elements(dims)).map(|_| JsValue::from_str("")).collect();
			return Ok(Self::from_tensor(binding::Tensor::new_from_buffer(dtype, data.into(), dims)?));
		}

		let byte_len = tensor_byte_size(dtype, num_elements(dims)).ok_or_else(|| JsValue::from_str("invalid data type"))?;
		let buffer = HostBuffer::new_zeroed(byte_len).map_err(|e| JsValue::from_str(e.message()))?;
		let mut tensor = unsafe { Self::from_ptr(dtype, buffer.as_ptr().cast(), byte_len, dims) }?;
		tensor.data = TensorData::RustView {
			ptr: buffer.as_ptr().cast(),
			byte_len,
			_owned: Some(buffer)
		};
		Ok(tensor)
	}

	pub(crate) fn from_tensor(tensor: binding::Tensor) -> Self {
		Self {
			sentinel: TENSOR_SENTINEL,
			memory_info: MemoryInfo::from_location(tensor.location()),
			shape: tensor.dims(),
			js: tensor,
			data: TensorData::External { buffer: None },
			dispose_on_drop: false
		}
	}

	pub(crate) fn from_session_output(tensor: binding::Tensor) -> Self {
		let mut tensor = Self::from_tensor(tensor);
		tensor.dispose_on_drop = true;
		tensor
	}

	pub(crate) fn dtype(&self) -> DataType {
		self.js.dtype()
	}

	pub(crate) fn as_runnable(&self) -> Result<binding::Tensor, JsValue> {
		let dims: Vec<i32> = self.shape.iter().map(|x| *x as i32).collect();
		match &self.data {
			TensorData::RustView { ptr, byte_len, .. } => {
				binding::Tensor::new_from_buffer(self.dtype(), unsafe { buffer_from_ptr(self.dtype(), *ptr, *byte_len) }?, &dims)
			}
			TensorData::External { buffer: Some(buffer) } if self.memory_info.is_cpu() => {
				let dtype = self.dtype();
				binding::Tensor::new_from_buffer(dtype, copy_buffer(dtype, buffer.as_slice())?, &dims)
			}
			TensorData::External { .. } => Ok(self.js.clone().unchecked_into())
		}
	}

	pub(crate) fn data_ptr(&mut self) -> Result<*mut c_void, SysError> {
		if self.dtype() == DataType::String {
			return Err(SysError::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "GetTensorMutableData does not support string tensors"));
		}
		let is_cpu = self.memory_info.is_cpu();
		match &mut self.data {
			TensorData::RustView { ptr, .. } => Ok(*ptr),
			TensorData::External { buffer: Some(buffer) } => Ok(buffer.as_ptr().cast()),
			TensorData::External { buffer } if is_cpu => {
				let data = self.js.data().map_err(|e| SysError::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?;
				let bytes = bytes_view(&data);
				let mut new = HostBuffer::new_zeroed(bytes.byte_length() as usize)?;
				bytes.copy_to(new.as_mut_slice());
				Ok(buffer.insert(new).as_ptr().cast())
			}
			TensorData::External { buffer: None } => {
				Err(SysError::new(OrtErrorCode::ORT_FAIL, "External data is not synchronized; you should call `TensorExt::sync` first."))
			}
		}
	}

	pub(crate) fn string_array(&self) -> Result<Array, SysError> {
		if self.dtype() != DataType::String {
			return Err(SysError::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "tensor is not a string tensor"));
		}
		self.js
			.data()
			.map(Array::unchecked_from_js)
			.map_err(|e| SysError::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))
	}

	pub(crate) fn strings(&self) -> Result<Vec<String>, SysError> {
		Ok(self.string_array()?.iter().map(|s| s.as_string().unwrap_or_default()).collect())
	}

	pub(crate) async fn copy_from(&mut self, src: &binding::Tensor) -> Result<(), SysError> {
		let (src_dtype, src_dims) = (src.dtype(), src.dims());
		if src_dtype != self.dtype() || src_dims != self.shape {
			return Err(SysError::new(
				OrtErrorCode::ORT_INVALID_ARGUMENT,
				format!(
					"shape mismatch when attempting to reuse tensor buffer; expected {:?} ({:?}), got {src_dims:?} ({src_dtype:?})",
					self.shape,
					self.dtype()
				)
			));
		}

		let data = src
			.get_data()
			.await
			.map_err(|e| SysError::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?;
		if src_dtype == DataType::String {
			let dst = self.string_array()?;
			for (i, value) in Array::unchecked_from_js(data).iter().enumerate() {
				dst.set(i as u32, value);
			}
			return Ok(());
		}

		let bytes = bytes_view(&data);
		match &mut self.data {
			TensorData::RustView { ptr, byte_len, .. } => {
				bytes.copy_to(unsafe { slice::from_raw_parts_mut(ptr.cast(), *byte_len) });
			}
			TensorData::External { buffer } => {
				if let Some(buffer) = buffer {
					bytes.copy_to(buffer.as_mut_slice());
				}
				if let Ok(dst) = self.js.data() {
					bytes_view(&dst).set(&bytes, 0);
				}
			}
		}
		Ok(())
	}

	pub(crate) async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()> {
		if self.dtype() == DataType::String {
			return Ok(());
		}

		match direction {
			SyncDirection::Rust => {
				let data = self.js.get_data().await?;
				let bytes = bytes_view(&data);
				match &mut self.data {
					TensorData::RustView { ptr, byte_len, .. } => {
						bytes.copy_to(unsafe { core::slice::from_raw_parts_mut(ptr.cast(), *byte_len) });
					}
					TensorData::External { buffer } => {
						let buffer = match buffer {
							Some(buffer) => buffer,
							None => buffer.insert(HostBuffer::new_zeroed(bytes.byte_length() as usize).map_err(|e| Error::new(e.message()))?)
						};
						bytes.copy_to(buffer.as_mut_slice());
					}
				}
			}
			SyncDirection::Runtime => {
				let Ok(data) = self.js.data() else {
					// we have a download function, but no upload...
					return Err(Error::new(
						"Cannot synchronize Rust data to a runtime tensor that is not on the CPU; modify the WebGPU/WebGL buffer directly."
					));
				};
				let bytes = bytes_view(&data);
				bytes.copy_from(match &self.data {
					TensorData::RustView { ptr, byte_len, .. } => unsafe { core::slice::from_raw_parts(ptr.cast(), *byte_len) },
					TensorData::External { buffer } => {
						let Some(buffer) = buffer else {
							return Ok(());
						};
						buffer.as_slice()
					}
				});
			}
		}
		Ok(())
	}
}

impl Drop for Tensor {
	fn drop(&mut self) {
		if self.dispose_on_drop {
			let _ = self.js.dispose();
		}
	}
}

fn bytes_view(data: &JsValue) -> Uint8Array {
	let generic_typed_array = Uint8Array::unchecked_from_js_ref(data);
	Uint8Array::new_with_byte_offset_and_length(&generic_typed_array.buffer(), generic_typed_array.byte_offset(), generic_typed_array.byte_length())
}

pub(crate) fn tensor_byte_size(dtype: binding::DataType, num_elements: usize) -> Option<usize> {
	Some(match dtype {
		binding::DataType::Int4 | binding::DataType::Uint4 => num_elements.div_ceil(2),
		binding::DataType::Bool | binding::DataType::Uint8 | binding::DataType::Int8 => num_elements,
		binding::DataType::Uint16 | binding::DataType::Int16 | binding::DataType::Float16 => num_elements * 2,
		binding::DataType::Uint32 | binding::DataType::Int32 | binding::DataType::Float32 => num_elements * 4,
		binding::DataType::Uint64 | binding::DataType::Int64 | binding::DataType::Float64 => num_elements * 8,
		binding::DataType::String | binding::DataType::__Invalid => return None
	})
}

// hope you like match arms

pub(crate) unsafe fn buffer_from_ptr(dtype: binding::DataType, ptr: *mut c_void, byte_len: usize) -> Result<JsValue, JsValue> {
	if byte_len == 0 {
		return copy_buffer(dtype, &[]);
	}

	Ok(match dtype {
		binding::DataType::Bool | binding::DataType::Uint8 | binding::DataType::Int4 | binding::DataType::Uint4 => {
			unsafe { js_sys::Uint8Array::view(slice::from_raw_parts(ptr.cast(), byte_len)) }.into()
		}
		binding::DataType::Int8 => unsafe { js_sys::Int8Array::view(slice::from_raw_parts(ptr.cast(), byte_len)) }.into(),
		// hoping uint16array works for float16
		binding::DataType::Uint16 | binding::DataType::Float16 => unsafe { js_sys::Uint16Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 2)) }.into(),
		binding::DataType::Int16 => unsafe { js_sys::Int16Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 2)) }.into(),
		binding::DataType::Uint32 => unsafe { js_sys::Uint32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Int32 => unsafe { js_sys::Int32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Uint64 => unsafe { js_sys::BigUint64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::Int64 => unsafe { js_sys::BigInt64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::Float32 => unsafe { js_sys::Float32Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 4)) }.into(),
		binding::DataType::Float64 => unsafe { js_sys::Float64Array::view(slice::from_raw_parts(ptr.cast(), byte_len / 8)) }.into(),
		binding::DataType::String | binding::DataType::__Invalid => return Err(JsValue::from_str("invalid or unsupported data type"))
	})
}

fn copy_buffer(dtype: binding::DataType, bytes: &[u8]) -> Result<JsValue, JsValue> {
	let copy = Uint8Array::new_from_slice(bytes);
	let buffer = copy.buffer();
	let len = bytes.len() as u32;
	Ok(match dtype {
		binding::DataType::Bool | binding::DataType::Uint8 | binding::DataType::Int4 | binding::DataType::Uint4 => copy.into(),
		binding::DataType::Int8 => js_sys::Int8Array::new_with_byte_offset_and_length(&buffer, 0, len).into(),
		binding::DataType::Uint16 | binding::DataType::Float16 => js_sys::Uint16Array::new_with_byte_offset_and_length(&buffer, 0, len / 2).into(),
		binding::DataType::Int16 => js_sys::Int16Array::new_with_byte_offset_and_length(&buffer, 0, len / 2).into(),
		binding::DataType::Uint32 => js_sys::Uint32Array::new_with_byte_offset_and_length(&buffer, 0, len / 4).into(),
		binding::DataType::Int32 => js_sys::Int32Array::new_with_byte_offset_and_length(&buffer, 0, len / 4).into(),
		binding::DataType::Uint64 => js_sys::BigUint64Array::new_with_byte_offset_and_length(&buffer, 0, len / 8).into(),
		binding::DataType::Int64 => js_sys::BigInt64Array::new_with_byte_offset_and_length(&buffer, 0, len / 8).into(),
		binding::DataType::Float32 => js_sys::Float32Array::new_with_byte_offset_and_length(&buffer, 0, len / 4).into(),
		binding::DataType::Float64 => js_sys::Float64Array::new_with_byte_offset_and_length(&buffer, 0, len / 8).into(),
		binding::DataType::String | binding::DataType::__Invalid => return Err(JsValue::from_str("invalid or unsupported data type"))
	})
}

pub(crate) fn dtype_to_onnx(dtype: binding::DataType) -> ort_sys::ONNXTensorElementDataType {
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
		binding::DataType::__Invalid => ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
	}
}

pub(crate) fn onnx_to_dtype(dtype: ort_sys::ONNXTensorElementDataType) -> Option<binding::DataType> {
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

pub(crate) struct TypeInfo {
	pub(crate) dtype: ort_sys::ONNXTensorElementDataType,
	pub(crate) shape: Option<Vec<i64>>,
	pub(crate) symbolic_dims: Vec<CString>
}

impl TypeInfo {
	pub(crate) fn new(dtype: ort_sys::ONNXTensorElementDataType, shape: Vec<i64>) -> Self {
		Self {
			dtype,
			symbolic_dims: vec![CString::default(); shape.len()],
			shape: Some(shape)
		}
	}

	pub(crate) fn from_tensor(tensor: &Tensor) -> Self {
		Self::new(dtype_to_onnx(tensor.dtype()), tensor.shape.clone())
	}

	pub(crate) fn from_value_metadata(metadata: &binding::ValueMetadata) -> Self {
		let dtype = metadata
			.r#type
			.map_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED, dtype_to_onnx);
		let Some(shape) = &metadata.shape else {
			return Self {
				dtype,
				shape: None,
				symbolic_dims: Vec::new()
			};
		};

		Self {
			dtype,
			shape: Some(
				shape
					.iter()
					.map(|el| match el {
						binding::ShapeElement::Value(v) => *v,
						binding::ShapeElement::Named(_) => -1
					})
					.collect()
			),
			symbolic_dims: shape
				.iter()
				.map(|el| match el {
					binding::ShapeElement::Value(_) => CString::default(),
					binding::ShapeElement::Named(name) => CString::new(name.as_str()).unwrap_or_default()
				})
				.collect()
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncDirection {
	/// Synchronize tensor data from the device/runtime so that it is accessible to Rust code.
	Rust,
	/// Synchronize tensor data from Rust code so that it is accessible to the runtime.
	Runtime
}

pub trait ValueExt {
	private_trait!();

	/// Synchronize data between Rust & the runtime.
	///
	/// See the [top-level documentation][crate] for more information on synchronization.
	#[allow(async_fn_in_trait)]
	async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()>;
}

impl<T: ValueTypeMarker> ValueExt for ort::value::Value<T> {
	private_impl!();

	async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()> {
		unsafe { sync_value(self.ptr_mut(), direction) }.await
	}
}

impl<T: ValueTypeMarker> ValueExt for ort::value::ValueRefMut<'_, T> {
	private_impl!();

	async fn sync(&mut self, direction: SyncDirection) -> crate::Result<()> {
		unsafe { sync_value((**self).ptr().cast_mut(), direction) }.await
	}
}

async unsafe fn sync_value(ptr: *mut ort_sys::OrtValue, direction: SyncDirection) -> crate::Result<()> {
	// definitely safe regardless of what backend is used since it's highly improbable that a backend's tensor would
	// be smaller than 4 bytes (which is pointer size on wasm32)
	let sentinel: [u8; 4] = unsafe { core::ptr::read(ptr.cast()) };
	if sentinel != TENSOR_SENTINEL {
		return Err(Error::new("Cannot synchronize Value that was not created by ort-web"));
	}

	let tensor: &mut Tensor = unsafe { &mut *ptr.cast() };
	tensor.sync(direction).await
}

#[derive(Default)]
pub struct TensorFromImageOptions {
	pub norm: Option<ImageNorm>,
	pub resized_height: Option<u32>,
	pub resized_width: Option<u32>,
	pub tensor_format: Option<ImageFormat>,
	pub tensor_layout: Option<ImageTensorLayout>
}

#[derive(Default)]
pub struct TensorFromUrlOptions {
	pub norm: Option<ImageNorm>,
	pub resized_height: Option<u32>,
	pub resized_width: Option<u32>,
	pub tensor_format: Option<ImageFormat>,
	pub tensor_layout: Option<ImageTensorLayout>
}

#[allow(async_fn_in_trait)]
pub trait TensorFromImage: Sized {
	private_trait!();

	async fn from_image_data(image_data: &ImageData, options: TensorFromImageOptions) -> crate::Result<Self>;
	async fn from_image_element(image_element: &HtmlImageElement, options: TensorFromImageOptions) -> crate::Result<Self>;
	async fn from_image_bitmap(image_bitmap: &ImageBitmap, options: TensorFromImageOptions) -> crate::Result<Self>;
	async fn from_image_url(url: &str, options: TensorFromImageOptions, original_dimensions: Option<(u32, u32)>) -> crate::Result<Self>;
}

impl TensorFromImage for ort::value::Tensor<f32> {
	private_impl!();

	async fn from_image_data(image: &ImageData, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_data(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Float32),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_element(image: &HtmlImageElement, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_element(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Float32),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_bitmap(image: &ImageBitmap, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_bitmap(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Float32),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_url(url: &str, options: TensorFromImageOptions, original_dimensions: Option<(u32, u32)>) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_url(
				url,
				&binding::TensorFromUrlOptions {
					base: binding::TensorFromImageOptions {
						data_type: Some(ImageDataType::Float32),
						norm: options.norm,
						resized_height: options.resized_height,
						resized_width: options.resized_width,
						tensor_format: options.tensor_format,
						tensor_layout: options.tensor_layout
					},
					width: original_dimensions.map(|(w, _)| w),
					height: original_dimensions.map(|(_, h)| h)
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}
}

impl TensorFromImage for ort::value::Tensor<u8> {
	private_impl!();

	async fn from_image_data(image: &ImageData, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_data(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Uint8),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_element(image: &HtmlImageElement, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_element(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Uint8),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_bitmap(image: &ImageBitmap, options: TensorFromImageOptions) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_bitmap(
				image,
				&binding::TensorFromImageOptions {
					data_type: Some(ImageDataType::Uint8),
					norm: options.norm,
					resized_height: options.resized_height,
					resized_width: options.resized_width,
					tensor_format: options.tensor_format,
					tensor_layout: options.tensor_layout
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}

	async fn from_image_url(url: &str, options: TensorFromImageOptions, original_dimensions: Option<(u32, u32)>) -> crate::Result<Self> {
		let tensor = Tensor::from_tensor(
			binding::Tensor::from_image_url(
				url,
				&binding::TensorFromUrlOptions {
					base: binding::TensorFromImageOptions {
						data_type: Some(ImageDataType::Uint8),
						norm: options.norm,
						resized_height: options.resized_height,
						resized_width: options.resized_width,
						tensor_format: options.tensor_format,
						tensor_layout: options.tensor_layout
					},
					width: original_dimensions.map(|(w, _)| w),
					height: original_dimensions.map(|(_, h)| h)
				}
			)
			.await?
		);
		Ok(unsafe { ort::value::Tensor::from_ptr(NonNull::from_mut(Box::leak(Box::new(tensor))).cast(), None) })
	}
}
