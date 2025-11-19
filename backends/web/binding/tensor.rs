use alloc::{string::ToString, vec::Vec};

use js_sys::JsString;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use web_sys::{HtmlImageElement, ImageBitmap, ImageData, WebGlTexture};

use crate::binding::DataType;

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "UPPERCASE")]
pub enum ImageFormat {
	Rgb,
	Rgba,
	Bgr,
	Rbg
}

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "UPPERCASE")]
pub enum ImageTensorLayout {
	Nhwc,
	Nchw
}

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ImageDataType {
	Float32,
	Uint8
}

impl Into<DataType> for ImageDataType {
	fn into(self) -> DataType {
		match self {
			Self::Float32 => DataType::Float32,
			Self::Uint8 => DataType::Uint8
		}
	}
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum ImageNormOption {
	Splat(f32),
	PerChannel([f32; 3]),
	PerChannelWithAlpha([f32; 4])
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ImageNorm {
	pub bias: Option<ImageNormOption>,
	pub mean: Option<ImageNormOption>
}

impl ImageNorm {
	pub const fn imagenet(format: ImageFormat) -> ImageNorm {
		const RGB_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
		const RGB_STD: [f32; 3] = [0.229, 0.224, 0.225];
		ImageNorm {
			mean: Some(match format {
				ImageFormat::Rgb => ImageNormOption::PerChannel(RGB_MEAN),
				ImageFormat::Rgba => ImageNormOption::PerChannelWithAlpha([RGB_MEAN[0], RGB_MEAN[1], RGB_MEAN[2], 0.5]),
				ImageFormat::Bgr => ImageNormOption::PerChannel([RGB_MEAN[2], RGB_MEAN[1], RGB_MEAN[0]]),
				ImageFormat::Rbg => ImageNormOption::PerChannel([RGB_MEAN[0], RGB_MEAN[2], RGB_MEAN[1]])
			}),
			bias: Some(match format {
				ImageFormat::Rgb => ImageNormOption::PerChannel(RGB_STD),
				ImageFormat::Rgba => ImageNormOption::PerChannelWithAlpha([RGB_STD[0], RGB_STD[1], RGB_STD[2], 0.5]),
				ImageFormat::Bgr => ImageNormOption::PerChannel([RGB_STD[2], RGB_STD[1], RGB_STD[0]]),
				ImageFormat::Rbg => ImageNormOption::PerChannel([RGB_STD[0], RGB_STD[2], RGB_STD[1]])
			})
		}
	}
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TensorFromImageOptions {
	pub data_type: Option<ImageDataType>,
	pub norm: Option<ImageNorm>,
	pub resized_height: Option<u32>,
	pub resized_width: Option<u32>,
	pub tensor_format: Option<ImageFormat>,
	pub tensor_layout: Option<ImageTensorLayout>
}

impl TensorFromImageOptions {
	pub(crate) fn to_value(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
		serde_wasm_bindgen::to_value(self)
	}
}

#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct TensorFromUrlOptions {
	#[serde(flatten)]
	base: TensorFromImageOptions,
	pub width: Option<u32>,
	pub height: Option<u32>
}

impl TensorFromUrlOptions {
	pub(crate) fn to_value(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
		serde_wasm_bindgen::to_value(self)
	}
}

#[derive(Serialize)]
#[serde(transparent)]
pub struct DisposeFunction(#[serde(with = "serde_wasm_bindgen::preserve")] JsValue);

impl<T> From<T> for DisposeFunction
where
	T: FnOnce() + 'static
{
	fn from(value: T) -> Self {
		DisposeFunction(Closure::once_into_js(value))
	}
}

#[derive(Serialize)]
#[serde(transparent)]
pub struct DownloadFunction(#[serde(with = "serde_wasm_bindgen::preserve")] JsValue);

impl<T, F, E> From<T> for DownloadFunction
where
	T: FnOnce() -> F + 'static,
	F: Future<Output = Result<JsValue, E>> + 'static,
	E: core::error::Error
{
	fn from(value: T) -> Self {
		DownloadFunction(Closure::once_into_js(move || {
			wasm_bindgen_futures::future_to_promise(async move {
				match value().await {
					Ok(value) => Ok(value),
					Err(e) => Err(JsString::from(e.to_string()).into())
				}
			})
		}))
	}
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TensorFromTextureOptions {
	pub width: u32,
	pub height: u32,
	pub format: Option<ImageFormat>,
	pub dispose: Option<DisposeFunction>,
	pub download: Option<DownloadFunction>
}

#[wasm_bindgen]
#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum DataLocation {
	None = "none", // indicates tensor is disposed
	Cpu = "cpu",
	CpuPinned = "cpu-pinned", // what is *pinned* in WASM?
	Texture = "texture",
	GpuBuffer = "gpu-buffer",
	MlTensor = "ml-tensor"
}

#[wasm_bindgen]
extern "C" {
	#[wasm_bindgen(js_namespace = ort)]
	pub type Tensor;

	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromImage)]
	async fn from_image_data_raw(image_data: &ImageData, options: JsValue) -> Result<Tensor, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromImage)]
	async fn from_image_element_raw(element: &HtmlImageElement, options: JsValue) -> Result<Tensor, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromImage)]
	async fn from_image_bitmap_raw(bitmap: &ImageBitmap, options: JsValue) -> Result<Tensor, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromImage)]
	async fn from_image_url_raw(url: &str, options: JsValue) -> Result<Tensor, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromTexture)]
	fn from_texture(texture: &WebGlTexture, options: JsValue) -> Result<Tensor, JsValue>;
	#[cfg(web_sys_unstable_apis)]
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromGpuBuffer)]
	fn from_gpu_buffer(buffer: &web_sys::GpuBuffer, options: JsValue) -> Result<Tensor, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = Tensor, js_name = fromPinnedBuffer)]
	fn from_pinned_buffer(dtype: DataType, buffer: JsValue, dims: JsValue) -> Result<Tensor, JsValue>;

	#[wasm_bindgen(constructor, catch, js_namespace = ort, js_class = Tensor)]
	fn new_from_buffer_raw(dtype: DataType, buffer: JsValue, dims: JsValue) -> Result<Tensor, JsValue>;

	#[wasm_bindgen(structural, catch, method, getter, js_name = data)]
	pub fn data(this: &Tensor) -> Result<JsValue, JsValue>;
	#[wasm_bindgen(structural, method, getter, js_name = location)]
	pub fn location(this: &Tensor) -> DataLocation;
	#[wasm_bindgen(structural, method, getter, js_name = type)]
	pub fn dtype(this: &Tensor) -> DataType;
	#[wasm_bindgen(structural, method, getter, js_name = size)]
	pub fn size(this: &Tensor) -> usize;
	#[wasm_bindgen(structural, method, getter, js_name = dims)]
	pub fn dims(this: &Tensor) -> Vec<i32>;

	#[wasm_bindgen(structural, catch, method, js_name = getData)]
	pub async fn get_data(this: &Tensor) -> Result<JsValue, JsValue>;

	#[wasm_bindgen(structural, catch, method, js_name = dispose)]
	pub fn dispose(this: &Tensor) -> Result<(), JsValue>;
	#[wasm_bindgen(structural, catch, method, js_name = reshape)]
	fn reshape(this: &Tensor, dims: JsValue) -> Result<Tensor, JsValue>;
}

impl Tensor {
	pub async fn from_image_data(image_data: &ImageData, options: &TensorFromImageOptions) -> Result<Tensor, JsValue> {
		Self::from_image_data_raw(image_data, options.to_value()?).await
	}

	pub async fn from_image_element(element: &HtmlImageElement, options: &TensorFromImageOptions) -> Result<Tensor, JsValue> {
		Self::from_image_element_raw(element, options.to_value()?).await
	}

	pub async fn from_image_bitmap(bitmap: &ImageBitmap, options: &TensorFromImageOptions) -> Result<Tensor, JsValue> {
		Self::from_image_bitmap_raw(bitmap, options.to_value()?).await
	}

	pub async fn from_image_url(url: &str, options: &TensorFromUrlOptions) -> Result<Tensor, JsValue> {
		Self::from_image_url_raw(url, options.to_value()?).await
	}

	pub fn new_from_buffer(dtype: DataType, buffer: JsValue, dims: &[i32]) -> Result<Tensor, JsValue> {
		Self::new_from_buffer_raw(dtype, buffer, convert_dims(dims))
	}
}

fn convert_dims(dims: &[i32]) -> JsValue {
	dims.iter().map(|d| js_sys::Number::from(*d)).collect::<js_sys::Array>().into()
}
