use alloc::{string::String, vec::Vec};
use std::collections::HashMap;

use js_sys::{JsString, Object, Reflect, Uint8Array};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::{binding::DataType, tensor::Tensor};

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
	Sequential,
	Parallel
}

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum GraphOptimizationLevel {
	Disabled,
	Basic,
	Layout,
	Extended,
	All
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebNNDeviceType {
	CPU,
	GPU,
	NPU
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum WebNNPowerPreference {
	Default,
	HighPerformance,
	LowPower
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum WebGPUPreferredLayout {
	NHWC,
	NCHW
}

#[derive(Serialize, Debug, Clone)]
#[serde(tag = "name", rename_all = "lowercase")]
pub enum ExecutionProvider {
	WASM,
	WebGL,
	#[serde(rename_all = "camelCase")]
	WebNN {
		device_type: Option<WebNNDeviceType>,
		num_threads: Option<u32>,
		power_preference: Option<WebNNPowerPreference>
	},
	#[serde(rename_all = "camelCase")]
	WebGPU {
		preferred_layout: Option<WebGPUPreferredLayout>
	}
}

#[derive(Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SessionOptions {
	pub enable_cpu_mem_arena: Option<bool>,
	pub enable_graph_capture: Option<bool>,
	pub enable_mem_pattern: Option<bool>,
	pub enable_profiling: Option<bool>,
	pub execution_mode: Option<ExecutionMode>,
	pub execution_providers: Option<Vec<ExecutionProvider>>,
	pub extra: Option<HashMap<String, String>>,
	pub free_dimension_override: Option<HashMap<String, i32>>,
	pub graph_optimization_level: Option<GraphOptimizationLevel>,
	pub inter_op_num_threads: Option<u32>,
	pub intra_op_num_threads: Option<u32>,
	pub log_id: Option<String>,
	pub log_severity_level: Option<u8>,
	pub log_verbosity_level: Option<u16>
}

impl SessionOptions {
	pub(crate) fn to_value(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
		self.serialize(&serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true))
	}
}

#[wasm_bindgen]
extern "C" {
	#[wasm_bindgen(js_namespace = ort)]
	pub type InferenceSession;

	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
	async fn create_from_uri_raw(uri: &str, options: JsValue) -> Result<InferenceSession, JsValue>;
	#[wasm_bindgen(catch, js_namespace = ort, static_method_of = InferenceSession, js_name = create)]
	async fn create_from_bytes_raw(buffer: &Uint8Array, options: JsValue) -> Result<InferenceSession, JsValue>;

	#[wasm_bindgen(catch, structural, method, js_name = startProfiling)]
	pub fn start_profiling(this: &InferenceSession) -> Result<(), JsValue>;
	#[wasm_bindgen(catch, structural, method, js_name = endProfiling)]
	pub fn end_profiling(this: &InferenceSession) -> Result<(), JsValue>;
	#[wasm_bindgen(catch, structural, method, js_name = release)]
	pub async fn release(this: &InferenceSession) -> Result<(), JsValue>;

	#[wasm_bindgen(structural, method, getter, js_name = inputMetadata)]
	fn input_metadata_raw(this: &InferenceSession) -> Vec<JsValue>;
	#[wasm_bindgen(structural, method, getter, js_name = outputMetadata)]
	fn output_metadata_raw(this: &InferenceSession) -> Vec<JsValue>;
	#[wasm_bindgen(structural, method, getter, js_name = inputNames)]
	fn input_names_raw(this: &InferenceSession) -> Vec<JsString>;
	#[wasm_bindgen(structural, method, getter, js_name = outputNames)]
	fn output_names_raw(this: &InferenceSession) -> Vec<JsString>;

	#[wasm_bindgen(catch, structural, method, js_name = run)]
	async fn run_raw(this: &InferenceSession, feeds: JsValue) -> Result<JsValue, JsValue>;
	#[wasm_bindgen(catch, structural, method, js_name = run)]
	async fn run_with_fetches_raw(this: &InferenceSession, feeds: JsValue, fetches: JsValue) -> Result<JsValue, JsValue>;
}

impl InferenceSession {
	pub async fn create_from_uri(uri: &str, options: &SessionOptions) -> Result<InferenceSession, JsValue> {
		InferenceSession::create_from_uri_raw(uri, options.to_value()?).await
	}
	pub async fn create_from_bytes(buffer: &Uint8Array, options: &SessionOptions) -> Result<InferenceSession, JsValue> {
		InferenceSession::create_from_bytes_raw(buffer, options.to_value()?).await
	}

	pub fn input_names(&self) -> Vec<String> {
		self.input_names_raw().into_iter().map(String::from).collect()
	}
	pub fn output_names(&self) -> Vec<String> {
		self.output_names_raw().into_iter().map(String::from).collect()
	}

	pub fn input_len(&self) -> usize {
		self.input_names_raw().len()
	}
	pub fn output_len(&self) -> usize {
		self.output_names_raw().len()
	}

	pub fn input_metadata(&self) -> Vec<ValueMetadata> {
		self.input_metadata_raw()
			.into_iter()
			.map(|x| serde_wasm_bindgen::from_value(x))
			.collect::<Result<Vec<_>, serde_wasm_bindgen::Error>>()
			.unwrap()
	}

	pub fn output_metadata(&self) -> Vec<ValueMetadata> {
		self.output_metadata_raw()
			.into_iter()
			.map(|x| serde_wasm_bindgen::from_value(x))
			.collect::<Result<Vec<_>, serde_wasm_bindgen::Error>>()
			.unwrap()
	}

	pub async fn run(&self, feeds: impl Iterator<Item = (&str, &Tensor)>) -> Result<Vec<(String, Tensor)>, JsValue> {
		let feeds_value = Object::new();
		for (name, tensor) in feeds {
			Reflect::set(&feeds_value, &JsValue::from_str(name), &tensor.js)?;
		}
		Self::to_outputs(self.run_raw(feeds_value.into()).await?)
	}

	pub async fn run_with_fetches(
		&self,
		feeds: impl Iterator<Item = (&str, &Tensor)>,
		fetches: impl Iterator<Item = (&str, Option<&Tensor>)>
	) -> Result<Vec<(String, Tensor)>, JsValue> {
		let feeds_value = Object::new();
		for (name, tensor) in feeds {
			Reflect::set(&feeds_value, &JsValue::from_str(name), &tensor.js)?;
		}
		let fetches_value = Object::new();
		for (name, tensor) in fetches {
			let null = JsValue::null();
			Reflect::set(
				&fetches_value,
				&JsValue::from_str(name),
				match tensor {
					Some(tensor) => &tensor.js,
					None => &null
				}
			)?;
		}
		Self::to_outputs(self.run_with_fetches_raw(feeds_value.into(), fetches_value.into()).await?)
	}

	fn to_outputs(value: JsValue) -> Result<Vec<(String, Tensor)>, JsValue> {
		Ok(Reflect::own_keys(&value)?
			.to_vec()
			.into_iter()
			.filter_map(|c| {
				c.dyn_ref::<JsString>().map(String::from).and_then(|k| {
					Reflect::get(&value, &c)
						.map(super::Tensor::unchecked_from_js)
						.ok()
						.map(|v| (k, Tensor::from_tensor(v)))
				})
			})
			.collect())
	}
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum ShapeElement {
	Named(String),
	Value(i64)
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ValueMetadata {
	pub is_tensor: bool,
	pub name: String,
	pub shape: Option<Vec<ShapeElement>>,
	pub r#type: Option<DataType>
}
