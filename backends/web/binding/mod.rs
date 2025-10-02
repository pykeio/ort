use js_sys::Boolean;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

mod session;
pub use self::session::*;
mod tensor;
pub use self::tensor::*;

#[wasm_bindgen]
#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
	Bool = "bool",
	Float16 = "float16",
	Float32 = "float32",
	Float64 = "float64",
	Int4 = "int4",
	Int8 = "int8",
	Int16 = "int16",
	Int32 = "int32",
	Int64 = "int64",
	Uint4 = "uint4",
	Uint8 = "uint8",
	Uint16 = "uint16",
	Uint32 = "uint32",
	Uint64 = "uint64",
	String = "string"
}

#[wasm_bindgen(module = "/_loader.js")]
extern "C" {
	#[wasm_bindgen(catch, js_name = "initRuntime")]
	pub async fn init_runtime(features: u8) -> Result<Boolean, JsValue>;
}

#[wasm_bindgen(module = "/_telemetry.js")]
extern "C" {
	#[wasm_bindgen(catch, js_name = "trackSessionInit")]
	pub fn track_session_init() -> Result<Boolean, JsValue>;
}
