//! `ort-web` is an [`ort`] backend that enables the usage of ONNX Runtime in the web.
//!
//! # Usage
//! ## CORS
//! `ort-web` dynamically fetches the required scripts & WASM binary at runtime. By default, it will fetch the build
//! from the `cdn.pyke.io` domain, so make sure it is accessible via CORS if you have that configured.
//!
//! You can also use a self-hosted build with [`Dist`]; see the [`api`](fn@api) function for an example. The scripts &
//! binary can be acquired from the `dist` folder of the [`onnxruntime-web` npm package](https://npmjs.com/package/onnxruntime-web).
//!
//! ### Telemetry
//! `ort-web` collects telemetry data by default and sends it to `signal.pyke.io`. This telemetry data helps us
//! understand how `ort-web` is being used so we can improve it. Zero PII is collected; you can see what is sent in
//! `_telemetry.js`. If you wish to contribute telemetry data, please allowlist `signal.pyke.io`; otherwise, it can be
//! disabled via [`EnvironmentBuilder::with_telemetry`](ort::environment::EnvironmentBuilder::with_telemetry).
//!
//! ## Initialization
//! `ort` must have the `alternative-backend` feature enabled, as this enables the usage of [`ort::set_api`].
//!
//! You can choose which build of ONNX Runtime to fetch by choosing any combination of these 3 feature flags:
//! [`FEATURE_WEBGL`], [`FEATURE_WEBGPU`], [`FEATURE_WEBNN`]. These enable the usage of the [WebGL][ort::ep::WebGL],
//! [WebGPU][ort::ep::WebGPU], and [WebNN][ort::ep::WebNN] EPs respectively. You can `|` features together to enable
//! multiple at once:
//!
//! ```no_run
//! use ort_web::{FEATURE_WEBGL, FEATURE_WEBGPU};
//! ort::set_api(ort_web::api(FEATURE_WEBGL | FEATURE_WEBGPU).await?);
//! ```
//!
//! You'll still need to configure the EPs on a per-session basis later like you would normally, but this allows you to
//! e.g. only fetch the CPU build if the user doesn't have hardware acceleration.
//!
//! ## Session creation
//! Sessions can only be created from a URL, or indirectly from memory - that means no
//! `SessionBuilder::commit_from_memory_directly` for `.ort` format models, and no `SessionBuilder::commit_from_file`.
//!
//! The remaining commit functions - `SessionBuilder::commit_from_url` and `SessionBuilder::commit_from_memory` are
//! marked `async` and need to be `await`ed. `commit_from_url` is always available when targeting WASM and does not
//! require the `fetch-models` feature flag to be enabled for `ort`.
//!
//! ## Inference
//! Only `Session::run_async` is supported; `Session::run` will always throw an error.
//!
//! Inference outputs are not synchronized by default (see the next section). If you need access to the data of all
//! session outputs from Rust, the [`sync_outputs`] function can be used to sync them all at once.
//!
//! ## Synchronization
//! ONNX Runtime is loaded as a separate WASM module, and `ort-web` acts as an intermediary between the two. There is no
//! mechanism in WASM for two modules to share memory, so tensors often need to be 'synchronized' when one side needs to
//! see data from the other.
//!
//! [`Tensor::new`](ort::value::Tensor::new) should never be used for creating inputs, as they start out allocated on
//! the ONNX Runtime side, thus requiring a sync (of empty data) to Rust before it can be written to. Prefer instead
//! [`Tensor::from_array`](ort::value::Tensor::from_array)/
//! [`TensorRef::from_array_view`](ort::value::TensorRef::from_array_view), as tensors created this way never require
//! synchronization.
//!
//! As previously stated, session outputs are **not** synchronized. If you wish to use their data in Rust, you must
//! either sync all outputs at once with [`sync_outputs`], or sync each tensor at a time (if you only use a few
//! outputs):
//! ```ignore
//! use ort_web::{TensorExt, SyncDirection};
//!
//! let mut outputs = session.run_async(ort::inputs![...]).await?;
//!
//! let mut bounding_boxes = outputs.remove("bounding_boxes").unwrap();
//! bounding_boxes.sync(SyncDirection::Rust).await?;
//!
//! // now we can use the data
//! let data = bounding_boxes.try_extract_tensor::<f32>()?;
//! ```
//!
//! Once a session output is `sync`ed, that tensor becomes backed by a Rust buffer. Updates to the tensor's data from
//! the Rust side will not reflect in ONNX Runtime until the tensor is `sync`ed with `SyncDirection::Runtime`. Likewise,
//! updates to the tensor's data from ONNX Runtime won't reflect in Rust until Rust syncs that tensor with
//! `SyncDirection::Rust`. You don't have to worry about this behavior if you only ever *read* from session outputs,
//! though.
//!
//! ## Limitations
//! - [`OutputSelector`](ort::session::run_options::OutputSelector) is not currently implemented.
//! - [`IoBinding`](ort::io_binding) is not supported by ONNX Runtime on the web.

#![deny(clippy::panic, clippy::panicking_unwrap)]
#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;
extern crate core;

use alloc::string::String;
use core::fmt;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::util::value_to_string;

mod api;
mod binding;
mod env;
mod memory;
mod session;
mod tensor;
mod util;
#[macro_use]
pub(crate) mod private;

pub use self::{
	session::sync_outputs,
	tensor::{SyncDirection, ValueExt}
};

pub type Result<T, E = Error> = core::result::Result<T, E>;

#[derive(Debug, Clone)]
pub struct Error {
	msg: String
}

impl Error {
	pub(crate) fn new(msg: impl Into<String>) -> Self {
		Self { msg: msg.into() }
	}
}

impl From<JsValue> for Error {
	fn from(value: JsValue) -> Self {
		Self::new(value_to_string(&value))
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		self.msg.fmt(f)
	}
}

impl core::error::Error for Error {}

/// Do not enable any execution provider features (CPU-only).
pub const FEATURE_NONE: u8 = 0;
/// Enable the WebGL execution provider for hardware acceleration.
///
/// See: <https://caniuse.com/webgl2>
pub const FEATURE_WEBGL: u8 = 1 << 0;
/// Enable the WebGPU execution provider for hardware acceleration.
///
/// See: <https://caniuse.com/webgpu>
pub const FEATURE_WEBGPU: u8 = 1 << 1;
/// Enable the WebNN execution provider for hardware acceleration.
///
/// See: <https://webmachinelearning.github.io/webnn-status/>
pub const FEATURE_WEBNN: u8 = FEATURE_WEBGPU;

/// Loads an `ort`-compatible ONNX Runtime API from `config`.
///
/// Returns an error if:
/// - The requested feature set is not supported by `ort-web`.
/// - The JavaScript/WASM modules fail to load.
///
/// `config` can be a feature set, in which case the default pyke-hosted builds will be used:
/// ```no_run
/// use ort::session::Session;
/// use ort_web::{FEATURE_WEBGL, FEATURE_WEBGPU};
///
/// async fn init_model() -> anyhow::Result<Session> {
/// 	// This must be called at least once before using any `ort` API.
/// 	ort::set_api(ort_web::api(FEATURE_WEBGL | FEATURE_WEBGPU).await?);
///
/// 	let session = Session::builder()?.commit_from_url("https://...").await?;
/// 	Ok(session)
/// }
/// ```
///
/// You can also use [`Dist`] to self-host the build:
/// ```no_run
/// use ort::session::Session;
/// use ort_web::Dist;
///
/// async fn init_model() -> anyhow::Result<Session> {
/// 	let dist = Dist::new("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/")
/// 		// load the WebGPU build
/// 		.with_script_name("ort.webgpu.min.js");
/// 	ort::set_api(ort_web::api(dist).await?);
/// }
/// ```
pub async fn api<L: Loadable>(config: L) -> Result<ort_sys::OrtApi> {
	let (features, dist) = config.into_features_and_dist()?;
	binding::init_runtime(features, dist).await?;

	Ok(self::api::api())
}

pub trait Loadable {
	#[doc(hidden)]
	fn into_features_and_dist(self) -> Result<(u8, JsValue)>;
}

impl Loadable for u8 {
	fn into_features_and_dist(self) -> Result<(u8, JsValue)> {
		Ok((self, JsValue::null()))
	}
}

impl Loadable for Dist {
	fn into_features_and_dist(self) -> Result<(u8, JsValue)> {
		Ok((0, serde_wasm_bindgen::to_value(&self).map_err(|e| Error::new(e.to_string()))?))
	}
}

#[derive(Default, Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Integrities {
	main: Option<String>,
	wrapper: Option<String>,
	binary: Option<String>
}

impl Integrities {
	/// Set the SHA-384 SRI hash for the main (entrypoint) script.
	pub fn set_main(&mut self, hash: impl Into<String>) {
		self.main = Some(hash.into());
	}

	/// Set the SHA-384 SRI hash for the Emscripten wrapper script.
	pub fn set_wrapper(&mut self, hash: impl Into<String>) {
		self.wrapper = Some(hash.into());
	}

	/// Set the SHA-384 SRI hash for the WASM binary.
	pub fn set_binary(&mut self, hash: impl Into<String>) {
		self.binary = Some(hash.into());
	}
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Dist {
	base_url: String,
	script_name: String,
	binary_name: Option<String>,
	wrapper_name: Option<String>,
	integrities: Integrities
}

impl Dist {
	pub fn new(base_url: impl Into<String>) -> Self {
		Self {
			base_url: base_url.into(),
			script_name: "ort.wasm.min.js".to_string(),
			binary_name: None,
			wrapper_name: None,
			integrities: Integrities::default()
		}
	}

	/// Configures the name of the entrypoint script file; defaults to `"ort.wasm.min.js"`.
	pub fn with_script_name(mut self, name: impl Into<String>) -> Self {
		self.script_name = name.into();
		self
	}

	/// Enables preloading the WASM binary loaded by the entrypoint script.
	pub fn with_binary_name(mut self, name: impl Into<String>) -> Self {
		self.binary_name = Some(name.into());
		self
	}

	/// Configures the name of the Emscripten wrapper script preloaded along with the WASM binary, if preloading is
	/// enabled. Defaults to the binary name with the `.wasm` extension replaced with `.mjs`.
	pub fn with_wrapper_name(mut self, name: impl Into<String>) -> Self {
		self.wrapper_name = Some(name.into());
		self
	}

	/// Modify Subresource Integrity (SRI) hashes.
	pub fn integrities(&mut self) -> &mut Integrities {
		&mut self.integrities
	}
}
