//! `ort-web` is an [`ort`] backend that enables the usage of ONNX Runtime in the web.
//!
//! For more information, see https://ort.pyke.io/backends/web

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
/// 	let dist = Dist::new("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/")
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
