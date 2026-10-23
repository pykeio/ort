use alloc::ffi::CString;
use std::collections::HashMap;

use js_sys::Uint8Array;
use ort::session::SessionOutputs;
use ort_sys::{OrtErrorCode, stub::Error};
use wasm_bindgen::JsCast;

use crate::{
	binding,
	tensor::{SyncDirection, ValueExt},
	util::value_to_string
};

pub(crate) const SESSION_SENTINEL: [u8; 4] = [0xFC, 0x86, 0xA5, 0x01];

#[repr(C)]
pub(crate) struct Session {
	sentinel: [u8; 4],
	pub(crate) js: binding::InferenceSession,
	pub(crate) input_names: Vec<String>,
	pub(crate) output_names: Vec<String>,
	pub(crate) disable_sync: bool
}

impl Session {
	fn new(js: binding::InferenceSession, options: &SessionOptions) -> Self {
		Session {
			sentinel: SESSION_SENTINEL,
			input_names: js.input_names(),
			output_names: js.output_names(),
			js,
			disable_sync: options.disable_sync
		}
	}

	pub(crate) async fn from_url(uri: &str, options: &SessionOptions) -> Result<Self, Error> {
		let js = binding::InferenceSession::create_from_uri(uri, &options.js)
			.await
			.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?;
		Ok(Self::new(js, options))
	}

	pub(crate) async fn from_bytes(bytes: &[u8], options: &SessionOptions) -> Result<Self, Error> {
		let js = binding::InferenceSession::create_from_bytes(
			// I'm fairly confident that the bytes are copied, at least when we're not using ONNX.js
			&unsafe { Uint8Array::view(bytes) },
			&options.js
		)
		.await
		.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?;
		Ok(Self::new(js, options))
	}
}

impl Drop for Session {
	fn drop(&mut self) {
		let js: binding::InferenceSession = self.js.clone().unchecked_into();
		wasm_bindgen_futures::spawn_local(async move {
			let _ = js.release().await;
		});
	}
}

#[derive(Default)]
pub(crate) struct RunOptions {
	pub(crate) js: binding::RunOptions,
	pub(crate) tag: CString,
	pub(crate) config: HashMap<String, CString>
}

/// Synchronize all outputs in `outputs` so that their data is available to Rust code.
///
/// See the [top-level documentation][crate] for more information on synchronization.
///
/// ```ignore
/// let mut outputs = session.run_async(ort::inputs![...]).await?;
/// ort_web::sync_outputs(&mut outputs).await?;
///
/// let bounding_boxes = outputs.remove("bounding_boxes").unwrap();
/// ...
/// ```
pub async fn sync_outputs(outputs: &mut SessionOutputs<'_>) -> crate::Result<()> {
	for (_, mut value) in outputs.iter_mut() {
		value.sync(SyncDirection::Rust).await?;
	}
	Ok(())
}

#[derive(Clone)]
pub(crate) struct SessionOptions {
	pub(crate) js: binding::SessionOptions,
	pub(crate) config: HashMap<String, CString>,
	pub(crate) disable_sync: bool
}

impl SessionOptions {
	pub(crate) fn new() -> Self {
		Self {
			js: binding::SessionOptions::default(),
			config: HashMap::new(),
			disable_sync: true
		}
	}
}
