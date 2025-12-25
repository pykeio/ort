use js_sys::Uint8Array;
use ort::session::SessionOutputs;
use ort_sys::{OrtErrorCode, stub::Error};

use crate::{
	binding,
	tensor::{SyncDirection, ValueExt},
	util::value_to_string
};

pub const SESSION_SENTINEL: [u8; 4] = [0xFC, 0x86, 0xA5, 0x01];

#[repr(C)]
pub struct Session {
	sentinel: [u8; 4],
	pub js: binding::InferenceSession,
	pub disable_sync: bool
}

impl Session {
	pub async fn from_url(uri: &str, options: &SessionOptions) -> Result<Self, Error> {
		Ok(Session {
			sentinel: SESSION_SENTINEL,
			js: binding::InferenceSession::create_from_uri(uri, &options.js)
				.await
				.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?,
			disable_sync: options.disable_sync
		})
	}

	pub async fn from_bytes(bytes: &[u8], options: &SessionOptions) -> Result<Self, Error> {
		Ok(Session {
			sentinel: SESSION_SENTINEL,
			js: binding::InferenceSession::create_from_bytes(
				// i'm fairly confident that the bytes are copied, at least when we're not using ONNX.js
				&unsafe { Uint8Array::view(bytes) },
				&options.js
			)
			.await
			.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?,
			disable_sync: options.disable_sync
		})
	}
}

pub struct RunOptions {}

impl RunOptions {
	pub const fn new() -> Self {
		RunOptions {}
	}
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
pub struct SessionOptions {
	pub js: binding::SessionOptions,
	pub disable_sync: bool
}

impl SessionOptions {
	pub fn new() -> Self {
		Self {
			js: binding::SessionOptions::default(),
			disable_sync: true
		}
	}
}
