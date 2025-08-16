#![deny(clippy::panic, clippy::panicking_unwrap)]
#![warn(clippy::std_instead_of_alloc, clippy::std_instead_of_core)]

extern crate alloc;
extern crate core;

use alloc::string::String;
use core::fmt;

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

pub mod prelude {
	pub use crate::{
		session::sync_outputs,
		tensor::{SyncDirection, ValueExt}
	};
}

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

pub const FEATURE_NONE: u8 = 0;
pub const FEATURE_WEBGL: u8 = 1 << 0;
pub const FEATURE_WEBGPU: u8 = 1 << 1;
pub const FEATURE_WEBNN: u8 = FEATURE_WEBGPU;

pub async fn api(features: u8) -> Result<ort_sys::OrtApi> {
	binding::init_runtime(features).await?;
	Ok(self::api::api())
}
