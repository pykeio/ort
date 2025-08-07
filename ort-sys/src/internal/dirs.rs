// based on https://github.com/dirs-dev/dirs-sys-rs/blob/main/src/lib.rs

#![allow(unused)]

pub const PYKE_ROOT: &str = "ort.pyke.io";

#[must_use]
pub fn cache_dir() -> Option<std::path::PathBuf> {
	Some(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join(PYKE_ROOT))
}
