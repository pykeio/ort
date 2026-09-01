#![allow(dead_code)]

use std::{env, path::PathBuf};

pub const SYSTEM_LIB_PATH: &[&str] = &["ORT_LIB_PATH", "ORT_LIB_LOCATION"];
pub const SYSTEM_LIB_PROFILE: &str = "ORT_LIB_PROFILE";
pub const VCPKG_TARGET: &str = "ORT_VCPKG_TARGET";
pub const IOS_ONNX_XCFWK_PATH: &[&str] = &["ORT_IOS_XCFWK_PATH", "ORT_IOS_XCFWK_LOCATION"];
pub const IOS_ONNX_EXT_XCFWK_PATH: &[&str] = &["ORT_EXT_IOS_XCFWK_PATH", "ORT_EXT_IOS_XCFWK_LOCATION"];
pub const PREFER_DYNAMIC_LINK: &str = "ORT_PREFER_DYNAMIC_LINK";
pub const SKIP_DOWNLOAD: &[&str] = &["CARGO_NET_OFFLINE", "ORT_SKIP_DOWNLOAD", "ORT_OFFLINE"];
pub const CXX_STDLIB: &[&str] = &[
	"ORT_CXX_STDLIB",
	"CXXSTDLIB" // Used by the `cc` crate - we should mirror if this is set for other C++ crates
];
pub const CUDA_VERSION: &str = "ORT_CUDA_VERSION";

pub fn get(var: &str) -> Option<String> {
	println!("cargo:rerun-if-env-changed={var}");
	env::var(var).ok()
}

pub fn get_any(vars: &[&str]) -> Option<String> {
	for var in vars {
		if let Some(r) = get(var) {
			return Some(r);
		}
	}
	None
}

pub fn target_dir() -> PathBuf {
	let out_dir = std::path::PathBuf::from(env::var("OUT_DIR").unwrap());
	// more recent rust versions have this set to `target/<profile>/build/ort-sys/<fingerprint>/out`
	// earlier, this was `target/<profile>/build/ort-sys-<fingerprint>/out`
	let mut ancestors = out_dir.ancestors().skip(2);
	let target_dir = if ancestors.next().is_some_and(|x| x.ends_with("ort-sys")) {
		ancestors.nth(1)
	} else {
		ancestors.next()
	};
	target_dir.expect("").to_path_buf()
}
