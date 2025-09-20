#![allow(dead_code)]

pub const SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
pub const SYSTEM_LIB_PROFILE: &str = "ORT_LIB_PROFILE";
pub const IOS_ONNX_XCFWK_LOCATION: &str = "ORT_IOS_XCFWK_LOCATION";
pub const IOS_ONNX_EXT_XCFWK_LOCATION: &str = "ORT_EXT_IOS_XCFWK_LOCATION";
pub const PREFER_DYNAMIC_LINK: &str = "ORT_PREFER_DYNAMIC_LINK";
pub const SKIP_DOWNLOAD: &str = "ORT_SKIP_DOWNLOAD";
pub const CXX_STDLIB: &str = "ORT_CXX_STDLIB";
pub const CXX_STDLIB_ALT: &str = "CXXSTDLIB"; // Used by the `cc` crate - we should mirror if this is set for other C++ crates

pub fn get(var: &str) -> Option<String> {
	println!("cargo:rerun-if-env-changed={var}");
	std::env::var(var).ok()
}
