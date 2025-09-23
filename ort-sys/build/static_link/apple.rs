use std::{env, path::Path, process::Command};

use crate::{
	log,
	vars::{self}
};

pub fn macos_rtlib_search_dir() -> Option<String> {
	let output = Command::new(vars::get("CC").unwrap_or_else(|| "clang".to_string()))
		.arg("--print-search-dirs")
		.output()
		.ok()?;
	if !output.status.success() {
		log::warning!("couldn't determine macOS rtlib dir: failed to run `$CC --print-search-dirs` (exit code {:?})", output.status.code());
		return None;
	}

	let stdout = String::from_utf8_lossy(&output.stdout);
	for line in stdout.lines() {
		if line.contains("libraries: =") {
			let path = line.split('=').nth(1)?;
			if !path.is_empty() {
				return Some(format!("{path}/lib/darwin"));
			}
		}
	}

	log::warning!("couldn't determine macOS rtlib dir: invalid output");

	None
}

pub fn ios_rtlib_search_dir() -> Option<String> {
	let output = Command::new("xcrun").args(["clang", "--print-resource-dir"]).output().ok()?;
	if !output.status.success() {
		log::warning!("couldn't determine iOS rtlib dir: failed to run `xcrun clang --print-resource-dir` (exit code {:?})", output.status.code());
		return None;
	}

	let resource_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
	Some(format!("{}/lib/darwin", resource_dir))
}

fn search_and_link_frameworks_in_sub_dir(sub_dir: &str) -> bool {
	let Some(xcfwk_dir) = vars::get(vars::IOS_ONNX_XCFWK_LOCATION) else {
		return false;
	};

	let fwk_dir = Path::new(&xcfwk_dir).join(sub_dir);
	if !fwk_dir.exists() {
		log::warning!("framework directory '{}' does not exist", fwk_dir.display());
		// Framework directory not found, dont add search path at all
		return false;
	}
	println!("cargo:rustc-link-search=framework={}", fwk_dir.display());

	if fwk_dir.join("onnxruntime.framework").exists() {
		println!("cargo:rustc-link-lib=framework=onnxruntime");
	} else {
		log::warning!("onnxruntime.framework not found in '{}'", fwk_dir.display());
		// Framework not found, skip attempting extension framework
		return false;
	}

	log::debug!("successfully linked framework from {}", fwk_dir.display());

	let Some(ext_xcfwk_dir) = vars::get(vars::IOS_ONNX_EXT_XCFWK_LOCATION) else {
		// If ext is not set, skip linking
		return true;
	};

	let ext_fwk_dir = Path::new(&ext_xcfwk_dir).join(sub_dir);
	if !ext_fwk_dir.exists() {
		// Extension framework directory not found, dont add search path at all
		return true;
	}
	println!("cargo:rustc-link-search=framework={}", ext_fwk_dir.display());

	// Link extensions framework if found
	if ext_fwk_dir.join("onnxruntime_extensions.framework").exists() {
		println!("cargo:rustc-link-lib=framework=onnxruntime_extensions");
	}

	true
}

pub fn link_ios_frameworks() -> bool {
	let Ok(target) = env::var("TARGET") else {
		return false;
	};

	// XCFramework for onnxruntime only has support for ios, ios-sim and macos.
	// Here we only care about ios target triples
	match &*target {
		"aarch64-apple-ios" => search_and_link_frameworks_in_sub_dir("ios-arm64"),
		"aarch64-apple-ios-sim" => {
			// Legacy xcode builds will have ios-arm64_x86_64-simulator and newer ones will use ios-arm64-simulator
			search_and_link_frameworks_in_sub_dir("ios-arm64_x86_64-simulator") || search_and_link_frameworks_in_sub_dir("ios-arm64-simulator")
		}
		_ => {
			if target.contains("apple") {
				log::warning!("can't do xcframework linking for target '{target}'");
			}
			false
		}
	}
}
