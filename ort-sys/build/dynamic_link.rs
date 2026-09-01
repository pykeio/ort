#[cfg(feature = "copy-dylibs")]
use std::path::Path;

use crate::vars;

pub fn prefer_dynamic_linking() -> bool {
	match vars::get(vars::PREFER_DYNAMIC_LINK) {
		Some(val) => val == "1" || val.to_lowercase() == "true",
		None => false
	}
}

#[cfg(feature = "copy-dylibs")]
pub fn copy_dylibs(lib_dir: &Path) {
	use std::fs;

	// we need to place the dlls next to the executable so they can be properly loaded by windows
	let target_dir = vars::target_dir();
	for out_dir in [target_dir.to_path_buf(), target_dir.join("examples"), target_dir.join("deps")] {
		#[cfg(windows)]
		let mut copy_fallback = false;
		#[cfg(not(windows))]
		let copy_fallback = false;

		let lib_files = fs::read_dir(lib_dir).unwrap_or_else(|_| panic!("Failed to read contents of `{}` (does it exist?)", lib_dir.display()));
		for lib_file in lib_files.filter(|e| {
			e.as_ref().ok().is_some_and(|e| {
				e.file_type().is_ok_and(|e| !e.is_dir()) && [".dll", ".so", ".dylib"].into_iter().any(|v| e.path().to_string_lossy().contains(v))
			})
		}) {
			let lib_file = lib_file.unwrap();
			let lib_path = lib_file.path();
			let lib_name = lib_path.file_name().unwrap();
			let out_path = out_dir.join(lib_name);
			if out_path.is_symlink() {
				fs::remove_file(&out_path).unwrap();
			}

			// recent rust versions no longer use the deps folder
			if out_dir.exists() && !out_path.exists() {
				#[cfg(windows)]
				if std::os::windows::fs::symlink_file(&lib_path, &out_path).is_err() {
					copy_fallback = true;
					fs::copy(&lib_path, &out_path).unwrap();
				}
				#[cfg(unix)]
				std::os::unix::fs::symlink(&lib_path, &out_path).unwrap();
			}
			if !copy_fallback {
				println!("cargo:rerun-if-changed={}", out_path.to_str().unwrap());
			}
		}

		// If we had to fallback to copying files on Windows, break early to avoid copying to 3 different directories
		if copy_fallback {
			crate::log::warning!(
				"had to copy dylibs because Windows Developer Mode is not enabled, or the cache dir is on a different drive. examples & tests will not be able to access the dylibs"
			);
			break;
		}
	}
}
