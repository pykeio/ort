use pkg_config::Config;

use crate::log;

#[path = "../src/version.rs"]
mod version;
use self::version::ORT_API_VERSION;

pub fn attempt() -> bool {
	match Config::new().probe("libonnxruntime") {
		Ok(lib) => {
			let got_minor = lib.version.split('.').nth(1).unwrap().parse::<usize>().unwrap();
			if got_minor < ORT_API_VERSION as _ {
				log::warning!(
					"libonnxruntime provided by `pkg-config` is out of date, so it will be ignored. Minor version was {} but ort expects {}",
					lib.version,
					ORT_API_VERSION
				);
				return false;
			}

			// Setting the link paths
			for path in lib.link_paths {
				println!("cargo:rustc-link-search=native={}", path.display());
			}

			// Setting the libraries to link against
			for lib in lib.libs {
				println!("cargo:rustc-link-lib={}", lib);
			}

			log::debug!("Using `libonnxruntime` as configured by `pkg-config`.");
			true
		}
		Err(_) => {
			log::debug!("`libonnxruntime` is not configured in `pkg-config`; will fall back to default linking routine.");
			false
		}
	}
}
