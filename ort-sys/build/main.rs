use std::{env, path::PathBuf};

#[cfg(feature = "download-binaries")]
mod download;
mod dynamic_link;
#[cfg(feature = "download-binaries")]
mod error;
mod log;
#[cfg(feature = "pkg-config")]
mod pkg_config;
mod static_link;
mod vars;

#[path = "../src/internal/mod.rs"]
#[cfg(feature = "download-binaries")]
mod internal;

use crate::static_link::BinariesSource;

fn main() {
	println!("cargo:rustc-check-cfg=cfg(link_error)");

	if env::var("DOCS_RS").is_ok() || cfg!(feature = "disable-linking") {
		// On docs.rs, A) we don't need to link, and B) we don't have network, so we couldn't download anything if we wanted to.
		// If `disable-linking` is specified, either:
		// - `load-dynamic` is enabled, thus we don't need to link since we load the DLL at runtime, and we don't need to
		//   download anything because we don't provide DLLs anymore.
		// - The application intends to configure a custom backend. This build script only does ONNX Runtime, so no need to do
		//   anything.
		return;
	}

	#[cfg(feature = "pkg-config")]
	if self::pkg_config::attempt() {
		// linking with `pkg-config` was successful, no need to do anything
		return;
	}

	// Try to link to iOS frameworks first since this process is very different from normal static linking.
	if self::static_link::link_ios_frameworks() {
		self::static_link::static_link_prerequisites(BinariesSource::UserProvided);
		return;
	}

	if let Some(lib_dir) = vars::get(vars::SYSTEM_LIB_LOCATION) {
		let lib_dir = PathBuf::from(lib_dir);
		if dynamic_link::prefer_dynamic_linking() {
			println!("cargo:rustc-link-lib=onnxruntime");
			println!("cargo:rustc-link-search=native={}", lib_dir.display());
			return;
		}

		if !self::static_link::static_link(&lib_dir) {
			println!(
				"cargo::error=ort-sys could not link to the ONNX Runtime build in `{}`\ncargo::error= | rerun the build with `cargo build -vv | grep ort-sys` to see debug messages",
				lib_dir.display()
			);
		} else {
			self::static_link::static_link_prerequisites(BinariesSource::UserProvided);
		}

		return;
	}

	#[cfg(not(feature = "download-binaries"))]
	let should_skip = true;
	#[cfg(feature = "download-binaries")]
	let should_skip = self::download::should_skip();
	if should_skip {
		// Defer the error to the linking step so `cargo check` still works when `default-features = false`.
		println!("cargo:rustc-cfg=link_error");
		return;
	}

	log::debug!("Using prebuilt binaries");

	#[cfg(feature = "download-binaries")]
	{
		use std::fs;

		let target = env::var("TARGET").unwrap();

		let dist = match download::resolve_dist() {
			Ok(dist) => dist,
			Err(feature_set) => {
				println!(
					r"cargo::error=ort does not provide prebuilt binaries for the target `{}` with feature set {}.
cargo::error= | You may have to compile ONNX Runtime from source and link `ort` to your custom build; see https://ort.pyke.io/setup/linking
cargo::error= | Alternatively, try a different backend like `ort-tract`; see https://ort.pyke.io/backends",
					target,
					feature_set.unwrap_or_else(|| String::from("(no features)"))
				);
				return;
			}
		};

		let bin_extract_dir = internal::dirs::cache_dir()
			.expect("could not determine cache directory")
			.join("dfbin")
			.join(target)
			.join(dist.hash);
		if !bin_extract_dir.exists() {
			let mut verified_reader = match download::fetch_file(dist.url) {
				Ok(reader) => download::VerifyReader::new(reader),
				Err(e) => {
					println!(r"cargo::error=ort-sys failed to download prebuilt binaries from `{}`: {e}", dist.url);
					return;
				}
			};

			let mut temp_extract_dir = bin_extract_dir
				.parent()
				.unwrap()
				.join(format!("tmp.{}_{}", self::internal::random_identifier(), dist.hash));
			let mut should_rename = true;
			if fs::create_dir_all(&temp_extract_dir).is_err() {
				temp_extract_dir = env::var("OUT_DIR").unwrap().into();
				should_rename = false;
			}
			if let Err(e) = self::download::extract_tgz(&mut verified_reader, &temp_extract_dir) {
				println!(r"cargo::error=Extraction of prebuilt binaries downloaded from `{}` failed: {e}", dist.url);
				return;
			}

			let (calculated_hash, _) = verified_reader.finalize().expect("Failed to finalize read");
			if calculated_hash[..] != download::hex_str_to_bytes(dist.hash) {
				println!(
					r"cargo::error=⚠️ The hash of the file downloaded from `{}` does not match the expected hash. ⚠️
cargo::error= | Got {}, expected {}
cargo::error= | If you're using a proxy, make sure it's not doing something weird. Otherwise, report this incident to https://github.com/pykeio/ort/issues & email contact@pyke.io.
cargo::error= | The downloaded binaries are available to inspect at: {}",
					dist.url,
					download::bytes_to_hex_str(&calculated_hash),
					dist.hash,
					temp_extract_dir.display()
				);
				return;
			}

			if should_rename {
				match fs::rename(&temp_extract_dir, &bin_extract_dir) {
					Ok(()) => {}
					Err(e) => {
						if bin_extract_dir.exists() {
							let _ = fs::remove_dir_all(temp_extract_dir);
						} else {
							panic!("failed to extract downloaded binaries: {e}");
						}
					}
				}
			}
		}

		static_link::static_link_prerequisites(BinariesSource::Pyke);

		#[cfg(feature = "copy-dylibs")]
		dynamic_link::copy_dylibs(&bin_extract_dir, &std::path::PathBuf::from(env::var("OUT_DIR").unwrap()));

		println!("cargo:rustc-link-search=native={}", bin_extract_dir.display());
		println!("cargo:rustc-link-lib=static=onnxruntime");
	}
}
