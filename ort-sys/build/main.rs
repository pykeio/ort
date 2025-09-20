use std::{
	env,
	fmt::{self, Display},
	path::PathBuf
};

#[cfg(feature = "download-binaries")]
mod download;
mod dynamic_link;
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
				"cargo:error=ort-sys could not link to the ONNX Runtime build in `{}`\nrerun the build with `cargo build -vv | grep ort-sys` to see debug messages",
				lib_dir.display()
			);
		}
		return;
	}

	#[cfg(not(feature = "download-binaries"))]
	let should_skip = true;
	#[cfg(feature = "download-binaries")]
	let should_skip = self::download::should_skip();
	if should_skip {
		println!(
			r#"cargo:error=ort-sys could not link to ONNX Runtime because:
- `libonnxruntime` is not configured via `pkg-config`
- {}
- Neither `{}` or `{}` were set to link to custom binaries

To rectify this:
- Compile ONNX Runtime from source and manually configure linking (see https://ort.pyke.io/setup/linking for more information)
- Enable the `download-binaries` feature if the target is supported
- Enable `ort`'s `alternative-backend` feature if you intend to use a different backend (or `ort-sys`'s `disable-linking` feature if you use this crate directly)"#,
			if cfg!(feature = "download-binaries") {
				"ort-sys was instructed not to download prebuilt binaries (`cargo build --offline`)"
			} else {
				"The `download-binaries` feature is not enabled, so prebuilt binaries can't be fetched"
			},
			vars::SYSTEM_LIB_LOCATION,
			vars::IOS_ONNX_XCFWK_LOCATION
		);
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
					r"cargo:error=ort does not provide prebuilt binaries for the target `{}` with feature set {}.
You may have to compile ONNX Runtime from source and link `ort` to your custom build; see https://ort.pyke.io/setup/linking
Alternatively, try a different backend like `ort-tract`; see https://ort.pyke.io/backends",
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

		let lib_dir = bin_extract_dir.join("onnxruntime").join("lib");
		if !lib_dir.exists() {
			let mut verified_reader = match download::fetch_file(dist.url) {
				Ok(reader) => download::VerifyReader::new(reader),
				Err(e) => {
					println!(r"cargo:error=ort-sys failed to download prebuilt binaries from `{}`: {e}", dist.url);
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
				println!(r"cargo:error=Extraction of prebuilt binaries downloaded from `{}` failed: {e}", dist.url);
				return;
			}

			let (calculated_hash, _) = verified_reader.finalize();
			if calculated_hash[..] != download::hex_str_to_bytes(dist.hash) {
				println!(
					r"cargo:error=⚠️ The hash of the file downloaded from `{}` does not match the expected hash. ⚠️
Got {}, expected {}
If you're using a proxy, make sure it's not doing something weird. Otherwise, report this incident to https://github.com/pykeio/ort/issues & email contact@pyke.io.
The downloaded binaries are available to inspect at: {}",
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
		dynamic_link::copy_dylibs(&lib_dir, &std::path::PathBuf::from(env::var("OUT_DIR").unwrap()));

		println!("cargo:rustc-link-search=native={}", lib_dir.display());
		println!("cargo:rustc-link-lib=static=onnxruntime");
	}
}

#[derive(Debug)]
struct Error {
	message: String
}

impl Error {
	pub fn new(message: impl Into<String>) -> Self {
		Self { message: message.into() }
	}
}

impl Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.message)
	}
}

impl<E: std::error::Error> From<E> for Error {
	fn from(value: E) -> Self {
		Self { message: value.to_string() }
	}
}

trait ResultExt<T, E> {
	fn with_context<S: fmt::Display, F: FnOnce() -> S>(self, ctx: F) -> Result<T, Error>;
}

impl<T, E: fmt::Display> ResultExt<T, E> for Result<T, E> {
	fn with_context<S: fmt::Display, F: FnOnce() -> S>(self, ctx: F) -> Result<T, Error> {
		self.map_err(|e| Error::new(format!("{}: {}", ctx(), e)))
	}
}
