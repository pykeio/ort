use std::{
	env,
	path::{Path, PathBuf}
};

const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_ENV_SYSTEM_LIB_PROFILE: &str = "ORT_LIB_PROFILE";
#[cfg(feature = "download-binaries")]
const ORT_EXTRACT_DIR: &str = "onnxruntime";

macro_rules! incompatible_providers {
	($($provider:ident),*) => {
		$(
			if env::var(concat!("CARGO_FEATURE_", stringify!($provider))).is_ok() {
				println!(concat!("cargo:warning=Provider not available for this strategy and/or target: ", stringify!($provider)));
			}
		)*
	}
}

#[cfg(feature = "download-binaries")]
fn fetch_file(source_url: &str) -> Vec<u8> {
	let resp = ureq::get(source_url)
		.timeout(std::time::Duration::from_secs(1800))
		.call()
		.unwrap_or_else(|err| panic!("[ort] failed to download {source_url}: {err:?}"));

	let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
	let mut reader = resp.into_reader();
	let mut buffer = Vec::new();
	reader.read_to_end(&mut buffer).unwrap();
	assert_eq!(buffer.len(), len);
	buffer
}

#[cfg(feature = "download-binaries")]
fn hex_str_to_bytes(c: impl AsRef<[u8]>) -> Vec<u8> {
	fn nibble(c: u8) -> u8 {
		match c {
			b'A'..=b'F' => c - b'A' + 10,
			b'a'..=b'f' => c - b'a' + 10,
			b'0'..=b'9' => c - b'0',
			_ => panic!()
		}
	}

	c.as_ref().chunks(2).map(|n| nibble(n[0]) << 4 | nibble(n[1])).collect()
}

#[cfg(feature = "download-binaries")]
fn verify_file(buf: &[u8], hash: impl AsRef<[u8]>) -> bool {
	use sha2::Digest;
	sha2::Sha256::digest(buf)[..] == hex_str_to_bytes(hash)
}

#[cfg(feature = "download-binaries")]
fn extract_tgz(buf: &[u8], output: &Path) {
	let buf: std::io::BufReader<&[u8]> = std::io::BufReader::new(buf);
	let tar = flate2::read::GzDecoder::new(buf);
	let mut archive = tar::Archive::new(tar);
	archive.unpack(output).unwrap();
}

#[cfg(feature = "copy-dylibs")]
fn copy_libraries(lib_dir: &Path, out_dir: &Path) {
	// get the target directory - we need to place the dlls next to the executable so they can be properly loaded by windows
	let out_dir = out_dir.ancestors().nth(3).unwrap();
	for out_dir in [out_dir.to_path_buf(), out_dir.join("examples"), out_dir.join("deps")] {
		#[cfg(windows)]
		let mut copy_fallback = false;

		let lib_files = std::fs::read_dir(lib_dir).unwrap();
		for lib_file in lib_files.filter(|e| {
			e.as_ref().ok().map_or(false, |e| {
				e.file_type().map_or(false, |e| !e.is_dir())
					&& [".dll", ".so", ".dylib"]
						.into_iter()
						.any(|v| e.path().into_os_string().into_string().unwrap().contains(v))
			})
		}) {
			let lib_file = lib_file.unwrap();
			let lib_path = lib_file.path();
			let lib_name = lib_path.file_name().unwrap();
			let out_path = out_dir.join(lib_name);
			if !out_path.exists() {
				#[cfg(windows)]
				if std::os::windows::fs::symlink_file(&lib_path, &out_path).is_err() {
					copy_fallback = true;
					std::fs::copy(&lib_path, out_path).unwrap();
				}
				#[cfg(unix)]
				std::os::unix::fs::symlink(&lib_path, out_path).unwrap();
			}
		}

		// If we had to fallback to copying files on Windows, break early to avoid copying to 3 different directories
		#[cfg(windows)]
		if copy_fallback {
			break;
		}
	}
}

fn add_search_dir<P: AsRef<Path>>(base: P) {
	let base = base.as_ref();
	if base.join("Release").is_dir() {
		println!("cargo:rustc-link-search=native={}", base.join("Release").display());
	} else if base.join("Debug").is_dir() {
		println!("cargo:rustc-link-search=native={}", base.join("Debug").display());
	} else {
		println!("cargo:rustc-link-search=native={}", base.display());
	}
}

fn static_link_prerequisites() {
	let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
	if target_os == "macos" || target_os == "ios" {
		println!("cargo:rustc-link-lib=c++");
		println!("cargo:rustc-link-lib=framework=Foundation");
	} else if target_os == "linux" || target_os == "android" {
		println!("cargo:rustc-link-lib=stdc++");
	} else if target_os == "windows" {
		println!("cargo:rustc-link-lib=dxguid");
		println!("cargo:rustc-link-lib=DirectML");
	}
}

fn system_strategy() -> (PathBuf, bool) {
	let lib_dir = PathBuf::from(env::var(ORT_ENV_SYSTEM_LIB_LOCATION).expect("[ort] system strategy requires ORT_LIB_LOCATION env var to be set"));

	let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap().to_lowercase();
	let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap().to_lowercase();
	let platform_format_lib = |a: &str| {
		if target_os.contains("windows") { format!("{}.lib", a) } else { format!("lib{}.a", a) }
	};

	let mut profile = env::var(ORT_ENV_SYSTEM_LIB_PROFILE).unwrap_or_default();
	if profile.is_empty() {
		for i in ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"] {
			if lib_dir.join(i).exists() && lib_dir.join(i).join(platform_format_lib("onnxruntime_common")).exists() {
				profile = String::from(i);
				break;
			}
		}
	}

	add_search_dir(&lib_dir);

	let mut needs_link = true;
	if lib_dir.join(platform_format_lib("onnxruntime")).exists() {
		println!("cargo:rustc-link-lib=static=onnxruntime");
		needs_link = false;
	} else {
		#[allow(clippy::type_complexity)]
		let static_configs: Vec<(PathBuf, PathBuf, PathBuf, Box<dyn Fn(PathBuf, &String) -> PathBuf>)> = vec![
			(lib_dir.join(&profile), lib_dir.join("lib"), lib_dir.join("_deps"), Box::new(|p: PathBuf, profile| p.join(profile))),
			(lib_dir.clone(), lib_dir.join("lib"), lib_dir.parent().unwrap().join("_deps"), Box::new(|p: PathBuf, _| p)),
			(lib_dir.join("onnxruntime"), lib_dir.join("onnxruntime").join("lib"), lib_dir.join("_deps"), Box::new(|p: PathBuf, _| p)),
		];
		for (lib_dir, extension_lib_dir, external_lib_dir, transform_dep) in static_configs {
			if lib_dir.join(platform_format_lib("onnxruntime_common")).exists() {
				add_search_dir(&lib_dir);

				for lib in &["common", "flatbuffers", "framework", "graph", "mlas", "optimizer", "providers", "session", "util"] {
					let lib_path = lib_dir.join(platform_format_lib(&format!("onnxruntime_{lib}")));
					// sanity check, just make sure the library exists before we try to link to it
					if lib_path.exists() {
						println!("cargo:rustc-link-lib=static=onnxruntime_{lib}");
					} else {
						panic!("[ort] unable to find ONNX Runtime library: {}", lib_path.display());
					}
				}

				if extension_lib_dir.exists() && extension_lib_dir.join(platform_format_lib("ortcustomops")).exists() {
					add_search_dir(&extension_lib_dir);
					println!("cargo:rustc-link-lib=static=ortcustomops");
					println!("cargo:rustc-link-lib=static=ocos_operators");
					println!("cargo:rustc-link-lib=static=noexcep_operators");
				}

				if target_arch == "wasm32" {
					for lib in &["webassembly", "providers_js"] {
						let lib_path = lib_dir.join(platform_format_lib(&format!("onnxruntime_{lib}")));
						if lib_path.exists() {
							println!("cargo:rustc-link-lib=static=onnxruntime_{lib}");
						}
					}
				}

				let protobuf_build = transform_dep(external_lib_dir.join("protobuf-build"), &profile);
				add_search_dir(&protobuf_build);
				for lib in ["protobuf-lited", "protobuf-lite", "protobuf"] {
					if target_os.contains("windows") && protobuf_build.join(platform_format_lib(&format!("lib{lib}"))).exists() {
						println!("cargo:rustc-link-lib=static=lib{lib}")
					} else if protobuf_build.join(platform_format_lib(lib)).exists() {
						println!("cargo:rustc-link-lib=static={lib}");
					}
				}

				add_search_dir(transform_dep(external_lib_dir.join("onnx-build"), &profile));
				println!("cargo:rustc-link-lib=static=onnx");
				println!("cargo:rustc-link-lib=static=onnx_proto");

				add_search_dir(transform_dep(external_lib_dir.join("google_nsync-build"), &profile));
				println!("cargo:rustc-link-lib=static=nsync_cpp");

				if target_arch != "wasm32" {
					add_search_dir(transform_dep(external_lib_dir.join("pytorch_cpuinfo-build"), &profile));
					add_search_dir(transform_dep(external_lib_dir.join("pytorch_cpuinfo-build").join("deps").join("clog"), &profile));
					println!("cargo:rustc-link-lib=static=cpuinfo");
					println!("cargo:rustc-link-lib=static=clog");
				}

				add_search_dir(transform_dep(external_lib_dir.join("re2-build"), &profile));
				println!("cargo:rustc-link-lib=static=re2");

				add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("base"), &profile));
				println!("cargo:rustc-link-lib=static=absl_base");
				println!("cargo:rustc-link-lib=static=absl_throw_delegate");
				add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("hash"), &profile));
				println!("cargo:rustc-link-lib=static=absl_hash");
				println!("cargo:rustc-link-lib=static=absl_city");
				println!("cargo:rustc-link-lib=static=absl_low_level_hash");
				add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("container"), &profile));
				println!("cargo:rustc-link-lib=static=absl_raw_hash_set");

				// #[cfg(feature = "rocm")]
				// println!("cargo:rustc-link-lib=onnxruntime_providers_rocm");

				needs_link = false;
				break;
			}
		}
		if needs_link {
			// none of the static link patterns matched, we might be trying to dynamic link so copy dylibs if requested
			#[cfg(feature = "copy-dylibs")]
			{
				let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
				if lib_dir.join("lib").is_dir() {
					copy_libraries(&lib_dir.join("lib"), &out_dir);
				} else if lib_dir.join(&profile).is_dir() {
					copy_libraries(&lib_dir.join(profile), &out_dir);
				}
			}
		}
	}

	(lib_dir, needs_link)
}

fn prepare_libort_dir() -> (PathBuf, bool) {
	let strategy = env::var(ORT_ENV_STRATEGY);
	println!("[ort] strategy: {:?}", strategy.as_ref().map(String::as_str).unwrap_or_else(|_| "unknown"));

	let target = env::var("TARGET").unwrap();
	let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
	if target_arch.eq_ignore_ascii_case("aarch64") {
		incompatible_providers![CUDA, OPENVINO, VITIS, TENSORRT, MIGRAPHX, ROCM];
	} else if target_arch.eq_ignore_ascii_case("x86_64") {
		incompatible_providers![ACL, ARMNN, CANN, RKNPU];
	}

	if target.contains("darwin") {
		incompatible_providers![CUDA, OPENVINO, VITIS, ACL, ARMNN, TENSORRT, WINML, CANN];
	} else if target.contains("windows") {
		incompatible_providers![COREML, VITIS, ACL, ARMNN, CANN];
	} else {
		incompatible_providers![COREML, DIRECTML, WINML];
	}

	println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);

	match strategy.as_ref().map_or("download", String::as_str) {
		#[cfg(feature = "download-binaries")]
		"download" => {
			if target.contains("darwin") {
				incompatible_providers![CUDA, ONEDNN, OPENVINO, VITIS, TVM, TENSORRT, MIGRAPHX, DIRECTML, WINML, ACML, ARMNN, ROCM];
			} else if target.contains("windows") {
				incompatible_providers![ONEDNN, OPENVINO, VITIS, TVM, MIGRAPHX, WINML, ACML, ARMNN, ROCM];
			} else {
				incompatible_providers![ONEDNN, OPENVINO, VITIS, TVM, MIGRAPHX, DIRECTML, WINML, ACML, ARMNN, ROCM];
			}

			let (prebuilt_url, prebuilt_hash) = match env::var("TARGET").unwrap().as_str() {
				"aarch64-apple-darwin" => (
					"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-aarch64-apple-darwin.tgz",
					"142644EB9A95E01C609873EEB6C581C1FB2C0699B60300C44A9DCB4BA2A75975"
				),
				"aarch64-pc-windows-msvc" => (
					"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-aarch64-pc-windows-msvc.tgz",
					"5730A8C23C1C1C3D62AAC24FD6F5496A2897FC4967B36F1CC5308519C332399E"
				),
				"aarch64-unknown-linux-gnu" => (
					"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-aarch64-unknown-linux-gnu.tgz",
					"A05AA5CB2FF4F9471ABFFA295F3F65423FF55A560727386E12A6BE96950A8E5B"
				),
				"wasm32-unknown-emscripten" => (
					"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-wasm32-unknown-emscripten.tgz",
					"51CB0F74A422501E810483E67CEFD74765C1164AD5D77C90527AB2697ADA48B4"
				),
				"x86_64-apple-darwin" => (
					"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-x86_64-apple-darwin.tgz",
					"F3B58C9AAA1DE2184BF58B5F928A26660FD8F26232BE3F8A2C6A61896D7FA23A"
				),
				"x86_64-pc-windows-msvc" => {
					if cfg!(any(feature = "cuda", feature = "tensorrt")) {
						(
							"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_dylib_cuda-v1.16.2-x86_64-pc-windows-msvc.tgz",
							"6C12CA3FDF4CF06D11AE4E8DE8E7243BEE259FB9998B8386EE46801BBC437D3D"
						)
					} else {
						(
							"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-x86_64-pc-windows-msvc.tgz",
							"B9EC5CC19032825645F8A2F34E66397C2D8E05DB87472EB3B293882178B043D6"
						)
					}
				}
				"x86_64-unknown-linux-gnu" => {
					if cfg!(any(feature = "cuda", feature = "tensorrt")) {
						(
							"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_dylib_cuda-v1.16.2-x86_64-unknown-linux-gnu.tgz",
							"6083247B508585B871264DDF3C44DB298A077D1CB092E3270EFA8B411288086A"
						)
					} else {
						(
							"https://parcel.pyke.io/v2/delivery/ortrs/packages/msort-binary/1.16.2+001/ortrs-msort_static-v1.16.2-x86_64-unknown-linux-gnu.tgz",
							"93F77F7BF5B95C2430BDBC063E830878328CA13CC8466AADA9F7EA5CE35F366A"
						)
					}
				}
				x => panic!("downloaded binaries not available for target {x}\nyou may have to compile ONNX Runtime from source")
			};

			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
			let lib_dir = out_dir.join(ORT_EXTRACT_DIR);
			if !lib_dir.exists() {
				let downloaded_file = fetch_file(prebuilt_url);
				assert!(verify_file(&downloaded_file, prebuilt_hash));
				extract_tgz(&downloaded_file, &out_dir);
			}

			static_link_prerequisites();

			#[cfg(feature = "copy-dylibs")]
			{
				copy_libraries(&lib_dir.join("lib"), &out_dir);
			}

			(lib_dir, true)
		}
		#[cfg(not(feature = "download-binaries"))]
		"download" => {
			if env::var(ORT_ENV_SYSTEM_LIB_LOCATION).is_ok() {
				return system_strategy();
			}

			println!("cargo:rustc-link-lib=add_ort_library_path_or_enable_feature_download-binaries_see_ort_docs");
			(PathBuf::default(), false)
		}
		"system" => system_strategy(),
		_ => panic!("[ort] unknown strategy: {} (valid options are `download` or `system`)", strategy.unwrap_or_else(|_| "unknown".to_string()))
	}
}

fn real_main(link: bool) {
	let (install_dir, needs_link) = prepare_libort_dir();

	let lib_dir = if install_dir.join("lib").exists() { install_dir.join("lib") } else { install_dir };

	if link {
		if needs_link {
			println!("cargo:rustc-link-lib=onnxruntime");
			println!("cargo:rustc-link-search=native={}", lib_dir.display());
		}

		static_link_prerequisites();

		println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);
	}

	println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
}

fn main() {
	if env::var("DOCS_RS").is_ok() {
		return;
	}

	if cfg!(feature = "load-dynamic") {
		// we only need to execute the real main step if we are using the download strategy...
		if cfg!(feature = "download-binaries") && std::env::var(ORT_ENV_STRATEGY).as_ref().map_or("download", String::as_str) == "download" {
			// but we don't need to link to the binaries we download (so all we are doing is downloading them and placing them in
			// the output directory)
			real_main(false);
		}
	} else {
		// if we are not using the load-dynamic feature then we need to link to dylibs.
		real_main(true);
	}
}
