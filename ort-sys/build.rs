use std::{
	env,
	path::{Path, PathBuf},
	process::Command
};

#[allow(unused)]
const ONNXRUNTIME_VERSION: &str = "1.22.0";

const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_ENV_SYSTEM_LIB_PROFILE: &str = "ORT_LIB_PROFILE";
const ORT_ENV_PREFER_DYNAMIC_LINK: &str = "ORT_PREFER_DYNAMIC_LINK";
const ORT_ENV_SKIP_DOWNLOAD: &str = "ORT_SKIP_DOWNLOAD";
const ORT_ENV_CXX_STDLIB: &str = "ORT_CXX_STDLIB";
const ENV_CXXSTDLIB: &str = "CXXSTDLIB"; // Used by the `cc` crate - we should mirror if this is set for other C++ crates
#[cfg(feature = "download-binaries")]
const ORT_EXTRACT_DIR: &str = "onnxruntime";

#[cfg(feature = "download-binaries")]
const DIST_TABLE: &str = include_str!("dist.txt");

#[path = "src/internal/mod.rs"]
#[cfg(feature = "download-binaries")]
mod internal;
#[cfg(feature = "download-binaries")]
use self::internal::dirs::cache_dir;

#[cfg(feature = "download-binaries")]
fn fetch_file(source_url: &str) -> Vec<u8> {
	let resp = ureq::Agent::new_with_config(
		ureq::config::Config::builder()
			.proxy(ureq::Proxy::try_from_env())
			.max_redirects(0)
			.https_only(true)
			.tls_config(
				ureq::tls::TlsConfig::builder()
					.provider(ureq::tls::TlsProvider::NativeTls)
					.root_certs(ureq::tls::RootCerts::PlatformVerifier)
					.build()
			)
			.user_agent(format!(
				"{}/{} (host {}; for {})",
				env!("CARGO_PKG_NAME"),
				env!("CARGO_PKG_VERSION"),
				std::env::var("HOST").unwrap(),
				std::env::var("TARGET").unwrap()
			))
			.timeout_global(Some(std::time::Duration::from_secs(1800)))
			.build()
	)
	.get(source_url)
	.call()
	.unwrap_or_else(|err| panic!("Failed to GET `{source_url}`: {err}"));

	resp.into_body()
		.into_with_config()
		.limit(1_073_741_824)
		.read_to_vec()
		.unwrap_or_else(|err| panic!("Failed to download from `{source_url}`: {err}"))
}

#[cfg(feature = "download-binaries")]
fn find_dist(target: &str, feature_set: &str) -> Option<(&'static str, &'static str)> {
	DIST_TABLE
		.split('\n')
		.filter(|c| !c.is_empty() && !c.starts_with('#'))
		.map(|c| c.split('\t').collect::<Vec<_>>())
		.find(|c| c[0] == feature_set && c[1] == target)
		.map(|c| (c[2], c[3]))
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

	c.as_ref().chunks(2).map(|n| (nibble(n[0]) << 4) | nibble(n[1])).collect()
}

#[cfg(feature = "download-binaries")]
fn verify_file(buf: &[u8], hash: impl AsRef<[u8]>) -> bool {
	<sha2::Sha256 as sha2::Digest>::digest(buf)[..] == hex_str_to_bytes(hash)
}

#[cfg(feature = "download-binaries")]
fn extract_tgz(buf: &[u8], output: &Path) {
	let buf: std::io::BufReader<&[u8]> = std::io::BufReader::new(buf);
	let tar = flate2::read::GzDecoder::new(buf);
	let mut archive = tar::Archive::new(tar);
	archive.unpack(output).expect("Failed to extract .tgz file");
}

#[cfg(feature = "copy-dylibs")]
fn copy_libraries(lib_dir: &Path, out_dir: &Path) {
	// get the target directory - we need to place the dlls next to the executable so they can be properly loaded by windows
	let out_dir = out_dir.ancestors().nth(3).unwrap();
	for out_dir in [out_dir.to_path_buf(), out_dir.join("examples"), out_dir.join("deps")] {
		#[cfg(windows)]
		let mut copy_fallback = false;
		#[cfg(not(windows))]
		let copy_fallback = false;

		let lib_files = std::fs::read_dir(lib_dir).unwrap_or_else(|_| panic!("Failed to read contents of `{}` (does it exist?)", lib_dir.display()));
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
				std::fs::remove_file(&out_path).unwrap();
			}
			if !out_path.exists() {
				#[cfg(windows)]
				if std::os::windows::fs::symlink_file(&lib_path, &out_path).is_err() {
					copy_fallback = true;
					std::fs::copy(&lib_path, &out_path).unwrap();
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

fn macos_rtlib_search_dir() -> Option<String> {
	let output = Command::new(std::env::var("CC").unwrap_or_else(|_| "clang".to_string()))
		.arg("--print-search-dirs")
		.output()
		.ok()?;
	if !output.status.success() {
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

	None
}

fn static_link_prerequisites(using_pyke_libs: bool) {
	let target_triple = env::var("TARGET").unwrap();

	let cpp_link_stdlib = if let Ok(stdlib) = env::var(ORT_ENV_CXX_STDLIB).or_else(|_| env::var(ENV_CXXSTDLIB)) {
		if stdlib.is_empty() { None } else { Some(stdlib) }
	} else if target_triple.contains("msvc") {
		None
	} else if target_triple.contains("apple") {
		Some("c++".to_string())
	} else if target_triple.contains("android") {
		Some("c++_shared".to_string())
	} else {
		Some("stdc++".to_string())
	};
	if let Some(cpp_link_stdlib) = cpp_link_stdlib {
		println!("cargo:rustc-link-lib={cpp_link_stdlib}");
	}

	if target_triple.contains("apple") {
		println!("cargo:rustc-link-lib=framework=Foundation");
		if let Some(dir) = macos_rtlib_search_dir() {
			println!("cargo:rustc-link-search={dir}");
			println!("cargo:rustc-link-lib=clang_rt.osx");
		}
	}
	if target_triple.contains("windows") && using_pyke_libs {
		println!("cargo:rustc-link-lib=dxguid");
		println!("cargo:rustc-link-lib=DXCORE");
		println!("cargo:rustc-link-lib=DXGI");
		println!("cargo:rustc-link-lib=D3D12");
		println!("cargo:rustc-link-lib=DirectML");
	}
	if cfg!(feature = "webgpu") && !target_triple.contains("wasm32") && using_pyke_libs {
		println!("cargo:rustc-link-lib=webgpu_dawn");
	}
}

fn prefer_dynamic_linking() -> bool {
	match env::var(ORT_ENV_PREFER_DYNAMIC_LINK) {
		Ok(val) => val == "1" || val.to_lowercase() == "true",
		Err(_) => false
	}
}

fn skip_download() -> bool {
	match env::var(ORT_ENV_SKIP_DOWNLOAD) {
		Ok(val) => val == "1" || val.to_lowercase() == "true",
		Err(_) => false
	}
}

fn prepare_libort_dir() -> (PathBuf, bool) {
	if let Ok(base_lib_dir) = env::var(ORT_ENV_SYSTEM_LIB_LOCATION) {
		let base_lib_dir = PathBuf::from(base_lib_dir);

		let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap().to_lowercase();
		let platform_format_lib = |a: &str| {
			if target_os.contains("windows") { format!("{}.lib", a) } else { format!("lib{}.a", a) }
		};
		let optional_link_lib = |dir: &Path, lib: &str| {
			if dir.exists() && dir.join(platform_format_lib(lib)).exists() {
				add_search_dir(dir);
				println!("cargo:rustc-link-lib=static={lib}");
				true
			} else {
				false
			}
		};
		let vcpkg_target = match env::var("TARGET").as_deref() {
			Ok("i686-pc-windows-msvc") => Some("x86-windows"),
			Ok("x86_64-pc-windows-msvc") => Some("x64-windows"),
			Ok("x86_64-uwp-windows-msvc") => Some("x64-uwp"),
			Ok("aarch64-pc-windows-msvc") => Some("arm64-windows"),
			Ok("aarch64-uwp-windows-msvc") => Some("arm64-uwp"),
			Ok("aarch64-apple-darwin") => Some("arm64-osx"),
			Ok("x86_64-apple-darwin") => Some("x64-osx"),
			Ok("x86_64-unknown-linux-gnu") => Some("x64-linux"),
			Ok("armv7-linux-androideabi") => Some("arm-neon-android"),
			Ok("x86_64-linux-android") => Some("x64-android"),
			Ok("aarch64-linux-android") => Some("arm64-android"),
			_ => None
		};

		let mut profile = env::var(ORT_ENV_SYSTEM_LIB_PROFILE).unwrap_or_default();
		if profile.is_empty() {
			for i in ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"] {
				if base_lib_dir.join(i).exists() && base_lib_dir.join(i).join(platform_format_lib("onnxruntime_common")).exists() {
					profile = String::from(i);
					break;
				}
			}
		}

		add_search_dir(&base_lib_dir);

		let mut needs_link = true;
		if base_lib_dir.join(platform_format_lib("onnxruntime")).exists() {
			println!("cargo:rustc-link-lib=static=onnxruntime");
			needs_link = false;
		} else if !prefer_dynamic_linking() {
			#[allow(clippy::type_complexity)]
			let static_configs: Vec<(PathBuf, PathBuf, PathBuf, Box<dyn Fn(PathBuf, &String) -> PathBuf>)> = vec![
				(base_lib_dir.join(&profile), base_lib_dir.join("lib"), base_lib_dir.join("_deps"), Box::new(|p: PathBuf, profile| p.join(profile))),
				(base_lib_dir.join(&profile), base_lib_dir.join("lib"), base_lib_dir.join(&profile).join("_deps"), Box::new(|p: PathBuf, _| p)),
				(base_lib_dir.clone(), base_lib_dir.join("lib"), base_lib_dir.parent().unwrap().join("_deps"), Box::new(|p: PathBuf, _| p)),
				(base_lib_dir.join("onnxruntime"), base_lib_dir.join("onnxruntime").join("lib"), base_lib_dir.join("_deps"), Box::new(|p: PathBuf, _| p)),
			];
			for (lib_dir, extension_lib_dir, external_lib_dir, transform_dep) in static_configs {
				if lib_dir.join(platform_format_lib("onnxruntime_common")).exists() && external_lib_dir.exists() {
					add_search_dir(&lib_dir);

					for lib in &["common", "flatbuffers", "framework", "graph", "lora", "mlas", "optimizer", "providers", "session", "util"] {
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

					let (vcpkg_lib_dir, has_vcpkg_link) = {
						let vcpkg_base_dir = base_lib_dir.join("vcpkg_installed");
						if let Some(vcpkg_target) = vcpkg_target {
							if vcpkg_base_dir.join(vcpkg_target).exists() {
								let vcpkg_lib_dir = vcpkg_base_dir.join(vcpkg_target).join("lib");
								add_search_dir(&vcpkg_lib_dir);
								(Some(vcpkg_lib_dir), true)
							} else {
								(None, false)
							}
						} else {
							(None, false)
						}
					};

					let protobuf_build = if !has_vcpkg_link {
						let protobuf_build = transform_dep(external_lib_dir.join("protobuf-build"), &profile);
						add_search_dir(&protobuf_build);
						protobuf_build
					} else {
						vcpkg_lib_dir.clone().unwrap()
					};
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

					// some builds of ONNX Runtime, particularly the default no-EP windows build, don't require nsync
					if !has_vcpkg_link {
						optional_link_lib(&transform_dep(external_lib_dir.join("google_nsync-build"), &profile), "nsync_cpp");
					} else {
						optional_link_lib(vcpkg_lib_dir.as_ref().unwrap(), "nsync_cpp");
					}

					add_search_dir(transform_dep(external_lib_dir.join("pytorch_cpuinfo-build"), &profile));
					if !has_vcpkg_link {
						// clog isn't built when not building unit tests, or when compiling for android
						for potential_clog_path in [
							transform_dep(external_lib_dir.join("pytorch_cpuinfo-build").join("deps").join("clog"), &profile),
							transform_dep(external_lib_dir.join("pytorch_clog-build"), &profile)
						] {
							if optional_link_lib(&potential_clog_path, "clog") {
								break;
							}
						}
					} else {
						optional_link_lib(vcpkg_lib_dir.as_ref().unwrap(), "clog");
					}
					println!("cargo:rustc-link-lib=static=cpuinfo");

					if !has_vcpkg_link {
						add_search_dir(transform_dep(external_lib_dir.join("re2-build"), &profile));
					}
					println!("cargo:rustc-link-lib=static=re2");

					if has_vcpkg_link && target_os.contains("windows") {
						println!("cargo:rustc-link-lib=static=abseil_dll");
					} else {
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("debugging"), &profile));
						println!("cargo:rustc-link-lib=static=absl_examine_stack");
						println!("cargo:rustc-link-lib=static=absl_debugging_internal");
						println!("cargo:rustc-link-lib=static=absl_demangle_internal");
						println!("cargo:rustc-link-lib=static=absl_demangle_rust");
						println!("cargo:rustc-link-lib=static=absl_decode_rust_punycode");
						println!("cargo:rustc-link-lib=static=absl_utf8_for_code_point");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("base"), &profile));
						println!("cargo:rustc-link-lib=static=absl_base");
						println!("cargo:rustc-link-lib=static=absl_spinlock_wait");
						println!("cargo:rustc-link-lib=static=absl_malloc_internal");
						println!("cargo:rustc-link-lib=static=absl_strerror");
						println!("cargo:rustc-link-lib=static=absl_raw_logging_internal");
						println!("cargo:rustc-link-lib=static=absl_throw_delegate");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("hash"), &profile));
						println!("cargo:rustc-link-lib=static=absl_hash");
						println!("cargo:rustc-link-lib=static=absl_city");
						println!("cargo:rustc-link-lib=static=absl_low_level_hash");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("container"), &profile));
						println!("cargo:rustc-link-lib=static=absl_raw_hash_set");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("synchronization"), &profile));
						println!("cargo:rustc-link-lib=static=absl_kernel_timeout_internal");
						println!("cargo:rustc-link-lib=static=absl_graphcycles_internal");
						println!("cargo:rustc-link-lib=static=absl_synchronization");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("time"), &profile));
						println!("cargo:rustc-link-lib=static=absl_time_zone");
						println!("cargo:rustc-link-lib=static=absl_time");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("numeric"), &profile));
						println!("cargo:rustc-link-lib=static=absl_int128");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("strings"), &profile));
						println!("cargo:rustc-link-lib=static=absl_str_format_internal");
						println!("cargo:rustc-link-lib=static=absl_strings");
						println!("cargo:rustc-link-lib=static=absl_string_view");
						println!("cargo:rustc-link-lib=static=absl_strings_internal");
						add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("debugging"), &profile));
						println!("cargo:rustc-link-lib=static=absl_symbolize");
						println!("cargo:rustc-link-lib=static=absl_stacktrace");
						let abseil_lib_log_dir = if !has_vcpkg_link {
							let dir = transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("log"), &profile);
							add_search_dir(&dir);
							dir
						} else {
							vcpkg_lib_dir.clone().unwrap()
						};
						println!("cargo:rustc-link-lib=static=absl_log_globals");
						println!("cargo:rustc-link-lib=static=absl_log_internal_format");
						println!("cargo:rustc-link-lib=static=absl_log_internal_proto");
						println!("cargo:rustc-link-lib=static=absl_log_internal_globals");
						optional_link_lib(&abseil_lib_log_dir, "absl_log_internal_check_op");
						println!("cargo:rustc-link-lib=static=absl_log_internal_log_sink_set");
						println!("cargo:rustc-link-lib=static=absl_log_sink");
						println!("cargo:rustc-link-lib=static=absl_log_internal_message");
					}

					// link static EPs if present
					// not sure if these are the right libs but they're optional links so...
					optional_link_lib(&lib_dir, "onnxruntime_providers_acl");
					optional_link_lib(&lib_dir, "onnxruntime_providers_armnn");
					optional_link_lib(&lib_dir, "onnxruntime_providers_azure");
					if optional_link_lib(&lib_dir, "onnxruntime_providers_coreml") {
						println!("cargo:rustc-link-lib=framework=CoreML");
						println!("cargo:rustc-link-lib=coreml_proto");
					}
					if optional_link_lib(&lib_dir, "onnxruntime_providers_dml") {
						println!("cargo:rustc-link-lib=dxguid");
						println!("cargo:rustc-link-lib=DXCORE");
						println!("cargo:rustc-link-lib=DXGI");
						println!("cargo:rustc-link-lib=D3D12");
						println!("cargo:rustc-link-lib=DirectML");
					}
					optional_link_lib(&lib_dir, "onnxruntime_providers_nnapi");
					optional_link_lib(&lib_dir, "onnxruntime_providers_qnn");
					optional_link_lib(&lib_dir, "onnxruntime_providers_rknpu");
					optional_link_lib(&lib_dir, "onnxruntime_providers_tvm");
					#[cfg(feature = "webgpu")]
					if optional_link_lib(&lib_dir, "onnxruntime_providers_webgpu") {
						let dawn_build_dir = transform_dep(external_lib_dir.join("dawn-build/src/dawn"), &profile);
						add_search_dir(&dawn_build_dir);
						println!("cargo:rustc-link-lib=static=dawn_proc");

						let dawn_native_build_dir = transform_dep(external_lib_dir.join("dawn-build/src/dawn/native"), &profile);
						add_search_dir(&dawn_native_build_dir);
						println!("cargo:rustc-link-lib=static=dawn_native");

						let dawn_platform_build_dir = transform_dep(external_lib_dir.join("dawn-build/src/dawn/platform"), &profile);
						add_search_dir(&dawn_platform_build_dir);
						println!("cargo:rustc-link-lib=static=dawn_platform");

						let dawn_common_build_dir = transform_dep(external_lib_dir.join("dawn-build/src/dawn/common"), &profile);
						add_search_dir(&dawn_common_build_dir);
						println!("cargo:rustc-link-lib=static=dawn_common");

						let tint_build_dir = transform_dep(external_lib_dir.join("dawn-build/src/tint"), &profile);
						add_search_dir(&tint_build_dir);
						let pattern = format!("{}/**/lib*.a", tint_build_dir.display());
						for entry in glob::glob(&pattern).unwrap() {
							match entry {
								Ok(path) => {
									if let Some(lib_name) = path.file_name() {
										if let Some(lib_name_str) = lib_name.to_str() {
											let lib_name = lib_name_str.trim_start_matches("lib").trim_end_matches(".a");
											println!("cargo:rustc-link-lib=static={}", lib_name);
										}
									}
								}
								Err(e) => eprintln!("error matching file: {}", e)
							}
						}
					};
					if optional_link_lib(&lib_dir, "onnxruntime_providers_xnnpack") {
						let xnnpack_build_dir = transform_dep(external_lib_dir.join("googlexnnpack-build"), &profile);
						add_search_dir(&xnnpack_build_dir);
						println!("cargo:rustc-link-lib=static=XNNPACK");
						optional_link_lib(&xnnpack_build_dir, "microkernels-prod");

						add_search_dir(transform_dep(external_lib_dir.join("pthreadpool-build"), &profile));
						println!("cargo:rustc-link-lib=static=pthreadpool");
					}

					if env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "aarch64" {
						let kleidi_build_dir = transform_dep(external_lib_dir.join("kleidiai-build"), &profile);
						optional_link_lib(&kleidi_build_dir, "kleidiai");
					}

					needs_link = false;
					break;
				}
			}
			if needs_link {
				// none of the static link patterns matched, we might be trying to dynamic link so copy dylibs if requested
				#[cfg(feature = "copy-dylibs")]
				{
					let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
					if base_lib_dir.join("lib").is_dir() {
						copy_libraries(&base_lib_dir.join("lib"), &out_dir);
					} else if base_lib_dir.join(&profile).is_dir() {
						copy_libraries(&base_lib_dir.join(profile), &out_dir);
					}
				}
			}
		}

		(base_lib_dir, needs_link)
	} else {
		#[cfg(feature = "download-binaries")]
		{
			if env::var("CARGO_NET_OFFLINE").as_deref() == Ok("true") || skip_download() {
				return (PathBuf::default(), true);
			}

			let target = env::var("TARGET").unwrap().to_string();

			let mut feature_set = Vec::new();
			if cfg!(feature = "training") {
				feature_set.push("train");
			}
			if cfg!(feature = "webgpu") {
				feature_set.push("wgpu");
			}
			if cfg!(any(feature = "cuda", feature = "tensorrt")) {
				feature_set.push("cu12");
			}
			if cfg!(feature = "rocm") {
				feature_set.push("rocm");
			}

			let feature_set = if !feature_set.is_empty() { feature_set.join(",") } else { "none".to_owned() };
			println!("selected feature set: {feature_set}");

			let mut dist = find_dist(&target, &feature_set);
			if dist.is_none() && feature_set != "none" {
				println!("full feature set {feature_set} not available, attempting to download with no features instead");
				// i dont like this behavior at all but the only thing i like less than it is rust-analyzer breaking because it
				// ***insists*** on enabling --all-features
				dist = find_dist(&target, "none");
			}

			if dist.is_none() {
				panic!(
					"downloaded binaries not available for target {target}{}\nyou may have to compile ONNX Runtime from source",
					if feature_set != "none" { format!(" and features `{feature_set}`") } else { String::new() }
				);
			}

			let (prebuilt_url, prebuilt_hash) = dist.unwrap();

			let bin_extract_dir = cache_dir()
				.expect("could not determine cache directory")
				.join("dfbin")
				.join(target)
				.join(prebuilt_hash);

			let lib_dir = bin_extract_dir.join(ORT_EXTRACT_DIR);
			if !lib_dir.exists() {
				let downloaded_file = fetch_file(prebuilt_url);
				assert!(verify_file(&downloaded_file, prebuilt_hash), "hash of downloaded ONNX Runtime binary does not match!");

				let mut temp_extract_dir = bin_extract_dir
					.parent()
					.unwrap()
					.join(format!("tmp.{}_{prebuilt_hash}", self::internal::random_identifier()));
				let mut should_rename = true;
				if std::fs::create_dir_all(&temp_extract_dir).is_err() {
					temp_extract_dir = env::var("OUT_DIR").unwrap().into();
					should_rename = false;
				}
				extract_tgz(&downloaded_file, &temp_extract_dir);
				if should_rename {
					match std::fs::rename(&temp_extract_dir, &bin_extract_dir) {
						Ok(()) => {}
						Err(e) => {
							if bin_extract_dir.exists() {
								let _ = std::fs::remove_dir_all(temp_extract_dir);
							} else {
								panic!("failed to extract downloaded binaries: {e}");
							}
						}
					}
				}
			}

			static_link_prerequisites(true);

			#[cfg(feature = "copy-dylibs")]
			{
				let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
				copy_libraries(&lib_dir.join("lib"), &out_dir);
			}

			(lib_dir, true)
		}
		#[cfg(not(feature = "download-binaries"))]
		{
			println!("cargo:rustc-link-lib=add_ort_library_path_or_enable_feature_download-binaries_see_ort_docs");
			(PathBuf::default(), false)
		}
	}
}

fn try_setup_with_pkg_config() -> bool {
	match pkg_config::Config::new().probe("libonnxruntime") {
		Ok(lib) => {
			let expected_minor = ONNXRUNTIME_VERSION.split('.').nth(1).unwrap().parse::<usize>().unwrap();
			let got_minor = lib.version.split('.').nth(1).unwrap().parse::<usize>().unwrap();
			if got_minor < expected_minor {
				println!("libonnxruntime provided by pkg-config is out of date, so it will be ignored - expected {}, got {}", ONNXRUNTIME_VERSION, lib.version);
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

			println!("Using onnxruntime found by pkg-config.");
			true
		}
		Err(_) => {
			println!("onnxruntime not found using pkg-config, falling back to manual setup.");
			false
		}
	}
}

fn real_main(link: bool) {
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_PROFILE);
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_PREFER_DYNAMIC_LINK);
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_SKIP_DOWNLOAD);
	println!("cargo:rerun-if-env-changed={}", ORT_ENV_CXX_STDLIB);
	println!("cargo:rerun-if-env-changed={}", ENV_CXXSTDLIB);

	let (install_dir, needs_link) = prepare_libort_dir();

	let lib_dir = if install_dir.join("lib").exists() { install_dir.join("lib") } else { install_dir };

	if link {
		if needs_link {
			let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
			let static_lib_file_name = if target_os.contains("windows") { "onnxruntime.lib" } else { "libonnxruntime.a" };

			let static_lib_path = lib_dir.join(static_lib_file_name);
			if !prefer_dynamic_linking() && static_lib_path.exists() {
				println!("cargo:rustc-link-lib=static=onnxruntime");
			} else {
				println!("cargo:rustc-link-lib=onnxruntime");
			}
			println!("cargo:rustc-link-search=native={}", lib_dir.display());
		}

		static_link_prerequisites(false);
	}
}

fn main() {
	if env::var("DOCS_RS").is_ok() || cfg!(feature = "disable-linking") {
		// On docs.rs, A) we don't need to link, and B) we don't have network, so we couldn't download anything if we wanted to.
		// If `disable-linking` is specified, presumably the application will configure a custom backend, and the crate
		// providing said backend will have its own linking logic, so no need to do anything.
		return;
	}

	if cfg!(feature = "load-dynamic") {
		if !try_setup_with_pkg_config() {
			// Only execute the real main step if pkg-config fails and if we are using the download
			// strategy
			if cfg!(feature = "download-binaries") && env::var(ORT_ENV_SYSTEM_LIB_LOCATION).is_err() {
				// but we don't need to link to the binaries we download (so all we are doing is
				// downloading them and placing them in the output directory)
				real_main(false); // but we don't need to link to the binaries we download
			}
		}
	} else {
		// If pkg-config setup was successful, we don't need further action
		// Otherwise, if we are not using the load-dynamic feature, we need to link to the dylibs.
		if !try_setup_with_pkg_config() {
			real_main(true);
		}
	}
}
