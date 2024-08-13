use std::{
	env, fs,
	path::{Path, PathBuf},
	process::Command
};

#[allow(unused)]
const ONNXRUNTIME_VERSION: &str = "1.19.0";

const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_ENV_SYSTEM_LIB_PROFILE: &str = "ORT_LIB_PROFILE";
#[cfg(feature = "download-binaries")]
const ORT_EXTRACT_DIR: &str = "onnxruntime";

const DIST_TABLE: &str = include_str!("dist.txt");

#[path = "src/internal/dirs.rs"]
mod dirs;
use self::dirs::cache_dir;

#[cfg(feature = "download-binaries")]
fn fetch_file(source_url: &str) -> Vec<u8> {
	let resp = ureq::AgentBuilder::new()
		.try_proxy_from_env(true)
		.build()
		.get(source_url)
		.timeout(std::time::Duration::from_secs(1800))
		.call()
		.unwrap_or_else(|err| panic!("Failed to GET `{source_url}`: {err}"));

	let len = resp
		.header("Content-Length")
		.and_then(|s| s.parse::<usize>().ok())
		.expect("Content-Length header should be present on archive response");
	let mut reader = resp.into_reader();
	let mut buffer = Vec::new();
	reader
		.read_to_end(&mut buffer)
		.unwrap_or_else(|err| panic!("Failed to download from `{source_url}`: {err}"));
	assert_eq!(buffer.len(), len);
	buffer
}

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

	c.as_ref().chunks(2).map(|n| nibble(n[0]) << 4 | nibble(n[1])).collect()
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
			e.as_ref().ok().map_or(false, |e| {
				e.file_type().map_or(false, |e| !e.is_dir()) && [".dll", ".so", ".dylib"].into_iter().any(|v| e.path().to_string_lossy().contains(v))
			})
		}) {
			let lib_file = lib_file.unwrap();
			let lib_path = lib_file.path();
			let lib_name = lib_path.file_name().unwrap();
			let out_path = out_dir.join(lib_name);
			if !out_path.exists() {
				if out_path.is_symlink() {
					fs::remove_file(&out_path).unwrap();
				}
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

		#[cfg(target_os = "linux")]
		{
			let main_dy = lib_dir.join("libonnxruntime.so");
			let versioned_dy = out_dir.join(format!("libonnxruntime.so.{}", ONNXRUNTIME_VERSION));
			if main_dy.exists() && !versioned_dy.exists() {
				if versioned_dy.is_symlink() {
					fs::remove_file(&versioned_dy).unwrap();
				}
				std::os::unix::fs::symlink(main_dy, versioned_dy).unwrap();
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

fn static_link_prerequisites(using_pyke_libs: bool) {
	let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
	if target_os == "macos" || target_os == "ios" {
		println!("cargo:rustc-link-lib=c++");
		println!("cargo:rustc-link-lib=framework=Foundation");
	} else if target_os == "linux" || target_os == "android" {
		println!("cargo:rustc-link-lib=stdc++");
	} else if target_os == "windows" && (using_pyke_libs || cfg!(feature = "directml")) {
		println!("cargo:rustc-link-lib=dxguid");
		println!("cargo:rustc-link-lib=DXCORE");
		println!("cargo:rustc-link-lib=DXGI");
		println!("cargo:rustc-link-lib=D3D12");
		println!("cargo:rustc-link-lib=DirectML");
	}
}

fn prepare_libort_dir() -> (PathBuf, bool) {
	if let Ok(lib_dir) = env::var(ORT_ENV_SYSTEM_LIB_LOCATION) {
		let lib_dir = PathBuf::from(lib_dir);

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
				(lib_dir.join(&profile), lib_dir.join("lib"), lib_dir.join(&profile).join("_deps"), Box::new(|p: PathBuf, _| p)),
				(lib_dir.clone(), lib_dir.join("lib"), lib_dir.parent().unwrap().join("_deps"), Box::new(|p: PathBuf, _| p)),
				(lib_dir.join("onnxruntime"), lib_dir.join("onnxruntime").join("lib"), lib_dir.join("_deps"), Box::new(|p: PathBuf, _| p)),
			];
			for (lib_dir, extension_lib_dir, external_lib_dir, transform_dep) in static_configs {
				if lib_dir.join(platform_format_lib("onnxruntime_common")).exists() && external_lib_dir.exists() {
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

					let nsync_path = transform_dep(external_lib_dir.join("google_nsync-build"), &profile);
					// some builds of ONNX Runtime, particularly the default no-EP windows build, don't require nsync
					if nsync_path.exists() {
						add_search_dir(nsync_path);
						println!("cargo:rustc-link-lib=static=nsync_cpp");
					}

					add_search_dir(transform_dep(external_lib_dir.join("pytorch_cpuinfo-build"), &profile));
					let clog_path = transform_dep(external_lib_dir.join("pytorch_cpuinfo-build").join("deps").join("clog"), &profile);
					if clog_path.exists() {
						add_search_dir(clog_path);
					} else {
						add_search_dir(transform_dep(external_lib_dir.join("pytorch_clog-build"), &profile));
					}
					println!("cargo:rustc-link-lib=static=cpuinfo");
					println!("cargo:rustc-link-lib=static=clog");

					add_search_dir(transform_dep(external_lib_dir.join("re2-build"), &profile));
					println!("cargo:rustc-link-lib=static=re2");

					add_search_dir(transform_dep(external_lib_dir.join("abseil_cpp-build").join("absl").join("base"), &profile));
					println!("cargo:rustc-link-lib=static=absl_base");
					println!("cargo:rustc-link-lib=static=absl_spinlock_wait");
					println!("cargo:rustc-link-lib=static=absl_malloc_internal");
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

					if cfg!(feature = "coreml") && (target_os == "macos" || target_os == "ios") {
						println!("cargo:rustc-link-lib=framework=CoreML");
						println!("cargo:rustc-link-lib=coreml_proto");
						println!("cargo:rustc-link-lib=onnxruntime_providers_coreml");
					}

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
	} else {
		#[cfg(feature = "download-binaries")]
		{
			let target = env::var("TARGET").unwrap().to_string();

			let mut feature_set = Vec::new();
			if cfg!(feature = "training") {
				feature_set.push("train");
			}
			if cfg!(any(feature = "cuda", feature = "tensorrt")) {
				// pytorch's CUDA docker images set `NV_CUDNN_VERSION`
				let cu12_tag = match env::var("NV_CUDNN_VERSION").or_else(|_| env::var("ORT_CUDNN_VERSION")).as_deref() {
					Ok(v) => {
						if v.starts_with("8") {
							"cu12+cudnn8"
						} else {
							"cu12"
						}
					}
					Err(_) => "cu12"
				};

				match env::var("ORT_DFBIN_FORCE_CUDA_VERSION").as_deref() {
					Ok("11") => feature_set.push("cu11"),
					Ok("12") => feature_set.push("cu12"),
					_ => {
						let mut success = false;
						if let Ok(nvcc_output) = Command::new("nvcc").arg("--version").output() {
							if nvcc_output.status.success() {
								let stdout = String::from_utf8_lossy(&nvcc_output.stdout);
								let version_line = stdout.lines().nth(3).unwrap();
								let release_section = version_line.split(", ").nth(1).unwrap();
								let version_number = release_section.split(' ').nth(1).unwrap();
								if version_number.starts_with("12") {
									feature_set.push(cu12_tag);
								} else {
									feature_set.push("cu11");
								}
								success = true;
							}
						}

						if !success {
							println!("cargo:warning=nvcc call did not succeed. falling back to CUDA 12");
							// fallback to CUDA 12.
							feature_set.push(cu12_tag);
						}
					}
				}
			} else if cfg!(feature = "rocm") {
				feature_set.push("rocm");
			}
			let feature_set = if !feature_set.is_empty() { feature_set.join(",") } else { "none".to_owned() };
			println!("selected feature set: {feature_set}");
			let mut dist = find_dist(&target, &feature_set);
			if dist.is_none() && feature_set != "none" {
				dist = find_dist(&target, "none");
			}

			if dist.is_none() {
				panic!(
					"downloaded binaries not available for target {target}{}\nyou may have to compile ONNX Runtime from source",
					if feature_set != "none" {
						format!(" (note: also requested features `{feature_set}`)")
					} else {
						String::new()
					}
				);
			}

			let (prebuilt_url, prebuilt_hash) = dist.unwrap();

			let mut cache_dir = cache_dir()
				.expect("could not determine cache directory")
				.join("dfbin")
				.join(target)
				.join(prebuilt_hash);
			if fs::create_dir_all(&cache_dir).is_err() {
				cache_dir = env::var("OUT_DIR").unwrap().into();
			}

			let lib_dir = cache_dir.join(ORT_EXTRACT_DIR);
			if !lib_dir.exists() {
				let downloaded_file = fetch_file(prebuilt_url);
				assert!(verify_file(&downloaded_file, prebuilt_hash), "hash of downloaded ONNX Runtime binary does not match!");
				extract_tgz(&downloaded_file, &cache_dir);
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

	let (install_dir, needs_link) = prepare_libort_dir();

	let lib_dir = if install_dir.join("lib").exists() { install_dir.join("lib") } else { install_dir };

	if link {
		if needs_link {
			let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
			let static_lib_file_name = if target_os.contains("windows") { "onnxruntime.lib" } else { "libonnxruntime.a" };

			let static_lib_path = lib_dir.join(static_lib_file_name);
			if static_lib_path.exists() {
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
	if env::var("DOCS_RS").is_ok() {
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
