use std::{
	env,
	path::{Path, PathBuf}
};

use crate::{log, vars};

mod apple;
pub use self::apple::link_ios_frameworks;

#[derive(Debug, PartialEq, Eq)]
pub enum BinariesSource {
	Pyke,
	UserProvided
}

pub fn static_link_prerequisites(source: BinariesSource) {
	let target_triple = env::var("TARGET").unwrap();

	let cpp_link_stdlib = if let Some(stdlib) = vars::get(vars::CXX_STDLIB).or_else(|| vars::get(vars::CXX_STDLIB_ALT)) {
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

	if target_triple.contains("apple-darwin") {
		println!("cargo:rustc-link-lib=framework=Foundation");
		if let Some(dir) = apple::macos_rtlib_search_dir() {
			println!("cargo:rustc-link-search={dir}");
			println!("cargo:rustc-link-lib=clang_rt.osx");
		}
	} else if target_triple.contains("apple-ios") {
		println!("cargo:rustc-link-lib=framework=Foundation");
		println!("cargo:rustc-link-lib=framework=CoreML");
		if let Some(dir) = apple::ios_rtlib_search_dir() {
			println!("cargo:rustc-link-search={dir}");
			if target_triple.contains("ios-sim") {
				println!("cargo:rustc-link-lib=clang_rt.iossim");
			} else {
				println!("cargo:rustc-link-lib=clang_rt.ios");
			}
		}
	}

	if source == BinariesSource::Pyke {
		if target_triple.contains("windows") {
			// pyke libs always ship compiled with DirectML on Windows, so we need to link to DX12 libraries.
			println!("cargo:rustc-link-lib=dxguid");
			println!("cargo:rustc-link-lib=DXCORE");
			println!("cargo:rustc-link-lib=DXGI");
			println!("cargo:rustc-link-lib=D3D12");
			println!("cargo:rustc-link-lib=DirectML");
		}
		if cfg!(feature = "webgpu") && !target_triple.contains("wasm32") {
			// Dawn cannot be linked statically yet so it's shipped as a dylib we need to link to.
			println!("cargo:rustc-link-lib=webgpu_dawn");
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

pub fn static_link(base_lib_dir: &Path) -> bool {
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

	let mut profile = vars::get(vars::SYSTEM_LIB_PROFILE).unwrap_or_default();
	if profile.is_empty() {
		for i in ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"] {
			if base_lib_dir.join(i).exists() && base_lib_dir.join(i).join(platform_format_lib("onnxruntime_common")).exists() {
				profile = String::from(i);
				break;
			}
		}
	}

	add_search_dir(base_lib_dir);

	if base_lib_dir.join(platform_format_lib("onnxruntime")).exists() {
		println!("cargo:rustc-link-lib=static=onnxruntime");
		return true;
	}

	log::debug!("doing full static linking since no single-file library was found");

	#[allow(clippy::type_complexity)]
	let static_configs: Vec<(PathBuf, PathBuf, PathBuf, Box<dyn Fn(PathBuf, &String) -> PathBuf>)> = vec![
		(base_lib_dir.join(&profile), base_lib_dir.join("lib"), base_lib_dir.join("_deps"), Box::new(|p: PathBuf, profile| p.join(profile))),
		(base_lib_dir.join(&profile), base_lib_dir.join("lib"), base_lib_dir.join(&profile).join("_deps"), Box::new(|p: PathBuf, _| p)),
		(base_lib_dir.to_owned(), base_lib_dir.join("lib"), base_lib_dir.parent().unwrap().join("_deps"), Box::new(|p: PathBuf, _| p)),
		(base_lib_dir.join("onnxruntime"), base_lib_dir.join("onnxruntime").join("lib"), base_lib_dir.join("_deps"), Box::new(|p: PathBuf, _| p)),
	];
	'main: for (lib_dir, extension_lib_dir, external_lib_dir, transform_dep) in static_configs {
		if lib_dir.join(platform_format_lib("onnxruntime_common")).exists() && external_lib_dir.exists() {
			log::debug!("attempting to link from {}", lib_dir.display());

			add_search_dir(&lib_dir);

			for lib in &["common", "flatbuffers", "framework", "graph", "lora", "mlas", "optimizer", "providers", "session", "util"] {
				let lib_name = platform_format_lib(&format!("onnxruntime_{lib}"));
				let lib_path = lib_dir.join(&lib_name);
				// sanity check, just make sure the library exists before we try to link to it
				if lib_path.exists() {
					println!("cargo:rustc-link-lib=static=onnxruntime_{lib}");
				} else {
					log::warning!("directory is missing {lib_name}!");
					continue 'main;
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
						log::debug!("using vcpkg libraries from {}", vcpkg_lib_dir.display());
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
				println!("cargo:rustc-link-lib=static=absl_hashtablez_sampler");
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
							if let Some(lib_name) = path.file_name()
								&& let Some(lib_name_str) = lib_name.to_str()
							{
								let lib_name = lib_name_str.trim_start_matches("lib").trim_end_matches(".a");
								println!("cargo:rustc-link-lib=static={}", lib_name);
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

			return true;
		}
	}

	false
}
