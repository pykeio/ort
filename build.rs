#![allow(unused)]

use std::{
	borrow::Cow,
	env, fs,
	io::{self, Read, Write},
	path::{Path, PathBuf},
	process::Stdio,
	str::FromStr
};

const ORT_VERSION: &str = "1.15.1";
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";
const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_ENV_CMAKE_TOOLCHAIN: &str = "ORT_CMAKE_TOOLCHAIN";
const ORT_ENV_CMAKE_PROGRAM: &str = "ORT_CMAKE_PROGRAM";
const ORT_ENV_PYTHON_PROGRAM: &str = "ORT_PYTHON_PROGRAM";
const ORT_EXTRACT_DIR: &str = "onnxruntime";
const ORT_GIT_DIR: &str = "onnxruntime";
const ORT_GIT_REPO: &str = "https://github.com/microsoft/onnxruntime";
const PROTOBUF_EXTRACT_DIR: &str = "protobuf";
const PROTOBUF_VERSION: &str = "3.18.1";
const PROTOBUF_RELEASE_BASE_URL: &str = "https://github.com/protocolbuffers/protobuf/releases/download";

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
trait OnnxPrebuiltArchive {
	fn as_onnx_str(&self) -> Cow<str>;
}

#[cfg(feature = "download-binaries")]
#[derive(Debug)]
enum Architecture {
	X86,
	X86_64,
	Arm,
	Arm64
}

#[cfg(feature = "download-binaries")]
impl FromStr for Architecture {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"x86" => Ok(Architecture::X86),
			"x86_64" => Ok(Architecture::X86_64),
			"arm" => Ok(Architecture::Arm),
			"aarch64" => Ok(Architecture::Arm64),
			_ => Err(format!(
				"Unsupported architecture for binary download: {s}\nMicrosoft does not provide prebuilt binaries for this platform.\nYou'll need to build ONNX Runtime from source, disable the `download-binaries` feature, and link `ort` to your compiled libraries. See https://github.com/pykeio/ort#how-to-get-binaries"
			))
		}
	}
}

#[cfg(feature = "download-binaries")]
impl OnnxPrebuiltArchive for Architecture {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Architecture::X86 => "x86".into(),
			Architecture::X86_64 => "x64".into(),
			Architecture::Arm => "arm".into(),
			Architecture::Arm64 => "arm64".into()
		}
	}
}

#[cfg(feature = "download-binaries")]
#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
	Windows,
	Linux,
	MacOS
}

#[cfg(feature = "download-binaries")]
impl Os {
	fn archive_extension(&self) -> &'static str {
		match self {
			Os::Windows => "zip",
			Os::Linux => "tgz",
			Os::MacOS => "tgz"
		}
	}
}

#[cfg(feature = "download-binaries")]
impl FromStr for Os {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"windows" => Ok(Os::Windows),
			"linux" => Ok(Os::Linux),
			"macos" => Ok(Os::MacOS),
			_ => Err(format!(
				"Unsupported OS for binary download: {s}\nMicrosoft does not provide prebuilt binaries for this platform.\nYou'll need to build ONNX Runtime from source, disable the `download-binaries` feature, and link `ort` to your compiled libraries. See https://github.com/pykeio/ort#how-to-get-binaries"
			))
		}
	}
}

#[cfg(feature = "download-binaries")]
impl OnnxPrebuiltArchive for Os {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Os::Windows => "win".into(),
			Os::Linux => "linux".into(),
			Os::MacOS => "osx".into()
		}
	}
}

#[cfg(feature = "download-binaries")]
#[derive(Debug)]
enum Accelerator {
	None,
	Gpu
}

#[cfg(feature = "download-binaries")]
impl OnnxPrebuiltArchive for Accelerator {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Accelerator::None => "unaccelerated".into(),
			Accelerator::Gpu => "gpu".into()
		}
	}
}

#[cfg(feature = "download-binaries")]
#[derive(Debug)]
struct Triplet {
	os: Os,
	arch: Architecture,
	accelerator: Accelerator
}

#[cfg(feature = "download-binaries")]
impl OnnxPrebuiltArchive for Triplet {
	fn as_onnx_str(&self) -> Cow<str> {
		match (&self.os, &self.arch, &self.accelerator) {
			(Os::Windows, Architecture::X86, Accelerator::None)
			| (Os::Windows, Architecture::X86_64, Accelerator::None)
			| (Os::Windows, Architecture::Arm, Accelerator::None)
			| (Os::Windows, Architecture::Arm64, Accelerator::None)
			| (Os::Linux, Architecture::X86_64, Accelerator::None)
			| (Os::MacOS, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str()).into(),
			// for some reason, arm64/Linux uses `aarch64` instead of `arm64`
			(Os::Linux, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "aarch64").into(),
			// for another odd reason, x64/macOS uses `x86_64` instead of `x64`
			(Os::MacOS, Architecture::X86_64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "x86_64").into(),
			(Os::Windows, Architecture::X86_64, Accelerator::Gpu) | (Os::Linux, Architecture::X86_64, Accelerator::Gpu) => {
				format!("{}-{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str(), self.accelerator.as_onnx_str()).into()
			}
			_ => panic!(
				"Microsoft does not provide ONNX Runtime downloads for triplet: {}-{}-{}; you may have to use the `system` strategy instead",
				self.os.as_onnx_str(),
				self.arch.as_onnx_str(),
				self.accelerator.as_onnx_str()
			)
		}
	}
}

#[cfg(feature = "download-binaries")]
fn prebuilt_onnx_url() -> (PathBuf, String) {
	let accelerator = if cfg!(feature = "cuda") || cfg!(feature = "tensorrt") {
		Accelerator::Gpu
	} else {
		Accelerator::None
	};

	let triplet = Triplet {
		os: env::var("CARGO_CFG_TARGET_OS")
			.expect("unable to get target OS")
			.parse()
			.expect("unsupported target OS"),
		arch: env::var("CARGO_CFG_TARGET_ARCH")
			.expect("unable to get target arch")
			.parse()
			.expect("unsupported target arch"),
		accelerator
	};

	let prebuilt_archive = format!("onnxruntime-{}-{}.{}", triplet.as_onnx_str(), ORT_VERSION, triplet.os.archive_extension());
	let prebuilt_url = format!("{ORT_RELEASE_BASE_URL}/v{ORT_VERSION}/{prebuilt_archive}");

	(PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn prebuilt_protoc_url() -> (PathBuf, String) {
	let host_platform = if cfg!(target_os = "windows") {
		std::string::String::from("win32")
	} else if cfg!(target_os = "macos") {
		format!(
			"osx-{}",
			if cfg!(target_arch = "x86_64") {
				"x86_64"
			} else if cfg!(target_arch = "x86") {
				"x86"
			} else {
				panic!("protoc does not have prebuilt binaries for darwin arm64 yet")
			}
		)
	} else {
		format!("linux-{}", if cfg!(target_arch = "x86_64") { "x86_64" } else { "x86_32" })
	};

	let prebuilt_archive = format!("protoc-{PROTOBUF_VERSION}-{host_platform}.zip");
	let prebuilt_url = format!("{PROTOBUF_RELEASE_BASE_URL}/v{PROTOBUF_VERSION}/{prebuilt_archive}");

	(PathBuf::from(prebuilt_archive), prebuilt_url)
}

#[cfg(feature = "download-binaries")]
fn download<P>(source_url: &str, target_file: P)
where
	P: AsRef<Path>
{
	let resp = ureq::get(source_url)
		.timeout(std::time::Duration::from_secs(1800))
		.call()
		.unwrap_or_else(|err| panic!("[ort] failed to download {source_url}: {err:?}"));

	let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
	let mut reader = resp.into_reader();
	// FIXME: Save directly to the file
	let mut buffer = vec![];
	let read_len = reader.read_to_end(&mut buffer).unwrap();
	assert_eq!(buffer.len(), len);
	assert_eq!(buffer.len(), read_len);

	let f = fs::File::create(&target_file).unwrap();
	let mut writer = io::BufWriter::new(f);
	writer.write_all(&buffer).unwrap();
}

#[cfg(feature = "download-binaries")]
fn extract_archive(filename: &Path, output: &Path) {
	match filename.extension().map(|e| e.to_str()) {
		Some(Some("zip")) => extract_zip(filename, output),
		#[cfg(not(target_os = "windows"))]
		Some(Some("tgz")) => extract_tgz(filename, output),
		_ => unimplemented!()
	}
}

#[cfg(all(feature = "download-binaries", not(target_os = "windows")))]
fn extract_tgz(filename: &Path, output: &Path) {
	let file = fs::File::open(filename).unwrap();
	let buf = io::BufReader::new(file);
	let tar = flate2::read::GzDecoder::new(buf);
	let mut archive = tar::Archive::new(tar);
	archive.unpack(output).unwrap();
}

#[cfg(feature = "download-binaries")]
fn extract_zip(filename: &Path, outpath: &Path) {
	let file = fs::File::open(filename).unwrap();
	let buf = io::BufReader::new(file);
	let mut archive = zip::ZipArchive::new(buf).unwrap();
	for i in 0..archive.len() {
		let mut file = archive.by_index(i).unwrap();
		#[allow(deprecated)]
		let outpath = outpath.join(file.enclosed_name().unwrap());
		if !file.name().ends_with('/') {
			println!("File {} extracted to \"{}\" ({} bytes)", i, outpath.as_path().display(), file.size());
			if let Some(p) = outpath.parent() {
				if !p.exists() {
					fs::create_dir_all(p).unwrap();
				}
			}
			let mut outfile = fs::File::create(&outpath).unwrap();
			io::copy(&mut file, &mut outfile).unwrap();
		}
	}
}

fn copy_libraries(lib_dir: &Path, out_dir: &Path) {
	// get the target directory - we need to place the dlls next to the executable so they can be properly loaded by windows
	let out_dir = out_dir.parent().unwrap().parent().unwrap().parent().unwrap();

	let lib_files = fs::read_dir(lib_dir).unwrap();
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
			fs::copy(&lib_path, out_path).unwrap();
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

fn system_strategy() -> (PathBuf, bool) {
	let lib_dir = PathBuf::from(env::var(ORT_ENV_SYSTEM_LIB_LOCATION).expect("[ort] system strategy requires ORT_LIB_LOCATION env var to be set"));

	let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap().to_lowercase();
	let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap().to_lowercase();
	let platform_format_lib = |a: &str| {
		if target_os.contains("windows") { format!("{}.lib", a) } else { format!("lib{}.a", a) }
	};

	let mut profile = String::new();
	for i in ["Release", "Debug", "MinSizeRel", "RelWithDebInfo"] {
		if lib_dir.join(i).exists() && lib_dir.join(i).join(platform_format_lib("onnxruntime_common")).exists() {
			profile = String::from(i);
			break;
		}
	}

	if cfg!(target_os = "macos") {
		println!("cargo:rustc-link-lib=c++");
		println!("cargo:rustc-link-lib=framework=Foundation");
	} else if cfg!(target_os = "linux") {
		println!("cargo:rustc-link-lib=stdc++");
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

				if extension_lib_dir.exists() {
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
				if lib_dir.join("lib").is_dir() {
					copy_libraries(&lib_dir.join("lib"), &PathBuf::from(env::var("OUT_DIR").unwrap()));
				} else if lib_dir.join(&profile).is_dir() {
					copy_libraries(&lib_dir.join(profile), &PathBuf::from(env::var("OUT_DIR").unwrap()));
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
			} else {
				incompatible_providers![ONEDNN, COREML, OPENVINO, VITIS, TVM, MIGRAPHX, DIRECTML, WINML, ACML, ARMNN, ROCM];
			}

			let (prebuilt_archive, prebuilt_url) = prebuilt_onnx_url();

			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
			let extract_dir = out_dir.join(ORT_EXTRACT_DIR);
			let downloaded_file = out_dir.join(&prebuilt_archive);

			println!("cargo:rerun-if-changed={}", downloaded_file.display());

			if !downloaded_file.exists() {
				fs::create_dir_all(&out_dir).unwrap();
				download(&prebuilt_url, &downloaded_file);
			}

			if !extract_dir.exists() {
				extract_archive(&downloaded_file, &extract_dir);
			}

			let lib_dir = extract_dir.join(prebuilt_archive.file_stem().unwrap());
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
		"compile" => {
			use std::process::Command;

			let target = env::var("TARGET").unwrap();
			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

			let python = env::var("PYTHON").unwrap_or_else(|_| "python".to_string());

			Command::new("git")
				.args([
					"clone",
					"--depth",
					"1",
					"--single-branch",
					"--branch",
					&format!("v{ORT_VERSION}"),
					"--shallow-submodules",
					"--recursive",
					ORT_GIT_REPO,
					ORT_GIT_DIR
				])
				.current_dir(&out_dir)
				.stdout(Stdio::null())
				.stderr(Stdio::null())
				.status()
				.expect("failed to clone ORT repo");

			let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
			let _cmake_toolchain = env::var(ORT_ENV_CMAKE_TOOLCHAIN).map_or_else(
				|_| {
					if cfg!(target_os = "linux") && target.contains("aarch64") && target.contains("linux") {
						root.join("toolchains").join("default-aarch64-linux-gnu.cmake")
					} else if cfg!(target_os = "linux") && target.contains("aarch64") && target.contains("windows") {
						root.join("toolchains").join("default-aarch64-w64-mingw32.cmake")
					} else if cfg!(target_os = "linux") && target.contains("x86_64") && target.contains("windows") {
						root.join("toolchains").join("default-x86_64-w64-mingw32.cmake")
					} else {
						PathBuf::default()
					}
				},
				PathBuf::from
			);

			let mut command = Command::new(python);
			command
				.current_dir(&out_dir.join(ORT_GIT_DIR))
				.stdout(Stdio::null())
				.stderr(Stdio::inherit());

			// note: --parallel will probably break something... parallel build *while* doing another parallel build (cargo)?
			let mut build_args = vec!["tools/ci_build/build.py", "--build", "--update", "--parallel", "--skip_tests", "--skip_submodule_sync"];
			let config = if cfg!(debug_assertions) {
				"Debug"
			} else if cfg!(feature = "minimal-build") {
				"MinSizeRel"
			} else {
				"Release"
			};
			build_args.push("--config");
			build_args.push(config);

			if cfg!(feature = "minimal-build") {
				build_args.push("--disable_exceptions");
			}

			build_args.push("--disable_rtti");

			if target.contains("windows") {
				build_args.push("--disable_memleak_checker");
			}

			if !cfg!(feature = "compile-static") {
				build_args.push("--build_shared_lib");
			} else {
				build_args.push("--enable_msvc_static_runtime");
			}

			// onnxruntime will still build tests when --skip_tests is enabled, this filters out most of them
			// this "fixes" compilation on alpine: https://github.com/microsoft/onnxruntime/issues/9155
			// but causes other compilation errors: https://github.com/microsoft/onnxruntime/issues/7571
			// build_args.push("--cmake_extra_defines");
			// build_args.push("onnxruntime_BUILD_UNIT_TESTS=0");

			#[cfg(windows)]
			{
				use vswhom::VsFindResult;
				let vs_find_result = VsFindResult::search();
				match vs_find_result {
					Some(VsFindResult { vs_exe_path: Some(vs_exe_path), .. }) => {
						let vs_exe_path = vs_exe_path.to_string_lossy();
						// the one sane thing about visual studio is that the version numbers are somewhat predictable...
						if vs_exe_path.contains("14.1") {
							build_args.push("--cmake_generator=Visual Studio 15 2017");
						} else if vs_exe_path.contains("14.2") {
							build_args.push("--cmake_generator=Visual Studio 16 2019");
						} else if vs_exe_path.contains("14.3") {
							build_args.push("--cmake_generator=Visual Studio 17 2022");
						}
					}
					Some(VsFindResult { vs_exe_path: None, .. }) | None => panic!("[ort] unable to find Visual Studio installation")
				};
			}

			build_args.push("--build_dir=build");
			command.args(build_args);

			let code = command.status().expect("failed to run build script");
			assert!(code.success(), "failed to build ONNX Runtime");

			let lib_dir = out_dir.join(ORT_GIT_DIR).join("build").join(config);
			let lib_dir = if cfg!(target_os = "windows") { lib_dir.join(config) } else { lib_dir };
			for lib in &["common", "flatbuffers", "framework", "graph", "mlas", "optimizer", "providers", "session", "util"] {
				let lib_path = lib_dir.join(if cfg!(target_os = "windows") {
					format!("onnxruntime_{lib}.lib")
				} else {
					format!("libonnxruntime_{lib}.a")
				});
				// sanity check, just make sure the library exists before we try to link to it
				if lib_path.exists() {
					println!("cargo:rustc-link-lib=static=onnxruntime_{lib}");
				} else {
					panic!("[ort] unable to find ONNX Runtime library: {}", lib_path.display());
				}
			}

			println!("cargo:rustc-link-search=native={}", lib_dir.display());

			let external_lib_dir = lib_dir.join("external");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("protobuf").join("cmake").display());
			println!("cargo:rustc-link-lib=static=protobuf-lited");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("onnx").display());
			println!("cargo:rustc-link-lib=static=onnx");
			println!("cargo:rustc-link-lib=static=onnx_proto");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("nsync").display());
			println!("cargo:rustc-link-lib=static=nsync_cpp");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("re2").display());
			println!("cargo:rustc-link-lib=static=re2");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("base").display());
			println!("cargo:rustc-link-lib=static=absl_base");
			println!("cargo:rustc-link-lib=static=absl_throw_delegate");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("hash").display());
			println!("cargo:rustc-link-lib=static=absl_hash");
			println!("cargo:rustc-link-lib=static=absl_low_level_hash");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("container").display());
			println!("cargo:rustc-link-lib=static=absl_raw_hash_set");

			if cfg!(target_os = "macos") {
				println!("cargo:rustc-link-lib=framework=Foundation");
			}

			println!("cargo:rustc-link-lib=onnxruntime_providers_shared");
			#[cfg(feature = "rocm")]
			println!("cargo:rustc-link-lib=onnxruntime_providers_rocm");

			(out_dir, false)
		}
		_ => panic!("[ort] unknown strategy: {} (valid options are `download` or `system`)", strategy.unwrap_or_else(|_| "unknown".to_string()))
	}
}

fn real_main(link: bool) {
	let (install_dir, needs_link) = prepare_libort_dir();

	let (include_dir, lib_dir) = if install_dir.join("include").exists() && install_dir.join("lib").exists() {
		(install_dir.join("include"), install_dir.join("lib"))
	} else {
		(install_dir.clone(), install_dir)
	};

	if link {
		if needs_link {
			println!("cargo:rustc-link-lib=onnxruntime");
			println!("cargo:rustc-link-search=native={}", lib_dir.display());
		}

		println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);
	}

	println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
}

fn main() {
	if std::env::var("DOCS_RS").is_err() {
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
}
