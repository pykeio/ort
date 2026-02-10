use std::{env, process::Command};

use crate::{log, vars};

const DIST_TABLE: &str = include_str!("dist.txt");

fn find_dist(target: &str, feature_set: &str) -> Option<Distribution> {
	DIST_TABLE
		.split('\n')
		.filter(|c| !c.is_empty() && !c.starts_with('#'))
		.map(|c| c.split('\t').collect::<Vec<_>>())
		.find(|c| c[0] == feature_set && c[1] == target)
		.map(|c| Distribution { url: c[2], hash: c[3] })
}

#[derive(Debug, PartialEq, Eq)]
pub struct Distribution {
	pub url: &'static str,
	pub hash: &'static str
}

pub fn resolve_dist() -> Result<Distribution, Option<String>> {
	let target = env::var("TARGET").unwrap().to_string();

	let mut feature_set = Vec::new();
	if cfg!(feature = "training") {
		feature_set.push("train");
	}
	if cfg!(feature = "webgpu") {
		feature_set.push("wgpu");
	}
	if cfg!(any(feature = "cuda", feature = "tensorrt")) {
		match vars::get(vars::CUDA_VERSION).as_deref() {
			Some("12") => feature_set.push("cu12"),
			Some("13") => feature_set.push("cu13"),
			_ => {
				if let Some(cuda_home) = vars::get("CUDA_HOME")
					&& (cuda_home.contains("v13.") || cuda_home.contains("-13."))
				{
					// People often have CUDA_HOME set to the CUDA dir.
					// On Windows this is usually C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
					// On Linux this is usually /usr/local/cuda-13.1
					// so detecting v13. or -13. in the path usually works
					feature_set.push("cu13");
				} else if let Some(ver) = vars::get("NV_CUDA_CUDART_VERSION")
					&& ver.starts_with("13.")
				{
					// Set by NVIDIA docker images (both devel & runtime)
					feature_set.push("cu13");
				} else if let Ok(output) = Command::new("nvcc").arg("--version").output()
					&& let Ok(stdout) = str::from_utf8(&output.stdout)
					&& stdout.contains("Build cuda_13")
				{
					feature_set.push("cu13");
				} else {
					feature_set.push("cu12");
				}
			}
		}
		feature_set.push("cu12");
	} else if cfg!(feature = "nvrtx") {
		// CUDA builds include NVRTX; only use the standalone NVRTX build if we aren't using CUDA as well
		feature_set.push("nvrtx");
	}
	if cfg!(feature = "rocm") {
		feature_set.push("rocm");
	}

	let feature_set = if !feature_set.is_empty() { feature_set.join(",") } else { "none".to_owned() };
	log::debug!("looking for prebuilt binaries matching feature set: {feature_set}");

	let mut dist = find_dist(&target, &feature_set);
	if dist.is_none() && feature_set != "none" {
		log::warning!("no prebuilt binaries available on this platform for combination of features '{feature_set}'");
		// i dont like this behavior at all but the only thing i like less than it is rust-analyzer breaking because it
		// ***insists*** on enabling --all-features
		dist = find_dist(&target, "none");
	}

	match dist {
		Some(dist) => Ok(dist),
		None => Err(if feature_set != "none" { Some(feature_set) } else { None })
	}
}
