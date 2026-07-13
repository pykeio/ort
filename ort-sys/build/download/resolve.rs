use std::{cmp::Reverse, collections::HashSet, env, process::Command};

use crate::{log, vars};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Distribution {
	pub features: HashSet<&'static str>,
	pub url: &'static str,
	pub hash: &'static str
}

impl Distribution {
	pub fn features_str(&self) -> String {
		features_str(&self.features)
	}
}

fn parse_dist_table(target: &str) -> impl Iterator<Item = Distribution> + '_ {
	include_str!("dist.tsv")
		.split('\n')
		.skip(1)
		.filter(|c| !c.is_empty() && !c.starts_with('#'))
		.map(|c| c.split('\t').collect::<Vec<_>>())
		.filter(move |c| c[1] == target)
		.map(|c| Distribution {
			features: if c[0] == "none" { HashSet::new() } else { c[0].split(',').collect() },
			url: c[2],
			hash: c[3]
		})
}

fn features_str(set: &HashSet<&'static str>) -> String {
	if !set.is_empty() {
		let mut set = set.iter().copied().collect::<Vec<_>>();
		set.sort();
		set.join(", ")
	} else {
		"(no features)".to_string()
	}
}

macro_rules! add_features {
	($set:ident <- [$($flag:literal),+]) => {
		$(
			#[cfg(feature = $flag)]
			$set.insert($flag);
		)+
	};
}

pub fn resolve_dist() -> Result<Distribution, (String, Vec<Distribution>)> {
	let target = env::var("TARGET").unwrap().to_string();

	let mut feature_set = HashSet::new();
	add_features!(feature_set <- [
		"training",
		// CUDA has its own handling below
		"tensorrt", "openvino", "onednn", "directml", "nnapi", "coreml", "xnnpack", "rocm", "acl", "armnn", "tvm", "migraphx",
		"rknpu", "vitis", "cann", "qnn", "webgpu", "azure", "nvrtx", "vsinpu"
	]);

	if cfg!(any(feature = "cuda", feature = "tensorrt")) {
		let cuda_feature = match vars::get(vars::CUDA_VERSION).as_deref() {
			Some("12") => "cuda12",
			Some("13") => "cuda13",
			_ => {
				if let Some(cuda_home) = vars::get("CUDA_HOME")
					&& (cuda_home.contains("v13.") || cuda_home.contains("-13."))
				{
					log::debug!("detected CUDA 13 from CUDA_HOME");
					// People often have CUDA_HOME set to the CUDA dir.
					// On Windows this is usually C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
					// On Linux this is usually /usr/local/cuda-13.1
					// so detecting v13. or -13. in the path usually works
					"cuda13"
				} else if let Some(ver) = vars::get("NV_CUDA_CUDART_VERSION")
					&& ver.starts_with("13.")
				{
					log::debug!("detected CUDA 13 from NV_CUDA_CUDART_VERSION");
					// Set by NVIDIA docker images (both devel & runtime)
					"cuda13"
				} else if let Ok(output) = Command::new("nvcc").arg("--version").output()
					&& let Ok(stdout) = str::from_utf8(&output.stdout)
					&& stdout.contains("Build cuda_13")
				{
					log::debug!("detected CUDA 13 from nvcc");
					"cuda13"
				} else {
					log::debug!("couldn't determine CUDA version, guessing 13");
					"cuda13" // "fallback" to the lowest version we ship (we only ship 13 for now)
				}
			}
		};
		feature_set.insert(cuda_feature);
	}

	let features_str = features_str(&feature_set);
	log::debug!("looking for prebuilt binaries matching feature set: {}", features_str);

	let mut candidates: Vec<_> = parse_dist_table(&target)
		.map(|c| {
			let common_features = c.features.intersection(&feature_set).count();
			(c, common_features)
		})
		.collect();
	if candidates.is_empty() {
		log::error!(
			"no prebuilt binaries available for target {target}
note: you may have to compile ONNX Runtime from source and link `ort` to your custom build; see https://ort.pyke.io/setup/linking
note: alternatively, try a different backend like `ort-tract`; see https://ort.pyke.io/backends"
		);
		return Err((features_str, Vec::new()));
	}
	// sort by best matches in descending order
	candidates.sort_by_key(|c| Reverse(c.1));

	let Some((best_dist, best_features)) = candidates
		.iter()
		// ideally find a build that matches our feature set exactly
		.find(|x| x.0.features.symmetric_difference(&feature_set).count() == 0)
		.or_else(|| candidates.first())
	else {
		unreachable!(); // we return on is_empty earlier
	};

	if *best_features == feature_set.len() {
		// perfect, we have all the features we need
		Ok(best_dist.clone())
	} else {
		Err((features_str, candidates.into_iter().map(|(d, _)| d).collect()))
	}
}
