use std::env;

use crate::log;

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
	if cfg!(feature = "nvrtx") {
		feature_set.push("nvrtx");
	}
	if cfg!(any(feature = "cuda", feature = "tensorrt")) {
		feature_set.push("cu12");
	}
	if cfg!(feature = "rocm") {
		feature_set.push("rocm");
	}

	let feature_set = if !feature_set.is_empty() { feature_set.join(",") } else { "none".to_owned() };
	log::debug!("looking for prebuilt binaries matching feature set: {feature_set}");

	let mut dist = find_dist(&target, &feature_set);
	if dist.is_none() && feature_set != "none" {
		log::warning!("full feature set '{feature_set}' not available; seeing if we can download with no features instead");
		// i dont like this behavior at all but the only thing i like less than it is rust-analyzer breaking because it
		// ***insists*** on enabling --all-features
		dist = find_dist(&target, "none");
	}

	match dist {
		Some(dist) => Ok(dist),
		None => Err(if feature_set != "none" { Some(feature_set) } else { None })
	}
}
