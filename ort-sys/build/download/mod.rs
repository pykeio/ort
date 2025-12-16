#[cfg(not(feature = "__tls"))]
compile_error!(
	"When using `download-binaries`, a TLS feature must be configured. Enable exactly one of: \
	`tls-rustls` (uses `ring` as provider), `tls-rustls-no-provider`, `tls-native`, or `tls-native-vendored`."
);

use std::{env, time::Duration};

use ureq::{
	Agent, BodyReader, Proxy,
	config::Config as UreqConfig,
	tls::{RootCerts, TlsConfig, TlsProvider}
};

use crate::{error::Error, log, vars};

mod extract;
mod resolve;
mod verify;
pub use self::{
	extract::extract_tgz,
	resolve::resolve_dist,
	verify::{VerifyReader, bytes_to_hex_str, hex_str_to_bytes}
};

pub fn fetch_file(source_url: &str) -> Result<BodyReader<'static>, Error> {
	let tls_provider = if cfg!(feature = "tls-rustls-no-provider") {
		TlsProvider::Rustls
	} else {
		TlsProvider::NativeTls
	};
	let root_certs = if cfg!(feature = "tls-rustls-no-provider") {
		RootCerts::WebPki
	} else {
		RootCerts::PlatformVerifier
	};

	log::debug!("downloading from '{source_url}'; tls_provider={tls_provider:?}, root_certs={root_certs:?}");

	let resp = Agent::new_with_config(
		UreqConfig::builder()
			.proxy(Proxy::try_from_env())
			.max_redirects(1)
			.https_only(true)
			.tls_config(TlsConfig::builder().provider(tls_provider).root_certs(root_certs).build())
			.user_agent(format!(
				"{}/{} (host {}; for {})",
				env!("CARGO_PKG_NAME"),
				env!("CARGO_PKG_VERSION"),
				env::var("HOST").unwrap(),
				env::var("TARGET").unwrap()
			))
			.timeout_global(Some(Duration::from_secs(1800)))
			.http_status_as_error(true)
			.build()
	)
	.get(source_url)
	.call()?;

	Ok(resp.into_body()
		.into_with_config()
		.limit(1_073_741_824) // 1 GiB
		.reader())
}

pub fn should_skip() -> bool {
	vars::get("CARGO_NET_OFFLINE").as_deref() == Some("true")
		|| match vars::get(vars::SKIP_DOWNLOAD) {
			Some(val) => val == "1" || val.to_lowercase() == "true",
			None => false
		}
}
