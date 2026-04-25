#[cfg(not(feature = "__tls"))]
compile_error!(
	"When using `download-binaries`, a TLS feature must be configured. Enable exactly one of: \
	`tls-rustls` (uses `ring` as provider), `tls-rustls-no-provider`, `tls-native`, or `tls-native-vendored`."
);

use std::{env, time::Duration};

use ureq::{
	Agent, BodyReader, Proxy,
	config::Config as UreqConfig,
	http::StatusCode,
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
			.http_status_as_error(false)
			.build()
	)
	.get(source_url)
	.call()?;

	match resp.status() {
		StatusCode::OK => Ok(resp.into_body()
			.into_with_config()
			.limit(1_073_741_824) // 1 GiB
			.reader()),
		StatusCode::NOT_FOUND => Err(Error::new(
			"CDN returned 404 for the prebuilt binaries used by this version of `ort`; this is usually a temporary brownout and means you're using a version that is no longer supported and should upgrade `ort` soon. You can continue to use `ort` if you build your own ONNX Runtime binaries; see https://ort.pyke.io/setup/linking for linking instructions."
		)),
		StatusCode::GONE => Err(Error::new(
			"CDN returned 410 for the prebuilt binaries used by this version of `ort` - you're using a version of `ort` that is no longer supported and should upgrade. Though not recommended, you can continue to use `ort` if you build your own ONNX Runtime binaries; see https://ort.pyke.io/setup/linking for linking instructions."
		)),
		code @ (StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT) => {
			Err(Error::new(format!(
				"The CDN that serves prebuilt ONNX Runtime binaries for the `ort` crate is currently down (code {code}). A report at https://github.com/pykeio/ort/issues would be appreciated. You can bypass this error by compiling ONNX Runtime from source and configuring custom linking for `ort`; see https://ort.pyke.io/setup/linking"
			)))
		}
		code => Err(ureq::Error::StatusCode(code.as_u16()))?
	}
}

pub fn should_skip() -> bool {
	match vars::get_any(vars::SKIP_DOWNLOAD) {
		Some(val) => val == "1" || val.to_lowercase() == "true",
		None => false
	}
}
