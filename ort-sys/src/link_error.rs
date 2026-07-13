unsafe extern "C" {
	#[link_name = "\n\n!!! The ort-sys crate could not link to ONNX Runtime because:
	- `libonnxruntime` is not configured via `pkg-config`, or the `pkg-config` feature is not enabled
	- ort-sys was instructed not to download prebuilt binaries (`cargo build --offline`), or the `download-binaries` feature is not enabled
	- Neither `ORT_LIB_PATH` or `ORT_IOS_XCFWK_PATH` (for iOS) were set to link to custom binaries

To rectify this:
	- Compile ONNX Runtime from source and manually configure linking (see https://ort.pyke.io/setup/linking for more information)
	- Enable the `download-binaries` feature if the target is supported
	- Enable ort's `alternative-backend` feature if you intend to use a different backend (or ort-sys' `disable-linking` feature if you use this crate directly)\n"]
	#[cfg(link_error_generic)]
	fn generic() -> !;

	#[link_name = concat!("\n\n!!! The ort-sys crate did not download prebuilt binaries because there are no builds available that satisfy the requested feature set '", env!("ORT_FEATURE_SET"), "'.

Builds with these feature sets are available for the current target (separated by ;):
", env!("ORT_AVAILABLE_DISTS"), "

You can enable the `lax-feature-matching` Cargo feature to have ort pick the best fit (marked above with *) if an exact match can't be found, but some EPs or features you requested won't be available.

To completely satisfy your feature set, you'll need to compile ONNX Runtime from source with those features and manually configure linking (see https://ort.pyke.io/setup/linking for more information)\n")]
	#[cfg(link_error_bad_dist_features)]
	fn bad_dist_features() -> !;
}

#[cfg(link_error_generic)]
#[used]
static _TRIGGER_GENERIC: unsafe extern "C" fn() -> ! = generic;
#[cfg(link_error_bad_dist_features)]
#[used]
static _TRIGGER_BAD_DIST_FEATURES: unsafe extern "C" fn() -> ! = bad_dist_features;
