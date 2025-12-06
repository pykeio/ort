unsafe extern "C" {
	#[link_name = "\n\nThe ort-sys crate could not link to ONNX Runtime because:
	- `libonnxruntime` is not configured via `pkg-config`
	- ort-sys was instructed not to download prebuilt binaries (`cargo build --offline`), or the `download-binaries` feature is not enabled
	- Neither `ORT_LIB_LOCATION` or `ORT_IOS_XCFWK_LOCATION` (for iOS) were set to link to custom binaries

To rectify this:
	- Compile ONNX Runtime from source and manually configure linking (see https://ort.pyke.io/setup/linking for more information)
	- Enable the `download-binaries` feature if the target is supported
	- Enable ort's `alternative-backend` feature if you intend to use a different backend (or ort-sys' `disable-linking` feature if you use this crate directly)\n"]
	fn trigger() -> !;
}

#[used]
static X: unsafe extern "C" fn() -> ! = trigger;
