fn main() {
	if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
		let res = winresource::WindowsResource::new();
		res.compile().unwrap();
	}
}
