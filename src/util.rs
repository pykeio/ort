#[cfg(not(target_family = "windows"))]
use std::os::raw::c_char;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(target_family = "windows")]
use std::os::windows::ffi::OsStrExt;
use std::{ffi::OsString, path::Path};

#[cfg(target_family = "windows")]
type OsCharArray = Vec<u16>;
#[cfg(not(target_family = "windows"))]
type OsCharArray = Vec<c_char>;

pub fn path_to_os_char(path: impl AsRef<Path>) -> OsCharArray {
	let model_path = OsString::from(path.as_ref());
	#[cfg(target_family = "windows")]
	let model_path: Vec<u16> = model_path.encode_wide().chain(std::iter::once(0)).collect();
	#[cfg(not(target_family = "windows"))]
	let model_path: Vec<c_char> = model_path
		.as_encoded_bytes()
		.iter()
		.chain(std::iter::once(&b'\0'))
		.map(|b| *b as c_char)
		.collect();
	model_path
}
