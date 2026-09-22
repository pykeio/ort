// based on https://github.com/dirs-dev/dirs-sys-rs/blob/main/src/lib.rs

#![allow(unused)]

use std::path::PathBuf;

pub const PYKE_ROOT: &str = "ort.pyke.io";

#[cfg(all(target_os = "windows", target_arch = "x86"))]
macro_rules! win32_extern {
    ($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
        #[link(name = $library, kind = "raw-dylib", modifiers = "+verbatim", import_name_type = "undecorated")]
        unsafe extern $abi {
            $(#[$doc])?
            $(#[link_name=$link_name])?
            fn $($function)*;
        }
    )
}
#[cfg(all(target_os = "windows", not(target_arch = "x86")))]
macro_rules! win32_extern {
	($library:literal $abi:literal $($link_name:literal)? $(#[$doc:meta])? fn $($function:tt)*) => (
		#[link(name = $library, kind = "raw-dylib", modifiers = "+verbatim")]
		unsafe extern "C" {
			$(#[$doc])?
			$(#[link_name=$link_name])?
			fn $($function)*;
		}
	)
}

#[cfg(target_os = "windows")]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
mod windows {
	use std::{
		ffi::{OsString, c_void},
		os::windows::prelude::OsStringExt,
		path::PathBuf,
		ptr, slice
	};

	#[repr(C)]
	#[derive(Clone, Copy)]
	struct GUID {
		data1: u32,
		data2: u16,
		data3: u16,
		data4: [u8; 8]
	}

	impl GUID {
		pub const fn from_u128(uuid: u128) -> Self {
			Self {
				data1: (uuid >> 96) as u32,
				data2: ((uuid >> 80) & 0xffff) as u16,
				data3: ((uuid >> 64) & 0xffff) as u16,
				#[allow(clippy::cast_possible_truncation)]
				data4: (uuid as u64).to_be_bytes()
			}
		}
	}

	type HRESULT = i32;
	type PWSTR = *mut u16;
	type PCWSTR = *const u16;
	type HANDLE = isize;
	type KNOWN_FOLDER_FLAG = i32;

	win32_extern!("SHELL32.DLL" "system" fn SHGetKnownFolderPath(rfid: *const GUID, dwflags: KNOWN_FOLDER_FLAG, htoken: HANDLE, ppszpath: *mut PWSTR) -> HRESULT);
	win32_extern!("KERNEL32.DLL" "system" fn lstrlenW(lpstring: PCWSTR) -> i32);
	win32_extern!("OLE32.DLL" "system" fn CoTaskMemFree(pv: *const ::core::ffi::c_void) -> ());

	fn known_folder(folder_id: GUID) -> Option<PathBuf> {
		unsafe {
			let mut path_ptr: PWSTR = ptr::null_mut();
			let result = SHGetKnownFolderPath(&folder_id, 0, HANDLE::default(), &mut path_ptr);
			if result == 0 {
				let len = lstrlenW(path_ptr) as usize;
				let path = slice::from_raw_parts(path_ptr, len);
				let ostr: OsString = OsStringExt::from_wide(path);
				CoTaskMemFree(path_ptr as *const c_void);
				Some(PathBuf::from(ostr))
			} else {
				CoTaskMemFree(path_ptr as *const c_void);
				None
			}
		}
	}

	#[allow(clippy::unusual_byte_groupings)]
	const FOLDERID_LOCAL_APP_DATA: GUID = GUID::from_u128(0xf1b32785_6fba_4fcf_9d557b8e7f157091);

	#[must_use]
	pub fn known_folder_local_app_data() -> Option<PathBuf> {
		known_folder(FOLDERID_LOCAL_APP_DATA)
	}
}
#[cfg(target_os = "windows")]
#[must_use]
fn cache_dir_default() -> Option<PathBuf> {
	self::windows::known_folder_local_app_data().map(|h| h.join(PYKE_ROOT))
}

#[cfg(target_os = "linux")]
fn is_absolute_path(path: std::ffi::OsString) -> Option<PathBuf> {
	let path = PathBuf::from(path);
	if path.is_absolute() { Some(path) } else { None }
}

#[cfg(target_os = "linux")]
#[must_use]
fn cache_dir_default() -> Option<PathBuf> {
	std::env::var_os("XDG_CACHE_HOME")
		.and_then(is_absolute_path)
		.or_else(|| std::env::home_dir().map(|h| h.join(".cache").join(PYKE_ROOT)))
}

#[cfg(target_vendor = "apple")]
#[must_use]
fn cache_dir_default() -> Option<PathBuf> {
	std::env::home_dir().map(|h| h.join("Library/Caches").join(PYKE_ROOT))
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_vendor = "apple")))]
fn cache_dir_default() -> Option<PathBuf> {
	None
}

pub fn cache_dir() -> Option<PathBuf> {
	#[cfg(miri)]
	return Some(PathBuf::from("."));

	std::env::var_os("ORT_CACHE_DIR").map(PathBuf::from).or_else(cache_dir_default)
}
