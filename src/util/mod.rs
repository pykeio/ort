use alloc::{borrow::ToOwned, format, string::String, vec::Vec};
use core::{
	ffi::{CStr, c_char},
	mem::ManuallyDrop,
	ops::Deref,
	ptr
};

use crate::{Result, memory::Allocator};

#[cfg(feature = "std")]
#[path = "mutex_std.rs"]
mod mutex;
#[cfg(not(feature = "std"))]
#[path = "mutex_spin.rs"]
mod mutex;

mod map;
mod once_lock;
mod stack;
pub(crate) use self::{
	map::MiniMap,
	mutex::{Mutex, MutexGuard},
	once_lock::OnceLock,
	stack::*
};

/// Preloads the dynamic library at the given `path`.
///
/// Internally, some ONNX Runtime execution providers load DLLs with specific names (like `onnxruntime_providers_cuda`).
/// There is no dedicated API to configure where these libraries are loaded from; the `preload_dylib` function offers a
/// way to configure library paths without resorting to other tricks like `rpath`.
///
/// This is because most operating systems will keep a list of currently loaded dylibs in memory and fall back to
/// an already loaded dylib if its file name matches what is requested. If we explicitly load `libfoo.so` from an
/// absolute path, then any subsequent requests to load a library called `foo` will use the `libfoo.so` we already
/// loaded, instead of searching the system for a `foo` library.
///
/// See also [`crate::ep::cuda::preload_dylibs`], a helper that uses `preload_dylib` to load all required dependencies
/// of the [CUDA execution provider](crate::ep::CUDA).
///
/// ```
/// use std::env;
///
/// // Use the DirectML DLL we ship alongside the application.
/// // This needs to come before we use any other `ort` API.
/// let application_dir = current_exe().unwrap();
/// let _ = ort::util::preload_dylib(application_dir.join("DirectML.dll"));
/// ```
#[cfg_attr(docsrs, doc(cfg(any(feature = "preload-dylibs", feature = "load-dynamic"))))]
#[cfg(feature = "preload-dylibs")]
pub fn preload_dylib<P: libloading::AsFilename>(path: P) -> Result<(), libloading::Error> {
	let library = unsafe { libloading::Library::new(path) }?;
	// Do not run `FreeLibrary` so the library remains in the loaded modules list.
	let _ = ManuallyDrop::new(library);
	Ok(())
}

#[cfg(target_family = "windows")]
pub(crate) type OsCharArray = Vec<u16>;
#[cfg(not(target_family = "windows"))]
pub(crate) type OsCharArray = Vec<core::ffi::c_char>;

#[cfg(feature = "std")]
pub(crate) fn path_to_os_char(path: impl AsRef<std::path::Path>) -> OsCharArray {
	#[cfg(not(target_family = "windows"))]
	use core::ffi::c_char;
	#[cfg(target_family = "windows")]
	use std::os::windows::ffi::OsStrExt as _;

	let path = std::ffi::OsString::from(path.as_ref());
	#[cfg(target_family = "windows")]
	let path: Vec<u16> = path.encode_wide().chain(core::iter::once(0)).collect();
	#[cfg(not(target_family = "windows"))]
	let path: Vec<c_char> = path
		.as_encoded_bytes()
		.iter()
		.chain(core::iter::once(&b'\0'))
		.map(|b| *b as c_char)
		.collect();
	path
}

#[allow(unused)]
pub(crate) fn str_to_os_char(string: &str) -> OsCharArray {
	#[cfg(target_family = "windows")]
	let os_char = string.encode_utf16().chain(core::iter::once(0)).collect();
	#[cfg(not(target_family = "windows"))]
	let os_char = string
		.as_bytes()
		.iter()
		.copied()
		.chain(core::iter::once(0))
		.map(|b| b as c_char)
		.collect();
	os_char
}

#[cold]
#[inline]
#[doc(hidden)] // Used in the `ortsys!` macro, which is publicly exposed.
pub fn cold() {}

#[allow(unused)]
pub(crate) struct RunOnDrop<F: FnOnce()> {
	runner: ManuallyDrop<F>
}

impl<F: FnOnce()> Drop for RunOnDrop<F> {
	#[inline]
	fn drop(&mut self) {
		let runner = unsafe { ptr::read(&*self.runner) };
		runner()
	}
}

/// Runs the given closure at the end of the scope.
#[allow(unused)]
pub(crate) fn run_on_drop<F: FnOnce()>(f: F) -> RunOnDrop<F> {
	RunOnDrop { runner: ManuallyDrop::new(f) }
}

pub(crate) fn char_p_to_string(raw: *const c_char) -> Result<String> {
	if raw.is_null() {
		return Ok(String::new());
	}
	let c_string = unsafe { CStr::from_ptr(raw.cast_mut()).to_owned() };
	Ok(c_string.to_string_lossy().into())
}

/// A string allocated by an ONNX Runtime [`Allocator`].
pub(crate) struct AllocatedString<'a, 'p> {
	data: &'p str,
	allocator: &'a Allocator
}

static EMPTY_DANGLING: &str = "";

impl<'a, 'p> AllocatedString<'a, 'p> {
	pub unsafe fn from_ptr(ptr: *const c_char, allocator: &'a Allocator) -> Result<Self> {
		if ptr.is_null() {
			// string slices cannot have null pointers, so we use an empty string instead
			// (as a static so we can differentiate between allocated but zero-length and null pointer/non-allocated)
			return Ok(Self { data: EMPTY_DANGLING, allocator });
		}

		let c_string = unsafe { CStr::from_ptr(ptr.cast_mut()) };
		match c_string.to_str() {
			Ok(data) => Ok(Self { data, allocator }),
			Err(e) => {
				unsafe { allocator.free(ptr.cast_mut()) };
				Err(crate::Error::new(format!("string could not be converted to UTF-8: {e}")))
			}
		}
	}
}

impl<'p> Deref for AllocatedString<'_, 'p> {
	type Target = str;
	fn deref(&self) -> &Self::Target {
		self.data
	}
}

impl<'a, 'p> Drop for AllocatedString<'a, 'p> {
	fn drop(&mut self) {
		let ptr = self.data.as_ptr();
		// strings originally created from null pointers don't need freeing since nothing was allocated
		if !ptr::eq(ptr, EMPTY_DANGLING.as_ptr()) {
			unsafe { self.allocator.free(ptr.cast_mut()) };
		}
	}
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
	use alloc::{ffi::CString, sync::Arc};
	use core::ffi::CStr;
	use std::thread;

	use super::{MiniMap, Mutex, char_p_to_string, run_on_drop, with_cstr, with_cstr_ptr_array};

	#[test]
	fn test_mutex_sanity() {
		let mutex = Mutex::new(());
		for _ in 0..4 {
			drop(mutex.lock());
		}
	}

	#[test]
	fn test_mutex_threaded() {
		let mutex = Arc::new(Mutex::new(0usize));
		let threads = (0..4)
			.map(|_| {
				let mutex = Arc::clone(&mutex);
				thread::spawn(move || {
					for _ in 0..1000 {
						*mutex.lock() += 1;
					}
				})
			})
			.collect::<Vec<_>>();

		for t in threads {
			t.join().unwrap();
		}

		assert_eq!(*mutex.lock(), 4000);
	}

	#[test]
	fn test_run_on_drop() {
		let mut x = 0;
		{
			let _d = run_on_drop(|| {
				x = 42;
			});
		}
		assert_eq!(x, 42);
	}

	#[test]
	fn test_with_cstr() {
		let normal_str = "hello world";
		with_cstr(normal_str.as_bytes(), &|s| {
			assert_eq!(s.to_string_lossy(), normal_str);
			Ok(())
		})
		.unwrap();

		let longer_str = "HyperSuperUltraLongInputNameLikeAReallyLongNameSoLongInFactThatItDoesntFitOnTheStackAsSpecifiedInTheSTACK_CSTR_ARRAY_MAX_TOTAL_ConstantDefinedInUtilDotRsThisStringIsSoLongThatImStartingToRunOutOfThingsToSaySoIllJustPutZeros000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000hi0000000000000000000000000000000000000000000000000000000000000000000000000000000";
		with_cstr(longer_str.as_bytes(), &|s| {
			assert_eq!(s.to_string_lossy(), longer_str);
			Ok(())
		})
		.unwrap();
	}

	#[test]
	fn test_with_cstr_arr() {
		let normal_str = "hello world";
		let few_normal_strs = vec![normal_str; 4];
		with_cstr_ptr_array(&few_normal_strs, &|arr| {
			for &ptr in arr {
				let cstr = unsafe { CStr::from_ptr(ptr) };
				assert_eq!(cstr.to_string_lossy(), normal_str);
			}
			Ok(())
		})
		.unwrap();

		let many_normal_strs = vec![normal_str; 1000];
		with_cstr_ptr_array(&many_normal_strs, &|arr| {
			for &ptr in arr {
				let cstr = unsafe { CStr::from_ptr(ptr) };
				assert_eq!(cstr.to_string_lossy(), normal_str);
			}
			Ok(())
		})
		.unwrap();

		let longer_str = "HyperSuperUltraLongInputNameLikeAReallyLongNameSoLongInFactThatItDoesntFitOnTheStackAsSpecifiedInTheSTACK_CSTR_ARRAY_MAX_TOTAL_ConstantDefinedInUtilDotRsThisStringIsSoLongThatImStartingToRunOutOfThingsToSaySoIllJustPutZeros000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000hi0000000000000000000000000000000000000000000000000000000000000000000000000000000";
		let mut mixed_strs = vec![normal_str; 3];
		mixed_strs[1] = longer_str;
		with_cstr_ptr_array(&mixed_strs, &|arr| {
			for (i, &ptr) in arr.iter().enumerate() {
				let cstr = unsafe { CStr::from_ptr(ptr) };
				if i != 1 {
					assert_eq!(cstr.to_string_lossy(), normal_str);
				} else {
					assert_eq!(cstr.to_string_lossy(), longer_str);
				}
			}
			Ok(())
		})
		.unwrap();

		let many_longer_strs = vec![longer_str; 200];
		with_cstr_ptr_array(&many_longer_strs, &|arr| {
			for &ptr in arr {
				let cstr = unsafe { CStr::from_ptr(ptr) };
				assert_eq!(cstr.to_string_lossy(), longer_str);
			}
			Ok(())
		})
		.unwrap();
	}

	#[test]
	fn test_mini_map() {
		let mut map = MiniMap::<&'static str, u32>::new();
		map.insert("meaning", 42);
		assert_eq!(map.get("meaning"), Some(&42));
		assert_eq!(map.get("nothing"), None);

		for (k, v) in map.iter() {
			assert_eq!(*k, "meaning");
			assert_eq!(*v, 42);
		}

		map.insert("meaning", 24);
		map.insert("other", 21);
		assert_eq!(map.get("meaning"), Some(&24));
		assert_eq!(map.get("other"), Some(&21));
		assert_eq!(map.len(), 2);

		assert_eq!(map.drain().next(), Some(("meaning", 24)));
		assert_eq!(map.len(), 0);
	}

	#[test]
	fn test_char_p_to_string() {
		let s = CString::new("foo").unwrap_or_else(|_| unreachable!());
		let ptr = s.as_c_str().as_ptr();
		assert_eq!("foo", char_p_to_string(ptr).expect("failed to convert string"));
	}
}
