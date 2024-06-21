//! Utilities for using `ort` in WebAssembly.
//!
//! You **must** call `ort::wasm::initialize()` before using any `ort` APIs in WASM:
//! ```
//! # use ort::Session;
//! # static MODEL_BYTES: &[u8] = include_bytes!("../tests/data/upsample.ort");
//! # fn main() -> ort::Result<()> {
//! #[cfg(target_arch = "wasm32")]
//! ort::wasm::initialize();
//!
//! let session = Session::builder()?.commit_from_memory_directly(MODEL_BYTES)?;
//! # 	Ok(())
//! # }
//! ```

use std::{
	alloc::{self, Layout},
	arch::wasm32,
	ptr, slice, str
};

mod fmt_shims {
	// localized time string formatting functions
	// TODO: remove any remaining codepaths to these

	#[no_mangle]
	pub unsafe extern "C" fn strftime_l(_s: *mut u8, _l: usize, _m: *const u8, _t: *const (), _lt: *const ()) -> usize {
		unimplemented!()
	}
	#[no_mangle]
	pub unsafe extern "C" fn _tzset_js(_timezone: *mut u32, _daylight: *const i32, _name: *const u8, _dst_name: *mut u8) {
		unimplemented!()
	}
	#[no_mangle]
	pub unsafe extern "C" fn _mktime_js(_tm: *mut ()) -> ! {
		unimplemented!()
	}
	#[no_mangle]
	pub unsafe extern "C" fn _localtime_js(_time_t: i64, _tm: *mut ()) -> ! {
		unimplemented!()
	}
	#[no_mangle]
	pub unsafe extern "C" fn _gmtime_js(_time_t: i64, _tm: *mut ()) -> ! {
		unimplemented!()
	}
}

pub(crate) mod libc_shims {
	use super::*;

	// Rust, unlike C, requires us to know the exact layout of an allocation in order to deallocate it, so we need to
	// store this data at the beginning of the allocation for us to be able to pick up on deallocation:
	//
	//     ┌---- actual allocated pointer
	//     ▼
	//     +-------------+-------+------+----------------+
	//     | ...padding  | align | size |    data...     |
	//     | -align..-8  |  -8   |  -4  |    0..size     |
	//     +-------------+- -----+------+----------------+
	//                                  ▲
	//       pointer returned to C   ---┘
	//
	// This does unfortunately mean we waste a little extra memory (note that most allocators *also* store the layout
	// information in a similar manner, but we can't access it).

	const _: () = assert!(std::mem::size_of::<usize>() == 4, "32-bit pointer width (wasm32) required");

	unsafe fn alloc_inner<const ZERO: bool>(size: usize, align: usize) -> *mut u8 {
		// need enough space to store the size & alignment bytes
		let align = align.max(8);

		let layout = Layout::from_size_align_unchecked(size + align, align);
		let ptr = if ZERO { alloc::alloc_zeroed(layout) } else { alloc::alloc(layout) };
		ptr::copy_nonoverlapping(size.to_le_bytes().as_ptr(), ptr.add(align - 4), 4);
		ptr::copy_nonoverlapping(align.to_le_bytes().as_ptr(), ptr.add(align - 8), 4);
		ptr.add(align)
	}

	unsafe fn free_inner(ptr: *mut u8) {
		// something likes to free(NULL) a lot, which is valid in C (because of course it is...)
		if ptr.is_null() {
			return;
		}

		let size = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(4), 4).try_into().unwrap_unchecked());
		let align = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(8), 4).try_into().unwrap_unchecked());
		let layout = Layout::from_size_align_unchecked(size + align, align);
		alloc::dealloc(ptr.sub(align), layout);
	}

	const DEFAULT_ALIGNMENT: usize = 32;

	#[no_mangle]
	pub unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
		alloc_inner::<false>(size, DEFAULT_ALIGNMENT)
	}
	#[no_mangle]
	pub unsafe extern "C" fn __libc_malloc(size: usize) -> *mut u8 {
		alloc_inner::<false>(size, DEFAULT_ALIGNMENT)
	}
	#[no_mangle]
	pub unsafe extern "C" fn __libc_calloc(n: usize, size: usize) -> *mut u8 {
		alloc_inner::<true>(size * n, DEFAULT_ALIGNMENT)
	}
	#[no_mangle]
	pub unsafe extern "C" fn free(ptr: *mut u8) {
		free_inner(ptr)
	}
	#[no_mangle]
	pub unsafe extern "C" fn __libc_free(ptr: *mut u8) {
		free_inner(ptr)
	}

	#[no_mangle]
	pub unsafe extern "C" fn posix_memalign(ptr: *mut *mut u8, align: usize, size: usize) -> i32 {
		*ptr = alloc_inner::<false>(size, align);
		0
	}

	#[no_mangle]
	pub unsafe extern "C" fn realloc(ptr: *mut u8, newsize: usize) -> *mut u8 {
		let size = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(4), 4).try_into().unwrap_unchecked());
		let align = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(8), 4).try_into().unwrap_unchecked());
		let layout = Layout::from_size_align_unchecked(size + align, align);
		let ptr = alloc::realloc(ptr.sub(align), layout, newsize);
		ptr::copy_nonoverlapping(size.to_le_bytes().as_ptr(), ptr.add(align - 4), 4);
		ptr.add(align)
	}

	#[no_mangle]
	pub unsafe extern "C" fn abort() -> ! {
		std::process::abort()
	}
}

#[cfg(not(target_os = "wasi"))]
mod wasi_shims {
	#[allow(non_camel_case_types)]
	type __wasi_errno_t = u16;

	const __WASI_ENOTSUP: __wasi_errno_t = 58;

	// mock filesystem for non-WASI platforms - most of the codepaths to any FS operations should've been removed, but we
	// return ENOTSUP just to be safe

	#[no_mangle]
	pub unsafe extern "C" fn __wasi_environ_sizes_get(argc: *mut usize, argv_buf_size: *mut usize) -> __wasi_errno_t {
		*argc = 0;
		*argv_buf_size = 0;
		__WASI_ENOTSUP
	}

	#[no_mangle]
	pub unsafe extern "C" fn __wasi_environ_get(_environ: *mut *mut u8, _buf: *mut u8) -> __wasi_errno_t {
		__WASI_ENOTSUP
	}

	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_seek(_fd: u32, _offset: i64, _whence: u8, _new_offset: *mut u64) -> __wasi_errno_t {
		__WASI_ENOTSUP
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_write(_fd: u32, _iovs: *const (), _iovs_len: usize, _nwritten: *mut usize) -> __wasi_errno_t {
		__WASI_ENOTSUP
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_read(_fd: u32, _iovs: *const (), _iovs_len: usize, _nread: *mut usize) -> __wasi_errno_t {
		__WASI_ENOTSUP
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_close(_fd: u32) -> __wasi_errno_t {
		__WASI_ENOTSUP
	}
}

mod emscripten_shims {
	use super::*;

	#[no_mangle]
	pub unsafe extern "C" fn emscripten_memcpy_js(dst: *mut (), src: *const (), n: usize) {
		std::ptr::copy_nonoverlapping(src, dst, n)
	}

	#[no_mangle]
	pub unsafe extern "C" fn emscripten_get_now() -> f64 {
		js_sys::Date::now()
	}

	#[no_mangle]
	pub unsafe extern "C" fn emscripten_get_heap_max() -> usize {
		wasm32::memory_size(0) << 16
	}

	#[no_mangle]
	pub unsafe extern "C" fn emscripten_date_now() -> f64 {
		js_sys::Date::now()
	}

	#[no_mangle]
	pub unsafe extern "C" fn _emscripten_get_now_is_monotonic() -> i32 {
		0
	}

	#[no_mangle]
	pub unsafe extern "C" fn emscripten_builtin_malloc(size: usize) -> *mut u8 {
		alloc::alloc_zeroed(Layout::from_size_align_unchecked(size, 32))
	}

	#[no_mangle]
	#[tracing::instrument]
	pub unsafe extern "C" fn emscripten_errn(str: *const u8, len: usize) {
		let c = str::from_utf8_unchecked(slice::from_raw_parts(str, len));
		tracing::error!("Emscripten error: {c}");
	}

	// despite disabling exceptions literally everywhere when compiling, we still have to stub this...
	#[no_mangle]
	pub unsafe extern "C" fn __cxa_throw(_ptr: *const (), _type: *const (), _destructor: *const ()) -> ! {
		std::process::abort();
	}
}

#[no_mangle]
#[export_name = "_initialize"]
pub fn initialize() {
	// The presence of an `_initialize` function prevents the linker from calling `__wasm_call_ctors` at the top of every
	// function - including the functions `wasm-bindgen` interprets to generate JS glue code. `__wasm_call_ctors` calls
	// complex functions that wbg's interpreter isn't equipped to handle, which was preventing wbg from outputting
	// anything.
	// I'm not entirely sure what `__wasm_call_ctors` is initializing, but it seems to have something to do with C++
	// vtables, and it's crucial for proper operation.
	extern "C" {
		fn __wasm_call_ctors();
	}
	unsafe { __wasm_call_ctors() };
}
