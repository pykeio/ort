use std::{
	alloc::{self, Layout},
	arch::wasm32,
	ffi::c_void,
	ptr, slice, str
};

macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

pub mod fmt_shims {
	use super::*;

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

pub mod libc_shims {
	use super::*;

	const _: () = assert!(std::mem::size_of::<usize>() == 4);

	unsafe fn alloc_inner(size: usize, align: usize) -> *mut u8 {
		let align = align.max(8);
		console_log!("allocating {size} bytes (align {align})");
		let ptr = alloc::alloc_zeroed(Layout::from_size_align_unchecked(size + align, align));
		ptr::copy_nonoverlapping(size.to_le_bytes().as_ptr(), ptr.add(align - 4), 4);
		ptr::copy_nonoverlapping(align.to_le_bytes().as_ptr(), ptr.add(align - 8), 4);
		ptr.add(align)
	}

	unsafe fn free_inner(ptr: *mut u8) {
		let size = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(4), 4).try_into().unwrap_unchecked());
		let align = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(8), 4).try_into().unwrap_unchecked());
		console_log!("freeing {size} bytes (align {align})");
		let layout = Layout::from_size_align_unchecked(size + align, align);
		alloc::dealloc(ptr.sub(align), layout);
	}

	#[no_mangle]
	pub unsafe extern "C" fn malloc(size: usize) -> *mut u8 {
		alloc_inner(size, 32)
	}
	#[no_mangle]
	pub unsafe extern "C" fn __libc_malloc(size: usize) -> *mut u8 {
		alloc_inner(size, 32)
	}
	#[no_mangle]
	pub unsafe extern "C" fn __libc_calloc(size: usize) -> *mut u8 {
		alloc_inner(size, 32)
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
	pub unsafe extern "C" fn posix_memalign(ptr: *mut *mut u8, size: usize, align: usize) -> i32 {
		*ptr = alloc_inner(size, align);
		0
	}

	#[no_mangle]
	pub unsafe extern "C" fn realloc(ptr: *mut u8, newsize: usize) -> *mut u8 {
		let size = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(4), 4).try_into().unwrap_unchecked());
		let align = usize::from_le_bytes(slice::from_raw_parts_mut(ptr.sub(8), 4).try_into().unwrap_unchecked());
		console_log!("reallocating {size} bytes -> {newsize} bytes (align {align})");
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
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_environ_sizes_get(argc: *mut usize, argv_buf_size: *mut usize) -> u16 {
		*argc = 0;
		*argv_buf_size = 0;
		58
	}

	#[no_mangle]
	pub unsafe extern "C" fn __wasi_environ_get(_environ: *mut *mut u8, _buf: *mut u8) -> u16 {
		58
	}

	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_seek(_fd: u32, _offset: i64, _whence: u8, _new_offset: *mut u64) -> u16 {
		58
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_write(_fd: u32, _iovs: *const (), _iovs_len: usize, _nwritten: *mut usize) -> u16 {
		58
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_read(_fd: u32, _iovs: *const (), _iovs_len: usize, _nread: *mut usize) -> u16 {
		58
	}
	#[no_mangle]
	pub unsafe extern "C" fn __wasi_fd_close(_fd: u32) -> u16 {
		58
	}
}

pub mod emscripten_shims {
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
	pub unsafe extern "C" fn emscripten_errn(str: *const u8, len: usize) {
		let c = str::from_utf8_unchecked(slice::from_raw_parts(str, len));
		eprintln!("{c}");
	}
}

#[no_mangle]
#[export_name = "_initialize"]
pub fn _initialize() {
	extern "C" {
		fn __wasm_call_ctors();
	}
	unsafe { __wasm_call_ctors() };
}
