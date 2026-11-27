use alloc::ffi::{CString, NulError};
use core::{
	ffi::{CStr, c_char},
	mem::MaybeUninit,
	ptr, slice
};

use smallvec::SmallVec;

use crate::Result;

// maximum number of session inputs to store on stack (~32 bytes per, + 16 bytes for run_async)
pub(crate) const STACK_SESSION_INPUTS: usize = 6;
// maximum number of session inputs to store on stack (~40 bytes per, + 16 bytes for run_async)
pub(crate) const STACK_SESSION_OUTPUTS: usize = 4;
// maximum number of EPs to store on stack in both session options and environment (24 bytes per)
pub(crate) const STACK_EXECUTION_PROVIDERS: usize = 6;
// maximum size of a single string to use stack instead of allocation in with_cstr
const STACK_CSTR_MAX: usize = 64;
// maximum size of all strings in an array to use stack instead of allocation in with_cstr_ptr_array
const STACK_CSTR_ARRAY_MAX_TOTAL: usize = 768;
// maximum number of string ptrs to keep on stack (16 bytes per)
const STACK_CSTR_ARRAY_MAX_ELEMENTS: usize = 12;

#[inline]
pub(crate) fn with_cstr<T>(bytes: &[u8], f: &dyn Fn(&CStr) -> Result<T>) -> Result<T> {
	fn run_with_heap_cstr<T>(bytes: &[u8], f: &dyn Fn(&CStr) -> Result<T>) -> Result<T> {
		let cstr = CString::new(bytes)?;
		f(&cstr)
	}

	fn run_with_stack_cstr<T>(bytes: &[u8], f: &dyn Fn(&CStr) -> Result<T>) -> Result<T> {
		let mut buf = MaybeUninit::<[u8; STACK_CSTR_MAX]>::uninit();
		let buf_ptr = buf.as_mut_ptr() as *mut u8;

		unsafe {
			ptr::copy_nonoverlapping(bytes.as_ptr(), buf_ptr, bytes.len());
			*buf_ptr.add(bytes.len()) = 0;
		};

		let cstr = CStr::from_bytes_with_nul(unsafe { slice::from_raw_parts(buf_ptr, bytes.len() + 1) })?;
		f(cstr)
	}

	if bytes.len() < STACK_CSTR_MAX {
		run_with_stack_cstr(bytes, f)
	} else {
		run_with_heap_cstr(bytes, f)
	}
}

#[inline]
pub(crate) fn with_cstr_ptr_array<T, R>(strings: &[T], f: &mut dyn FnMut(&mut [*const c_char]) -> Result<R>) -> Result<R>
where
	T: AsRef<str>
{
	let mut buf = MaybeUninit::<[c_char; STACK_CSTR_ARRAY_MAX_TOTAL]>::uninit();
	let buf_ptr = buf.as_mut_ptr() as *mut c_char;
	let mut offset = 0;

	let mut out_strings: SmallVec<[*const c_char; STACK_CSTR_ARRAY_MAX_ELEMENTS]> = SmallVec::with_capacity(strings.len());
	for s in strings {
		let s = s.as_ref();
		if s.len() < STACK_CSTR_ARRAY_MAX_TOTAL - offset {
			let start = unsafe { buf_ptr.add(offset) };
			unsafe {
				ptr::copy_nonoverlapping(s.as_ptr().cast::<c_char>(), start, s.len());
				*start.add(s.len()) = 0;
			};
			offset += s.len() + 1;
			out_strings.push(start);
		} else {
			let cstrings: SmallVec<[CString; STACK_CSTR_ARRAY_MAX_ELEMENTS]> =
				strings.iter().map(|s| CString::new(s.as_ref())).collect::<Result<_, NulError>>()?;
			let mut string_ptrs: SmallVec<[*const c_char; STACK_CSTR_ARRAY_MAX_ELEMENTS]> = cstrings.into_iter().map(|s| s.into_raw().cast_const()).collect();
			let res = f(&mut string_ptrs);
			for string in string_ptrs {
				drop(unsafe { CString::from_raw(string.cast_mut()) });
			}
			return res;
		}
	}
	f(&mut out_strings)
}
