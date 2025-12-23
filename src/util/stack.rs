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
pub(crate) fn with_cstr_ptr_array<T, R>(strings: &[T], f: &dyn Fn(&[*const c_char]) -> Result<R>) -> Result<R>
where
	T: AsRef<str>
{
	fn run_with_heap_cstr_array<T: AsRef<str>, R>(strings: &[T], f: &dyn Fn(&[*const c_char]) -> Result<R>) -> Result<R> {
		let strings: SmallVec<[*const c_char; STACK_CSTR_ARRAY_MAX_ELEMENTS]> = strings
			.iter()
			.map(|s| CString::new(s.as_ref()).map(|s| s.into_raw().cast_const()))
			.collect::<Result<SmallVec<[*const c_char; STACK_CSTR_ARRAY_MAX_ELEMENTS]>, NulError>>()?;
		let res = f(&strings);
		for string in strings {
			drop(unsafe { CString::from_raw(string.cast_mut()) });
		}
		res
	}

	fn run_with_stack_cstr_array<T: AsRef<str>, R>(strings: &[T], f: &dyn Fn(&[*const c_char]) -> Result<R>) -> Result<R> {
		let mut buf = MaybeUninit::<[c_char; STACK_CSTR_ARRAY_MAX_TOTAL]>::uninit();
		let mut buf_ptr = buf.as_mut_ptr() as *mut c_char;

		let strings: SmallVec<[*const c_char; STACK_CSTR_ARRAY_MAX_ELEMENTS]> = strings
			.iter()
			.map(|s| {
				let s = s.as_ref();
				let ptr = buf_ptr;
				unsafe {
					ptr::copy_nonoverlapping(s.as_ptr().cast::<c_char>(), buf_ptr, s.len());
					buf_ptr = buf_ptr.add(s.len());
					*buf_ptr = 0;
					buf_ptr = buf_ptr.add(1);
				};
				ptr.cast_const()
			})
			.collect();

		f(&strings)
	}

	let total_bytes = strings.iter().fold(0, |acc, s| acc + s.as_ref().len() + 1);
	if total_bytes < STACK_CSTR_ARRAY_MAX_TOTAL {
		run_with_stack_cstr_array(strings, f)
	} else {
		run_with_heap_cstr_array(strings, f)
	}
}
