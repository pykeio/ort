use alloc::{
	ffi::{CString, NulError},
	format,
	vec::Vec
};
#[cfg(not(feature = "std"))]
use core::sync::atomic::Ordering;
use core::{
	borrow::Borrow,
	cell::UnsafeCell,
	ffi::{CStr, c_char},
	fmt,
	marker::PhantomData,
	mem::{self, ManuallyDrop, MaybeUninit},
	ops::Deref,
	ptr, slice
};

use smallvec::SmallVec;

use crate::{Result, memory::Allocator};

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

#[cfg(target_family = "windows")]
type OsCharArray = Vec<u16>;
#[cfg(not(target_family = "windows"))]
type OsCharArray = Vec<core::ffi::c_char>;

#[cfg(feature = "std")]
pub(crate) fn path_to_os_char(path: impl AsRef<std::path::Path>) -> OsCharArray {
	#[cfg(not(target_family = "windows"))]
	use core::ffi::c_char;
	#[cfg(unix)]
	use std::os::unix::ffi::OsStrExt;
	#[cfg(target_family = "windows")]
	use std::os::windows::ffi::OsStrExt;

	let path = std::ffi::OsString::from(path.as_ref());
	#[cfg(target_family = "windows")]
	let path: Vec<u16> = path.encode_wide().chain(std::iter::once(0)).collect();
	#[cfg(not(target_family = "windows"))]
	let path: Vec<c_char> = path.as_bytes().iter().chain(std::iter::once(&b'\0')).map(|b| *b as c_char).collect();
	path
}

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

// generally as performant or faster than HashMap<K, V> for <50 items. good enough for #[no_std]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct MiniMap<K, V> {
	values: Vec<(K, V)>
}

impl<K, V> Default for MiniMap<K, V> {
	fn default() -> Self {
		Self { values: Vec::new() }
	}
}

impl<K, V> MiniMap<K, V> {
	pub const fn new() -> Self {
		Self { values: Vec::new() }
	}
}

impl<K: Eq, V> MiniMap<K, V> {
	pub fn get<Q>(&self, key: &Q) -> Option<&V>
	where
		K: Borrow<Q>,
		Q: Eq + ?Sized
	{
		self.values.iter().find(|(k, _)| key.eq(k.borrow())).map(|(_, v)| v)
	}

	pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
	where
		K: Borrow<Q>,
		Q: Eq + ?Sized
	{
		self.values.iter_mut().find(|(k, _)| key.eq(k.borrow())).map(|(_, v)| v)
	}

	pub fn insert(&mut self, key: K, value: V) -> Option<V> {
		match self.get_mut(&key) {
			Some(v) => Some(mem::replace(v, value)),
			None => {
				self.values.push((key, value));
				None
			}
		}
	}

	pub fn drain(&mut self) -> alloc::vec::Drain<(K, V)> {
		self.values.drain(..)
	}

	pub fn len(&self) -> usize {
		self.values.len()
	}

	pub fn iter(&self) -> core::slice::Iter<'_, (K, V)> {
		self.values.iter()
	}
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for MiniMap<K, V> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_map().entries(self.values.iter().map(|(k, v)| (k, v))).finish()
	}
}

pub(crate) struct OnceLock<T> {
	data: UnsafeCell<MaybeUninit<T>>,
	#[cfg(not(feature = "std"))]
	status: core::sync::atomic::AtomicU8,
	#[cfg(feature = "std")]
	once: std::sync::Once,
	phantom: PhantomData<T>
}

unsafe impl<T: Send> Send for OnceLock<T> {}
unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}

#[cfg(not(feature = "std"))]
const STATUS_UNINITIALIZED: u8 = 0;
#[cfg(not(feature = "std"))]
const STATUS_RUNNING: u8 = 1;
#[cfg(not(feature = "std"))]
const STATUS_INITIALIZED: u8 = 2;

#[cfg(not(feature = "std"))]
impl<T> OnceLock<T> {
	pub const fn new() -> Self {
		Self {
			data: UnsafeCell::new(MaybeUninit::uninit()),
			status: core::sync::atomic::AtomicU8::new(STATUS_UNINITIALIZED),
			phantom: PhantomData
		}
	}

	#[inline]
	pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
		match self.get_or_try_init(|| Ok::<T, core::convert::Infallible>(f())) {
			Ok(x) => x,
			Err(e) => match e {}
		}
	}

	#[inline]
	pub fn get(&self) -> Option<&T> {
		match self.status.load(Ordering::Acquire) {
			STATUS_INITIALIZED => Some(unsafe { self.get_unchecked() }),
			_ => None
		}
	}

	#[inline]
	pub unsafe fn get_unchecked(&self) -> &T {
		&*(*self.data.get()).as_ptr()
	}

	#[inline]
	pub fn get_or_try_init<F: FnOnce() -> Result<T, E>, E>(&self, f: F) -> Result<&T, E> {
		if let Some(value) = self.get() { Ok(value) } else { self.try_init_inner(f) }
	}

	#[cold]
	fn try_init_inner<F: FnOnce() -> Result<T, E>, E>(&self, f: F) -> Result<&T, E> {
		'a: loop {
			match self
				.status
				.compare_exchange(STATUS_UNINITIALIZED, STATUS_RUNNING, Ordering::Acquire, Ordering::Acquire)
			{
				Ok(_) => {
					struct SetStatusOnPanic<'a> {
						status: &'a core::sync::atomic::AtomicU8
					}
					impl Drop for SetStatusOnPanic<'_> {
						fn drop(&mut self) {
							self.status.store(STATUS_UNINITIALIZED, Ordering::SeqCst);
						}
					}

					let panic_catcher = SetStatusOnPanic { status: &self.status };
					let val = match f() {
						Ok(val) => val,
						Err(err) => {
							core::mem::forget(panic_catcher);
							self.status.store(STATUS_UNINITIALIZED, Ordering::Release);
							return Err(err);
						}
					};
					unsafe {
						*(*self.data.get()).as_mut_ptr() = val;
					};
					core::mem::forget(panic_catcher);

					self.status.store(STATUS_INITIALIZED, Ordering::Release);

					return Ok(unsafe { self.get_unchecked() });
				}
				Err(STATUS_INITIALIZED) => return Ok(unsafe { self.get_unchecked() }),
				Err(STATUS_RUNNING) => loop {
					match self.status.load(Ordering::Acquire) {
						STATUS_RUNNING => core::hint::spin_loop(),
						STATUS_INITIALIZED => return Ok(unsafe { self.get_unchecked() }),
						// STATUS_UNINITIALIZED - running thread failed, time for us to step in
						_ => continue 'a
					}
				},
				_ => continue
			}
		}
	}
}

#[cfg(feature = "std")]
impl<T> OnceLock<T> {
	pub const fn new() -> Self {
		Self {
			data: UnsafeCell::new(MaybeUninit::uninit()),
			once: std::sync::Once::new(),
			phantom: PhantomData
		}
	}

	#[inline]
	pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
		match self.get_or_try_init(|| Ok::<T, core::convert::Infallible>(f())) {
			Ok(x) => x,
			Err(e) => match e {}
		}
	}

	#[inline]
	pub fn get(&self) -> Option<&T> {
		if self.once.is_completed() { Some(unsafe { self.get_unchecked() }) } else { None }
	}

	#[inline]
	pub unsafe fn get_unchecked(&self) -> &T {
		unsafe { &*(*self.data.get()).as_ptr() }
	}

	#[inline]
	pub fn get_or_try_init<F: FnOnce() -> Result<T, E>, E>(&self, f: F) -> Result<&T, E> {
		if let Some(value) = self.get() { Ok(value) } else { self.try_init_inner(f) }
	}

	#[cold]
	fn try_init_inner<F: FnOnce() -> Result<T, E>, E>(&self, f: F) -> Result<&T, E> {
		let mut res: Result<(), E> = Ok(());
		let slot = &self.data;
		self.once.call_once_force(|_| match f() {
			Ok(value) => unsafe {
				(*slot.get()).write(value);
			},
			Err(e) => {
				res = Err(e);
			}
		});
		res.map(|_| unsafe { self.get_unchecked() })
	}
}

impl<T> OnceLock<T> {
	pub fn try_insert_with<F: FnOnce() -> T>(&self, inserter: F) -> bool {
		let mut container = Some(inserter);
		self.get_or_init(|| (unsafe { container.take().unwrap_unchecked() })());
		container.is_none()
	}

	pub fn try_insert_with_fallible<F: FnOnce() -> Result<T, E>, E>(&self, inserter: F) -> Result<bool, E> {
		let mut container = Some(inserter);
		self.get_or_try_init(|| (unsafe { container.take().unwrap_unchecked() })())?;
		Ok(container.is_none())
	}
}

impl<T> Drop for OnceLock<T> {
	fn drop(&mut self) {
		#[cfg(not(feature = "std"))]
		let status = *self.status.get_mut() == STATUS_INITIALIZED;
		#[cfg(feature = "std")]
		let status = self.once.is_completed();
		if status {
			unsafe {
				core::ptr::drop_in_place((*self.data.get()).as_mut_ptr());
			}
		}
	}
}

pub use self::mutex::{Mutex, MutexGuard};
#[cfg(feature = "std")]
mod mutex {
	use std::sync::Mutex as StdMutex;
	pub use std::sync::MutexGuard;

	#[repr(transparent)]
	pub struct Mutex<T>(StdMutex<T>);

	impl<T> Mutex<T> {
		pub fn new(data: T) -> Self {
			Self(StdMutex::new(data))
		}

		pub fn lock(&self) -> MutexGuard<'_, T> {
			match self.0.lock() {
				Ok(guard) => guard,
				// ignore poison error
				Err(p) => p.into_inner()
			}
		}
	}
}
#[cfg(not(feature = "std"))]
mod mutex {
	use core::{
		cell::UnsafeCell,
		ops::{Deref, DerefMut},
		sync::atomic::{AtomicBool, Ordering}
	};

	pub struct Mutex<T> {
		is_locked: AtomicBool,
		data: UnsafeCell<T>
	}

	unsafe impl<T: Send> Send for Mutex<T> {}
	unsafe impl<T: Send> Sync for Mutex<T> {}

	impl<T> Mutex<T> {
		pub fn new(data: T) -> Self {
			Mutex {
				is_locked: AtomicBool::new(false),
				data: UnsafeCell::new(data)
			}
		}

		pub fn lock(&self) -> MutexGuard<'_, T> {
			loop {
				if self
					.is_locked
					.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
					.is_ok()
				{
					return MutexGuard {
						is_locked: &self.is_locked,
						data: unsafe { &mut *self.data.get() }
					};
				}

				while self.is_locked.load(Ordering::Relaxed) {
					core::hint::spin_loop();
				}
			}
		}
	}

	pub struct MutexGuard<'a, T: 'a> {
		is_locked: &'a AtomicBool,
		data: *mut T
	}

	unsafe impl<T: Send> Send for MutexGuard<'_, T> {}
	unsafe impl<T: Sync> Sync for MutexGuard<'_, T> {}

	impl<T> Deref for MutexGuard<'_, T> {
		type Target = T;

		fn deref(&self) -> &Self::Target {
			unsafe { &*self.data }
		}
	}

	impl<T> DerefMut for MutexGuard<'_, T> {
		fn deref_mut(&mut self) -> &mut Self::Target {
			unsafe { &mut *self.data }
		}
	}

	impl<T> Drop for MutexGuard<'_, T> {
		fn drop(&mut self) {
			self.is_locked.store(false, Ordering::Release);
		}
	}
}

#[cold]
#[inline]
#[doc(hidden)]
pub fn cold() {}

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
		let strings: SmallVec<*const c_char, STACK_CSTR_ARRAY_MAX_ELEMENTS> = strings
			.iter()
			.map(|s| CString::new(s.as_ref()).map(|s| s.into_raw().cast_const()))
			.collect::<Result<SmallVec<*const c_char, STACK_CSTR_ARRAY_MAX_ELEMENTS>, NulError>>()?;
		let res = f(&strings);
		for string in strings {
			drop(unsafe { CString::from_raw(string.cast_mut()) });
		}
		res
	}

	fn run_with_stack_cstr_array<T: AsRef<str>, R>(strings: &[T], f: &dyn Fn(&[*const c_char]) -> Result<R>) -> Result<R> {
		let mut buf = MaybeUninit::<[c_char; STACK_CSTR_ARRAY_MAX_TOTAL]>::uninit();
		let mut buf_ptr = buf.as_mut_ptr() as *mut c_char;

		let strings: SmallVec<*const c_char, STACK_CSTR_ARRAY_MAX_ELEMENTS> = strings
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

#[allow(unused)]
pub(crate) fn run_on_drop<F: FnOnce()>(f: F) -> RunOnDrop<F> {
	RunOnDrop { runner: ManuallyDrop::new(f) }
}

#[cfg(feature = "load-dynamic")]
pub fn preload_dylib<P: AsRef<std::ffi::OsStr>>(path: P) -> Result<(), libloading::Error> {
	let library = unsafe { libloading::Library::new(path) }?;
	// Do not run `FreeLibrary` so the library remains in the loaded modules list.
	mem::forget(library);
	Ok(())
}

pub(crate) struct AllocatedString<'a, 'p> {
	data: &'p str,
	allocator: &'a Allocator
}

static EMPTY_DANGLING: &str = "";

impl<'a, 'p> AllocatedString<'a, 'p> {
	pub unsafe fn from_ptr(ptr: *const c_char, allocator: &'a Allocator) -> Result<Self> {
		if ptr.is_null() {
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
		if !ptr::eq(ptr, EMPTY_DANGLING.as_ptr()) {
			unsafe { self.allocator.free(ptr.cast_mut()) };
		}
	}
}
