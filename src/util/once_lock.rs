#[cfg(not(feature = "std"))]
use core::sync::atomic::Ordering;
use core::{cell::UnsafeCell, marker::PhantomData, mem::MaybeUninit};

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
		unsafe { &*(*self.data.get()).as_ptr() }
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
						(*self.data.get()).as_mut_ptr().write(val);
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
