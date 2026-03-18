use core::{
	cell::UnsafeCell,
	convert::Infallible,
	hint::spin_loop,
	marker::PhantomData,
	mem::{MaybeUninit, forget},
	ptr,
	sync::atomic::{AtomicU8, Ordering}
};

pub(crate) struct OnceLock<T> {
	data: UnsafeCell<MaybeUninit<T>>,
	status: core::sync::atomic::AtomicU8,
	phantom: PhantomData<T>
}

unsafe impl<T: Send> Send for OnceLock<T> {}
unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}

const STATUS_UNINITIALIZED: u8 = 0;
const STATUS_RUNNING: u8 = 1;
const STATUS_INITIALIZED: u8 = 2;

impl<T> OnceLock<T> {
	pub const fn new() -> Self {
		Self {
			data: UnsafeCell::new(MaybeUninit::uninit()),
			status: AtomicU8::new(STATUS_UNINITIALIZED),
			phantom: PhantomData
		}
	}

	#[inline]
	pub fn get_or_init<F: FnOnce() -> T>(&self, f: F) -> &T {
		match self.get_or_try_init(|| Ok::<T, Infallible>(f())) {
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
						status: &'a AtomicU8
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
							forget(panic_catcher);
							self.status.store(STATUS_UNINITIALIZED, Ordering::Release);
							return Err(err);
						}
					};
					unsafe {
						(*self.data.get()).as_mut_ptr().write(val);
					};
					forget(panic_catcher);

					self.status.store(STATUS_INITIALIZED, Ordering::Release);

					return Ok(unsafe { self.get_unchecked() });
				}
				Err(STATUS_INITIALIZED) => return Ok(unsafe { self.get_unchecked() }),
				Err(STATUS_RUNNING) => loop {
					match self.status.load(Ordering::Acquire) {
						STATUS_RUNNING => spin_loop(),
						STATUS_INITIALIZED => return Ok(unsafe { self.get_unchecked() }),
						// STATUS_UNINITIALIZED - running thread failed, time for us to step in
						_ => continue 'a
					}
				},
				_ => continue
			}
		}
	}

	pub fn try_insert_with<F: FnOnce() -> T>(&self, inserter: F) -> bool {
		let mut container = Some(inserter);
		self.get_or_init(|| (unsafe { container.take().unwrap_unchecked() })());
		container.is_none()
	}
}

impl<T> Drop for OnceLock<T> {
	fn drop(&mut self) {
		if *self.status.get_mut() == STATUS_INITIALIZED {
			unsafe {
				ptr::drop_in_place((*self.data.get()).as_mut_ptr());
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::OnceLock;

	#[test]
	fn test_once() {
		let once = OnceLock::new();
		let mut called = 0;
		once.get_or_init(|| called += 1);
		assert_eq!(called, 1);
		once.get_or_init(|| called += 1);
		assert_eq!(called, 1);

		assert!(!once.try_insert_with(|| called += 1));
		assert_eq!(called, 1);
	}
}
