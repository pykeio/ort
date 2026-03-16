use core::{cell::UnsafeCell, convert::Infallible, marker::PhantomData, mem::MaybeUninit, ptr};
use std::sync::Once;

pub(crate) struct OnceLock<T> {
	data: UnsafeCell<MaybeUninit<T>>,
	once: std::sync::Once,
	phantom: PhantomData<T>
}

unsafe impl<T: Send> Send for OnceLock<T> {}
unsafe impl<T: Send + Sync> Sync for OnceLock<T> {}

impl<T> OnceLock<T> {
	pub const fn new() -> Self {
		Self {
			data: UnsafeCell::new(MaybeUninit::uninit()),
			once: Once::new(),
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
		if self.once.is_completed() {
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
