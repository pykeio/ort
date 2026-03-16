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
	pub const fn new(data: T) -> Self {
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

#[cfg(test)]
mod tests {
	use alloc::sync::Arc;
	use std::thread;

	use super::Mutex;

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
			t.join().expect("");
		}

		assert_eq!(*mutex.lock(), 4000);
	}
}
