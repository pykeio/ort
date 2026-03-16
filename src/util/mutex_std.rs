use std::sync::Mutex as StdMutex;
pub use std::sync::MutexGuard;

#[repr(transparent)]
pub struct Mutex<T>(StdMutex<T>);

impl<T> Mutex<T> {
	pub const fn new(data: T) -> Self {
		Self(StdMutex::new(data))
	}

	pub fn lock(&self) -> MutexGuard<'_, T> {
		match self.0.lock() {
			Ok(guard) => guard,
			Err(_) => panic!("Mutex poisoned")
		}
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
