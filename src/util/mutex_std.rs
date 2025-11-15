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
