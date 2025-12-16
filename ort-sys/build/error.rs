use std::fmt;

#[derive(Debug)]
pub struct Error {
	message: String
}

impl Error {
	pub fn new(message: impl Into<String>) -> Self {
		Self { message: message.into() }
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.message)
	}
}

impl<E: std::error::Error> From<E> for Error {
	fn from(value: E) -> Self {
		Self { message: value.to_string() }
	}
}

pub trait ResultExt<T, E> {
	fn with_context<S: fmt::Display, F: FnOnce() -> S>(self, ctx: F) -> Result<T, Error>;
}

impl<T, E: fmt::Display> ResultExt<T, E> for Result<T, E> {
	fn with_context<S: fmt::Display, F: FnOnce() -> S>(self, ctx: F) -> Result<T, Error> {
		self.map_err(|e| Error::new(format!("{}: {}", ctx(), e)))
	}
}
