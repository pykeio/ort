use alloc::{
	boxed::Box,
	format,
	string::{String, ToString}
};
use core::{
	convert::Infallible,
	error::Error as CoreError,
	ffi::c_char,
	fmt,
	ptr::{self, NonNull}
};

use crate::{
	ortsys,
	util::{char_p_to_string, cold, with_cstr}
};

/// Type alias for the `Result` type returned by `ort` functions.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub(crate) trait IntoStatus {
	fn into_status(self) -> ort_sys::OrtStatusPtr;
}

impl<T> IntoStatus for Result<T, Error> {
	fn into_status(self) -> ort_sys::OrtStatusPtr {
		let (code, message) = match &self {
			Ok(_) => return ort_sys::OrtStatusPtr(ptr::null_mut()),
			Err(e) => (ort_sys::OrtErrorCode::ORT_FAIL, e.to_string())
		};
		with_cstr(message.as_bytes(), &|message| Ok(ortsys![unsafe CreateStatus(code, message.as_ptr())])).expect("invalid error message")
	}
}

struct ErrorInternal {
	code: ErrorCode,
	message: String,
	cause: Option<Box<dyn CoreError + Send + Sync + 'static>>,
	status_ptr: NonNull<ort_sys::OrtStatus>
}

unsafe impl Send for ErrorInternal {}
unsafe impl Sync for ErrorInternal {}

impl ErrorInternal {
	#[cold]
	pub(crate) unsafe fn from_ptr(ptr: NonNull<ort_sys::OrtStatus>) -> Self {
		let code = ErrorCode::from(ortsys![unsafe GetErrorCode(ptr.as_ptr())]);
		let raw: *const c_char = ortsys![unsafe GetErrorMessage(ptr.as_ptr())];
		match char_p_to_string(raw) {
			Ok(message) => ErrorInternal {
				code,
				message,
				cause: None,
				status_ptr: ptr
			},
			Err(err) => ErrorInternal {
				code,
				message: format!("(failed to convert UTF-8: {err})"),
				cause: None,
				status_ptr: ptr
			}
		}
	}
}

impl Drop for ErrorInternal {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseStatus(self.status_ptr.as_ptr())];
	}
}

/// An error returned by any `ort` API.
pub struct Error<R = ()> {
	recover: R,
	inner: Box<ErrorInternal>
}

impl<R> fmt::Debug for Error<R> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Error")
			.field("code", &self.inner.code)
			.field("message", &self.message())
			.field("ptr", &self.inner.status_ptr.as_ptr())
			.finish()
	}
}

impl Error<()> {
	/// Converts an [`ort_sys::OrtStatusPtr`] to a [`Result`].
	///
	/// This takes ownership of the status pointer.
	///
	/// # Safety
	/// `ptr` must be a valid `OrtStatusPtr` returned from an `ort-sys` API.
	#[inline]
	pub unsafe fn result_from_status(ptr: ort_sys::OrtStatusPtr) -> Result<(), Self> {
		match NonNull::new(ptr.0) {
			None => Ok(()),
			Some(ptr) => {
				cold();

				Err(Self {
					recover: (),
					inner: Box::new(unsafe { ErrorInternal::from_ptr(ptr) })
				})
			}
		}
	}

	/// Wrap a custom, user-provided error in an [`ort::Error`](Error).
	///
	/// This can be used to return custom errors from e.g. training dataloaders or custom operators if a non-`ort`
	/// related operation fails.
	pub fn wrap<T: CoreError + Send + Sync + 'static>(err: T) -> Self {
		Self::new_internal(ErrorCode::GenericFailure, err.to_string(), Some(Box::new(err)))
	}

	/// Creates a custom [`Error`] with the given message.
	pub fn new(msg: impl Into<String>) -> Self {
		Self::new_internal(ErrorCode::GenericFailure, msg, None)
	}

	/// Creates a custom [`Error`] with the given [`ErrorCode`] and message.
	pub fn new_with_code(code: ErrorCode, msg: impl Into<String>) -> Self {
		Self::new_internal(code, msg, None)
	}

	fn new_internal(code: ErrorCode, message: impl Into<String>, cause: Option<Box<dyn CoreError + Send + Sync + 'static>>) -> Self {
		let message = message.into();
		let ptr = with_cstr(message.as_bytes(), &|message| Ok(ortsys![unsafe CreateStatus(code.into(), message.as_ptr())])).expect("invalid error message");
		Self {
			recover: (),
			inner: Box::new(ErrorInternal {
				code,
				message,
				cause,
				status_ptr: unsafe { NonNull::new_unchecked(ptr.0) }
			})
		}
	}

	pub(crate) fn with_recover<R>(self, recover: R) -> Error<R> {
		Error { recover, inner: self.inner }
	}
}

impl<R> Error<R> {
	pub fn code(&self) -> ErrorCode {
		self.inner.code
	}

	pub fn message(&self) -> &str {
		self.inner.message.as_str()
	}
}

impl<R: Sized> Error<R> {
	/// Recovers from this error.
	///
	/// ```
	/// # use ort::session::{builder::GraphOptimizationLevel, Session};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::All)
	/// 	// Optimization isn't enabled in minimal builds of ONNX Runtime, so throws an error. We can just ignore it.
	/// 	.unwrap_or_else(|e| e.recover())
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn recover(self) -> R {
		self.recover
	}
}

impl<R> fmt::Display for Error<R> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.inner.message)
	}
}

impl<R> CoreError for Error<R> {
	fn source(&self) -> Option<&(dyn CoreError + 'static)> {
		self.inner.cause.as_ref().map(|x| &**x as &dyn CoreError)
	}
}

impl From<Box<dyn CoreError + Send + Sync + 'static>> for Error {
	fn from(err: Box<dyn CoreError + Send + Sync + 'static>) -> Self {
		Error::new_internal(ErrorCode::GenericFailure, err.to_string(), Some(err))
	}
}

impl From<Infallible> for Error {
	fn from(value: Infallible) -> Self {
		match value {}
	}
}

impl From<alloc::ffi::NulError> for Error {
	fn from(e: alloc::ffi::NulError) -> Self {
		Error::new(format!("Attempted to pass invalid string to C: {e}"))
	}
}

impl From<core::ffi::FromBytesWithNulError> for Error {
	fn from(e: core::ffi::FromBytesWithNulError) -> Self {
		Error::new(format!("Attempted to pass invalid string to C: {e}"))
	}
}

impl From<core::str::Utf8Error> for Error {
	fn from(e: core::str::Utf8Error) -> Self {
		Error::new(format!("C returned invalid string: {e}"))
	}
}

impl From<alloc::ffi::FromVecWithNulError> for Error {
	fn from(e: alloc::ffi::FromVecWithNulError) -> Self {
		Error::new(format!("C returned invalid string: {e}"))
	}
}

impl From<alloc::ffi::IntoStringError> for Error {
	fn from(e: alloc::ffi::IntoStringError) -> Self {
		Error::new(format!("C returned invalid string: {e}"))
	}
}

impl From<Error<crate::session::builder::SessionBuilder>> for Error<()> {
	fn from(err: Error<crate::session::builder::SessionBuilder>) -> Self {
		Self { recover: (), inner: err.inner }
	}
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorCode {
	Ok,
	GenericFailure,
	InvalidArgument,
	NoSuchFile,
	NoModel,
	EngineError,
	RuntimeException,
	InvalidProtobuf,
	ModelLoaded,
	NotImplemented,
	InvalidGraph,
	ExecutionProviderFailure
}

impl From<ort_sys::OrtErrorCode> for ErrorCode {
	fn from(value: ort_sys::OrtErrorCode) -> Self {
		match value {
			ort_sys::OrtErrorCode::ORT_OK => Self::Ok,
			ort_sys::OrtErrorCode::ORT_FAIL => Self::GenericFailure,
			ort_sys::OrtErrorCode::ORT_INVALID_ARGUMENT => Self::InvalidArgument,
			ort_sys::OrtErrorCode::ORT_NO_SUCHFILE => Self::NoSuchFile,
			ort_sys::OrtErrorCode::ORT_NO_MODEL => Self::NoModel,
			ort_sys::OrtErrorCode::ORT_ENGINE_ERROR => Self::EngineError,
			ort_sys::OrtErrorCode::ORT_RUNTIME_EXCEPTION => Self::RuntimeException,
			ort_sys::OrtErrorCode::ORT_INVALID_PROTOBUF => Self::InvalidProtobuf,
			ort_sys::OrtErrorCode::ORT_MODEL_LOADED => Self::ModelLoaded,
			ort_sys::OrtErrorCode::ORT_NOT_IMPLEMENTED => Self::NotImplemented,
			ort_sys::OrtErrorCode::ORT_INVALID_GRAPH => Self::InvalidGraph,
			ort_sys::OrtErrorCode::ORT_EP_FAIL => Self::ExecutionProviderFailure,
			#[allow(unreachable_patterns)]
			_ => Self::GenericFailure
		}
	}
}

impl From<ErrorCode> for ort_sys::OrtErrorCode {
	fn from(value: ErrorCode) -> Self {
		match value {
			ErrorCode::Ok => ort_sys::OrtErrorCode::ORT_OK,
			ErrorCode::GenericFailure => ort_sys::OrtErrorCode::ORT_FAIL,
			ErrorCode::InvalidArgument => ort_sys::OrtErrorCode::ORT_INVALID_ARGUMENT,
			ErrorCode::NoSuchFile => ort_sys::OrtErrorCode::ORT_NO_SUCHFILE,
			ErrorCode::NoModel => ort_sys::OrtErrorCode::ORT_NO_MODEL,
			ErrorCode::EngineError => ort_sys::OrtErrorCode::ORT_ENGINE_ERROR,
			ErrorCode::RuntimeException => ort_sys::OrtErrorCode::ORT_RUNTIME_EXCEPTION,
			ErrorCode::InvalidProtobuf => ort_sys::OrtErrorCode::ORT_INVALID_PROTOBUF,
			ErrorCode::ModelLoaded => ort_sys::OrtErrorCode::ORT_MODEL_LOADED,
			ErrorCode::NotImplemented => ort_sys::OrtErrorCode::ORT_NOT_IMPLEMENTED,
			ErrorCode::InvalidGraph => ort_sys::OrtErrorCode::ORT_INVALID_GRAPH,
			ErrorCode::ExecutionProviderFailure => ort_sys::OrtErrorCode::ORT_EP_FAIL
		}
	}
}
