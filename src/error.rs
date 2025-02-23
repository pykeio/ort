use alloc::{
	ffi::CString,
	format,
	string::{String, ToString}
};
use core::{convert::Infallible, ffi::c_char, fmt, ptr};

use crate::{char_p_to_string, ortsys};

/// Type alias for the Result type returned by ORT functions.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub(crate) trait IntoStatus {
	fn into_status(self) -> ort_sys::OrtStatusPtr;
}

impl<T> IntoStatus for Result<T, Error> {
	fn into_status(self) -> ort_sys::OrtStatusPtr {
		let (code, message) = match &self {
			Ok(_) => return ort_sys::OrtStatusPtr(ptr::null_mut()),
			Err(e) => (ort_sys::OrtErrorCode::ORT_FAIL, Some(e.to_string()))
		};
		let message = message.map(|c| CString::new(c).unwrap_or_else(|_| unreachable!()));
		// message will be copied, so this shouldn't leak
		ortsys![unsafe CreateStatus(code, message.map(|c| c.as_ptr()).unwrap_or_else(ptr::null))]
	}
}

/// An error returned by any `ort` API.
#[derive(Debug)]
pub struct Error {
	code: ErrorCode,
	msg: String
}

impl Error {
	/// Wrap a custom, user-provided error in an [`ort::Error`](Error)..
	///
	/// This can be used to return custom errors from e.g. training dataloaders or custom operators if a non-`ort`
	/// related operation fails.
	#[cfg(feature = "std")]
	pub fn wrap<T: std::error::Error + Send + Sync + 'static>(err: T) -> Self {
		Error {
			code: ErrorCode::GenericFailure,
			msg: err.to_string()
		}
	}

	/// Wrap a custom, user-provided error in an [`ort::Error`](Error)..
	///
	/// This can be used to return custom errors from e.g. training dataloaders or custom operators if a non-`ort`
	/// related operation fails.
	#[cfg(not(feature = "std"))]
	pub fn wrap<T: core::fmt::Display + Send + Sync + 'static>(err: T) -> Self {
		Error {
			code: ErrorCode::GenericFailure,
			msg: err.to_string()
		}
	}

	/// Creates a custom [`Error`] with the given message.
	pub fn new(msg: impl Into<String>) -> Self {
		Error {
			code: ErrorCode::GenericFailure,
			msg: msg.into()
		}
	}

	/// Creates a custom [`Error`] with the given [`ErrorCode`] and message.
	pub fn new_with_code(code: ErrorCode, msg: impl Into<String>) -> Self {
		Error { code, msg: msg.into() }
	}

	pub fn code(&self) -> ErrorCode {
		self.code
	}

	pub fn message(&self) -> &str {
		self.msg.as_str()
	}
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(&self.msg)
	}
}

#[cfg(feature = "std")] // sigh...
impl std::error::Error for Error {}

#[cfg(feature = "std")]
impl From<Box<dyn std::error::Error + Send + Sync + 'static>> for Error {
	fn from(err: Box<dyn std::error::Error + Send + Sync + 'static>) -> Self {
		Error {
			code: ErrorCode::GenericFailure,
			msg: err.to_string()
		}
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

pub(crate) fn assert_non_null_pointer<T>(ptr: *const T, name: &'static str) -> Result<()> {
	(!ptr.is_null())
		.then_some(())
		.ok_or_else(|| Error::new(format!("Expected pointer `{name}` to not be null")))
}

/// Converts an [`ort_sys::OrtStatus`] to a [`Result`].
///
/// Note that this frees `status`!
pub(crate) unsafe fn status_to_result(status: ort_sys::OrtStatusPtr) -> Result<(), Error> {
	let status = status.0;
	if status.is_null() {
		Ok(())
	} else {
		let code = ErrorCode::from(ortsys![unsafe GetErrorCode(status)]);
		let raw: *const c_char = ortsys![unsafe GetErrorMessage(status)];
		match char_p_to_string(raw) {
			Ok(msg) => {
				ortsys![unsafe ReleaseStatus(status)];
				Err(Error { code, msg })
			}
			Err(err) => {
				ortsys![unsafe ReleaseStatus(status)];
				Err(Error {
					code,
					msg: format!("(failed to convert UTF-8: {err})")
				})
			}
		}
	}
}
