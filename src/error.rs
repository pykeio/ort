//! Types and helpers for handling ORT errors.

use std::{convert::Infallible, ffi::CString, io, path::PathBuf, ptr, string};

use thiserror::Error;

use super::{char_p_to_string, ortsys, tensor::TensorElementType, ValueType};

/// Type alias for the Result type returned by ORT functions.
pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) trait IntoStatus {
	fn into_status(self) -> *mut ort_sys::OrtStatus;
}

impl<T> IntoStatus for Result<T, Error> {
	fn into_status(self) -> *mut ort_sys::OrtStatus {
		let (code, message) = match &self {
			Ok(_) => return ptr::null_mut(),
			Err(e) => (ort_sys::OrtErrorCode::ORT_FAIL, Some(e.to_string()))
		};
		let message = message.map(|c| CString::new(c).unwrap_or_else(|_| unreachable!()));
		// message will be copied, so this shouldn't leak
		ortsys![unsafe CreateStatus(code, message.map(|c| c.as_ptr()).unwrap_or_else(std::ptr::null))]
	}
}

/// An enum of all errors returned by ORT functions.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum Error {
	/// Workaround to get [`crate::inputs`] to accept `Value`s, since it will attempt to `Value::try_from` on provided
	/// values, and the implementation `TryFrom<T> for T` uses `Infallible` as the error type.
	#[error("unreachable")]
	#[doc(hidden)]
	Infallible,
	/// An error occurred when converting an FFI C string to a Rust `String`.
	#[error("Failed to construct Rust String")]
	FfiStringConversion(ErrorInternal),
	/// An error occurred while creating an ONNX environment.
	#[error("Failed to create ONNX Runtime environment: {0}")]
	CreateEnvironment(ErrorInternal),
	/// Error occurred when creating ONNX session options.
	#[error("Failed to create ONNX Runtime session options: {0}")]
	CreateSessionOptions(ErrorInternal),
	/// Error occurred when creating an allocator from a [`crate::MemoryInfo`] struct while building a session.
	#[error("Failed to create allocator from memory info: {0}")]
	CreateAllocator(ErrorInternal),
	/// Error occurred when creating an ONNX session.
	#[error("Failed to create ONNX Runtime session: {0}")]
	CreateSession(ErrorInternal),
	/// Error occurred when creating an IO binding.
	#[error("Failed to create IO binding: {0}")]
	CreateIoBinding(ErrorInternal),
	/// Error occurred when counting ONNX session input/output count.
	#[error("Failed to get input or output count: {0}")]
	GetInOutCount(ErrorInternal),
	/// Error occurred when getting ONNX input name.
	#[error("Failed to get input name: {0}")]
	GetInputName(ErrorInternal),
	/// Error occurred when getting ONNX type information
	#[error("Failed to get type info: {0}")]
	GetTypeInfo(ErrorInternal),
	/// Error occurred when getting ONNX type information
	#[error("Failed to get onnx type from type info: {0}")]
	GetOnnxTypeFromTypeInfo(ErrorInternal),
	/// Error occurred when casting ONNX type information to tensor information
	#[error("Failed to cast type info to tensor info: {0}")]
	CastTypeInfoToTensorInfo(ErrorInternal),
	/// Error occurred when casting ONNX type information to sequence type info
	#[error("Failed to cast type info to sequence type info: {0}")]
	CastTypeInfoToSequenceTypeInfo(ErrorInternal),
	/// Error occurred when casting ONNX type information to map type info
	#[error("Failed to cast type info to map typ info: {0}")]
	CastTypeInfoToMapTypeInfo(ErrorInternal),
	/// Error occurred when getting map key type
	#[error("Failed to get map key type: {0}")]
	GetMapKeyType(ErrorInternal),
	/// Error occurred when getting map value type
	#[error("Failed to get map value type: {0}")]
	GetMapValueType(ErrorInternal),
	/// Error occurred when getting sequence element type
	#[error("Failed to get sequence element type: {0}")]
	GetSequenceElementType(ErrorInternal),
	/// Error occurred when getting tensor elements type
	#[error("Failed to get tensor element type: {0}")]
	GetTensorElementType(ErrorInternal),
	/// Error occurred when getting ONNX dimensions count
	#[error("Failed to get dimensions count: {0}")]
	GetDimensionsCount(ErrorInternal),
	/// Error occurred when getting ONNX dimensions
	#[error("Failed to get dimensions: {0}")]
	GetDimensions(ErrorInternal),
	/// Error occurred when getting string length
	#[error("Failed to get string tensor length: {0}")]
	GetStringTensorDataLength(ErrorInternal),
	/// Error occurred when getting tensor element count
	#[error("Failed to get tensor element count: {0}")]
	GetTensorShapeElementCount(ErrorInternal),
	/// Error occurred when creating ONNX tensor
	#[error("Failed to create tensor: {0}")]
	CreateTensor(ErrorInternal),
	/// Error occurred when creating ONNX tensor with specific data
	#[error("Failed to create tensor with data: {0}")]
	CreateTensorWithData(ErrorInternal),
	/// Error occurred when attempting to create a [`crate::Sequence`].
	#[error("Failed to create sequence value: {0}")]
	CreateSequence(ErrorInternal),
	/// Error occurred when attempting to create a [`crate::Map`].
	#[error("Failed to create map value: {0}")]
	CreateMap(ErrorInternal),
	/// Invalid dimension when creating tensor from raw data
	#[error("Invalid dimension at {0}; all dimensions must be >= 1 when creating a tensor from raw data")]
	InvalidDimension(usize),
	/// Shape does not match data length when creating tensor from raw data
	#[error("Cannot create a tensor from raw data; shape {input:?} ({total}) is larger than the length of the data provided ({expected})")]
	TensorShapeMismatch { input: Vec<i64>, total: usize, expected: usize },
	/// Cannot create a tensor from non-contiguous array
	#[error("Cannot create this type of tensor from an array that is not in contiguous standard layout")]
	TensorDataNotContiguous,
	/// Error occurred when filling a tensor with string data
	#[error("Failed to fill string tensor: {0}")]
	FillStringTensor(ErrorInternal),
	/// Error occurred when getting tensor type and shape
	#[error("Failed to get tensor type and shape: {0}")]
	GetTensorTypeAndShape(ErrorInternal),
	/// Error occurred when ONNX inference operation was called
	#[error("Failed to run inference on model: {0}")]
	SessionRun(ErrorInternal),
	/// Error occurred when ONNX inference operation was called using `IoBinding`.
	#[error("Failed to run inference on model with IoBinding: {0}")]
	SessionRunWithIoBinding(ErrorInternal),
	/// Error occurred when extracting data from an ONNX tensor into an C array to be used as an `ndarray::ArrayView`.
	#[error("Failed to get tensor data: {0}")]
	GetTensorMutableData(ErrorInternal),
	#[error("Failed to get memory info from tensor: {0}")]
	GetTensorMemoryInfo(ErrorInternal),
	/// Error occurred when extracting string data from an ONNX tensor
	#[error("Failed to get tensor string data: {0}")]
	GetStringTensorContent(ErrorInternal),
	/// Error occurred when creating run options.
	#[error("Failed to create run options: {0}")]
	CreateRunOptions(ErrorInternal),
	/// Error occurred when terminating run options.
	#[error("Failed to terminate run options: {0}")]
	RunOptionsSetTerminate(ErrorInternal),
	/// Error occurred when unterminating run options.
	#[error("Failed to unterminate run options: {0}")]
	RunOptionsUnsetTerminate(ErrorInternal),
	/// Error occurred when setting run tag.
	#[error("Failed to set run tag: {0}")]
	RunOptionsSetTag(ErrorInternal),
	/// Error occurred when converting data to a String
	#[error("Data was not UTF-8: {0}")]
	StringFromUtf8Error(#[from] string::FromUtf8Error),
	/// Error occurred when downloading a pre-trained ONNX model from the [ONNX Model Zoo](https://github.com/onnx/models).
	#[error("Failed to download ONNX model: {0}")]
	DownloadError(#[from] FetchModelError),
	/// File does not exist
	#[error("File `{filename:?}` does not exist")]
	FileDoesNotExist {
		/// Path which does not exists
		filename: PathBuf
	},
	/// Path is invalid UTF-8
	#[error("Path `{path:?}` cannot be converted to UTF-8")]
	NonUtf8Path {
		/// Path with invalid UTF-8
		path: PathBuf
	},
	/// Attempt to build a Rust `CString` when the original string contains a null character.
	#[error("Failed to build CString when original contains null: {0}")]
	FfiStringNull(#[from] std::ffi::NulError),
	#[error("`{0}` should be a null pointer")]
	/// ORT pointer should have been null
	PointerShouldBeNull(&'static str),
	/// ORT pointer should not have been null
	#[error("`{0}` should not be a null pointer")]
	PointerShouldNotBeNull(&'static str),
	/// Could not retrieve model metadata.
	#[error("Failed to retrieve model metadata: {0}")]
	GetModelMetadata(ErrorInternal),
	/// The user tried to extract the wrong type of tensor from the underlying data
	#[error("Data type mismatch: was {actual:?}, tried to convert to {requested:?}")]
	DataTypeMismatch {
		/// The actual type of the ort output
		actual: TensorElementType,
		/// The type corresponding to the attempted conversion into a Rust type, not equal to `actual`
		requested: TensorElementType
	},
	#[error("Error trying to load symbol `{symbol}` from dynamic library: {error}")]
	DlLoad { symbol: &'static str, error: String },
	#[error("{0}")]
	ExecutionProvider(ErrorInternal),
	#[error("Execution provider `{0}` was not registered because its corresponding Cargo feature is disabled.")]
	ExecutionProviderNotRegistered(&'static str),
	#[error("Expected tensor to be on CPU in order to get data, but had allocation device `{0}`.")]
	TensorNotOnCpu(&'static str),
	#[error("Cannot extract scalar value from a {0}-dimensional tensor")]
	TensorNot0Dimensional(usize),
	#[error("Failed to create memory info: {0}")]
	CreateMemoryInfo(ErrorInternal),
	#[error("Could not get allocation device from `MemoryInfo`: {0}")]
	GetAllocationDevice(ErrorInternal),
	#[error("Failed to get available execution providers: {0}")]
	GetAvailableProviders(ErrorInternal),
	#[error("Unknown allocation device `{0}`")]
	UnknownAllocationDevice(String),
	#[error("Error when binding input: {0}")]
	BindInput(ErrorInternal),
	#[error("Error when binding output: {0}")]
	BindOutput(ErrorInternal),
	#[error("Error when retrieving session outputs from `IoBinding`: {0}")]
	GetBoundOutputs(ErrorInternal),
	#[error("Cannot use `extract_tensor` on a value that is {0:?}")]
	NotTensor(ValueType),
	#[error("Cannot use `extract_sequence` on a value that is {0:?}")]
	NotSequence(ValueType),
	#[error("Cannot use `extract_map` on a value that is {0:?}")]
	NotMap(ValueType),
	#[error("Tried to extract a map with a key type of {expected:?}, but the map has key type {actual:?}")]
	InvalidMapKeyType { expected: TensorElementType, actual: TensorElementType },
	#[error("Tried to extract a map with a value type of {expected:?}, but the map has value type {actual:?}")]
	InvalidMapValueType { expected: TensorElementType, actual: TensorElementType },
	#[error("Tried to extract a sequence with a different element type than its actual type {actual:?}")]
	InvalidSequenceElementType { actual: ValueType },
	#[error("Error occurred while attempting to extract data from sequence value: {0}")]
	ExtractSequence(ErrorInternal),
	#[error("Error occurred while attempting to extract data from map value: {0}")]
	ExtractMap(ErrorInternal),
	#[error("Failed to add custom operator to operator domain: {0}")]
	AddCustomOperator(ErrorInternal),
	#[error("Failed to create custom operator domain: {0}")]
	CreateOperatorDomain(ErrorInternal),
	#[error("Failed to add custom operator domain to session: {0}")]
	AddCustomOperatorDomain(ErrorInternal),
	#[error("Failed to create kernel context: {0}")]
	CreateKernelContext(ErrorInternal),
	#[error("Failed to get operator input: {0}")]
	GetOperatorInput(ErrorInternal),
	#[error("Failed to get operator output: {0}")]
	GetOperatorOutput(ErrorInternal),
	#[error("Failed to retrieve GPU compute stream from kernel context: {0}")]
	GetOperatorGPUComputeStream(ErrorInternal),
	#[error("{0}")]
	CustomError(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
	#[error("String tensors cannot be borrowed as mutable")]
	StringTensorNotMutable,
	#[error("Could't get `MemoryInfo` from allocator: {0}")]
	AllocatorGetInfo(ErrorInternal),
	#[error("Could't get `MemoryType` from memory info: {0}")]
	GetMemoryType(ErrorInternal),
	#[error("Could't get `AllocatorType` from memory info: {0}")]
	GetAllocatorType(ErrorInternal),
	#[error("Could't get device ID from memory info: {0}")]
	GetDeviceId(ErrorInternal)
}

impl Error {
	/// Wrap a custom, user-provided error in an [`ort::Error`](Error). The resulting error will be the
	/// [`Error::CustomError`] variant.
	///
	/// This can be used to return custom errors from e.g. training dataloaders or custom operators if a non-`ort`
	/// related operation fails.
	pub fn wrap<T: std::error::Error + Send + Sync + 'static>(err: T) -> Self {
		Error::CustomError(Box::new(err) as Box<dyn std::error::Error + Send + Sync + 'static>)
	}
}

impl From<Infallible> for Error {
	fn from(_: Infallible) -> Self {
		Error::Infallible
	}
}

/// Error details when ONNX C API returns an error.
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum ErrorInternal {
	/// Details about the error.
	#[error("{0}")]
	Msg(String),
	/// Converting the ONNX error message to UTF-8 failed.
	#[error("an error occurred, but ort failed to convert the error message to UTF-8")]
	IntoStringError(std::ffi::IntoStringError)
}

impl ErrorInternal {
	#[must_use]
	pub fn as_str(&self) -> Option<&str> {
		match self {
			ErrorInternal::Msg(msg) => Some(msg.as_str()),
			ErrorInternal::IntoStringError(_) => None
		}
	}
}

/// Error from downloading pre-trained model from the [ONNX Model Zoo](https://github.com/onnx/models).
#[non_exhaustive]
#[derive(Error, Debug)]
pub enum FetchModelError {
	/// Generic input/output error
	#[error("Error reading file: {0}")]
	IoError(#[from] io::Error),
	/// Download error by ureq
	#[cfg(feature = "fetch-models")]
	#[cfg_attr(docsrs, doc(cfg(feature = "fetch-models")))]
	#[error("Error downloading to file: {0}")]
	FetchError(#[from] Box<ureq::Error>),
	/// Error getting Content-Length from HTTP GET request.
	#[error("Error getting Content-Length from HTTP GET")]
	ContentLengthError,
	/// Mismatch between amount of downloaded and expected bytes.
	#[error("Error copying data to file: expected {expected} length, but got {io}")]
	CopyError {
		/// Expected amount of bytes to download
		expected: u64,
		/// Number of bytes read from network and written to file
		io: u64
	}
}

/// Wrapper type around ONNX's `OrtStatus` pointer.
///
/// This wrapper exists to facilitate conversion from C raw pointers to Rust error types.
pub struct OrtStatusWrapper(*mut ort_sys::OrtStatus);

impl From<*mut ort_sys::OrtStatus> for OrtStatusWrapper {
	fn from(status: *mut ort_sys::OrtStatus) -> Self {
		OrtStatusWrapper(status)
	}
}

pub(crate) fn assert_null_pointer<T>(ptr: *const T, name: &'static str) -> Result<()> {
	ptr.is_null().then_some(()).ok_or_else(|| Error::PointerShouldBeNull(name))
}

pub(crate) fn assert_non_null_pointer<T>(ptr: *const T, name: &'static str) -> Result<()> {
	(!ptr.is_null()).then_some(()).ok_or_else(|| Error::PointerShouldNotBeNull(name))
}

impl From<OrtStatusWrapper> for Result<(), ErrorInternal> {
	fn from(status: OrtStatusWrapper) -> Self {
		if status.0.is_null() {
			Ok(())
		} else {
			let raw: *const std::os::raw::c_char = ortsys![unsafe GetErrorMessage(status.0)];
			match char_p_to_string(raw) {
				Ok(msg) => Err(ErrorInternal::Msg(msg)),
				Err(err) => match err {
					Error::FfiStringConversion(ErrorInternal::IntoStringError(e)) => Err(ErrorInternal::IntoStringError(e)),
					_ => unreachable!()
				}
			}
		}
	}
}

impl Drop for OrtStatusWrapper {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseStatus(self.0)];
	}
}

pub(crate) fn status_to_result(status: *mut ort_sys::OrtStatus) -> Result<(), ErrorInternal> {
	let status_wrapper: OrtStatusWrapper = status.into();
	status_wrapper.into()
}
