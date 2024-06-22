//! Module containing tensor types.
//!
//! Two main types of tensors are available.
//!
//! The first one, [`OrtTensor`], is an _owned_ tensor that is backed by [`ndarray`](https://crates.io/crates/ndarray).
//! This kind of tensor is used to pass input data for the inference.
//!
//! The second one, [`OrtOwnedTensor`], is used internally to pass to the ONNX Runtime inference execution to place its
//! output values. Once "extracted" from the runtime environment, this tensor will contain an [`ndarray::ArrayView`]
//! containing _a view_ of the data. When going out of scope, this tensor will free the required memory on the C side.
//!
//! **NOTE**: Tensors are not meant to be created directly. When performing inference, the
//! [`Session::run`](crate::Session::run) method takes an `ndarray::Array` as input (taking ownership of it) and will
//! convert it internally to an [`OrtTensor`]. After inference, a [`OrtOwnedTensor`] will be returned by the method
//! which can be derefed into its internal [`ndarray::ArrayView`].

#[cfg(feature = "ndarray")]
mod ndarray;
mod types;

#[cfg(feature = "ndarray")]
pub use self::ndarray::ArrayExtensions;
#[cfg(feature = "ndarray")]
pub(crate) use self::types::{extract_primitive_array, extract_primitive_array_mut};
pub use self::types::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType, Utf8Data};
