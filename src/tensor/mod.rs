//! Traits related to [`Tensor`](crate::value::Tensor)s.

#[cfg(feature = "ndarray")]
mod ndarray;
mod types;

#[cfg(feature = "ndarray")]
pub use self::ndarray::ArrayExtensions;
pub use self::types::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType, Utf8Data};
#[cfg(feature = "ndarray")]
pub(crate) use self::types::{extract_primitive_array, extract_primitive_array_mut};
