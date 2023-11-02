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

mod ndarray;
mod types;

use std::{fmt::Debug, ops::Deref, ptr};

use ::ndarray::{ArrayView, IxDyn};

pub use self::ndarray::ArrayExtensions;
pub use self::types::{ExtractTensorData, IntoTensorElementDataType, TensorData, TensorElementDataType, Utf8Data};
use super::{ortsys, Error, Result};

/// Tensor containing data owned by the ONNX Runtime C library, used to return values from inference.
///
/// This tensor type is returned by the [`Session::run()`](../session/struct.Session.html#method.run) method.
/// It is not meant to be created directly.
///
/// The tensor hosts an [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html)
/// of the data on the C side. This allows manipulation on the Rust side using `ndarray` without copying the data.
///
/// `OrtOwnedTensor` implements the [`std::deref::Deref`](#impl-Deref) trait for ergonomic access to
/// the underlying [`ndarray::ArrayView`](https://docs.rs/ndarray/latest/ndarray/type.ArrayView.html).
#[derive(Debug)]
pub struct Tensor<'t, T>
where
	T: ExtractTensorData
{
	pub(crate) data: TensorData<'t, T>
}

impl<'t, T> Tensor<'t, T>
where
	T: ExtractTensorData
{
	/// Produce an [`ArrayViewHolder`] for the underlying data.
	pub fn view<'s>(&'s self) -> ArrayViewHolder<'s, T>
	where
		't: 's // tensor ptr can outlive the TensorData
	{
		ArrayViewHolder::new(&self.data)
	}
}

/// An intermediate step on the way to an [`ArrayView`].
// Since Deref has to produce a reference, and the referent can't be a local in deref(), it must
// be a field in a struct. This struct exists only to hold that field.
// Its lifetime 's is bound to the TensorData its view was created around, not the underlying tensor
// pointer, since in the case of strings the data is the Array in the TensorData, not the pointer.
pub struct ArrayViewHolder<'s, T>
where
	T: ExtractTensorData
{
	array_view: ArrayView<'s, T, IxDyn>
}

impl<'s, T> ArrayViewHolder<'s, T>
where
	T: ExtractTensorData
{
	fn new<'t>(data: &'s TensorData<'t, T>) -> ArrayViewHolder<'s, T>
	where
		't: 's // underlying tensor ptr lives at least as long as TensorData
	{
		match data {
			TensorData::PrimitiveView { array_view, .. } => ArrayViewHolder {
				// we already have a view, but creating a view from a view is cheap
				array_view: array_view.view()
			},
			TensorData::Strings { strings } => ArrayViewHolder {
				// This view creation has to happen here, not at new()'s callsite, because
				// a field can't be a reference to another field in the same struct. Thus, we have
				// this separate struct to hold the view that refers to the `Array`.
				array_view: strings.view()
			}
		}
	}
}

impl<'t, T> Deref for ArrayViewHolder<'t, T>
where
	T: ExtractTensorData
{
	type Target = ArrayView<'t, T, IxDyn>;

	fn deref(&self) -> &Self::Target {
		&self.array_view
	}
}

/// Holds on to a tensor pointer until dropped.
///
/// This allows for creating an [`OrtOwnedTensor`] from a [`DynOrtTensor`] without consuming `self`, which would prevent
/// retrying extraction and avoids awkward interaction with the outputs `Vec`. It also avoids requiring `OrtOwnedTensor`
/// to keep a reference to `DynOrtTensor`, which would be inconvenient.
#[derive(Debug)]
pub struct TensorPointerHolder {
	pub(crate) tensor_ptr: *mut ort_sys::OrtValue
}

impl Drop for TensorPointerHolder {
	#[tracing::instrument]
	fn drop(&mut self) {
		ortsys![unsafe ReleaseValue(self.tensor_ptr)];

		self.tensor_ptr = ptr::null_mut();
	}
}
