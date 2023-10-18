//! Provides [`OrtOwnedTensor`], a temporary view of tensor data owned by ONNX Runtime in C++.

use std::{fmt::Debug, ops::Deref, ptr};

use ndarray::{ArrayView, IxDyn};

use super::{TensorData, TensorDataToType};
use crate::ortsys;

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
pub struct OrtOwnedTensor<'t, T>
where
	T: TensorDataToType
{
	pub(crate) data: TensorData<'t, T>
}

impl<'t, T> OrtOwnedTensor<'t, T>
where
	T: TensorDataToType
{
	/// Produce a [`ViewHolder`] for the underlying data.
	pub fn view<'s>(&'s self) -> ViewHolder<'s, T>
	where
		't: 's // tensor ptr can outlive the TensorData
	{
		ViewHolder::new(&self.data)
	}
}

/// An intermediate step on the way to an [`ArrayView`].
// Since Deref has to produce a reference, and the referent can't be a local in deref(), it must
// be a field in a struct. This struct exists only to hold that field.
// Its lifetime 's is bound to the TensorData its view was created around, not the underlying tensor
// pointer, since in the case of strings the data is the Array in the TensorData, not the pointer.
pub struct ViewHolder<'s, T>
where
	T: TensorDataToType
{
	array_view: ndarray::ArrayView<'s, T, IxDyn>
}

impl<'s, T> ViewHolder<'s, T>
where
	T: TensorDataToType
{
	fn new<'t>(data: &'s TensorData<'t, T>) -> ViewHolder<'s, T>
	where
		't: 's // underlying tensor ptr lives at least as long as TensorData
	{
		match data {
			TensorData::TensorPtr { array_view, .. } => ViewHolder {
				// we already have a view, but creating a view from a view is cheap
				array_view: array_view.view()
			},
			TensorData::Strings { strings } => ViewHolder {
				// This view creation has to happen here, not at new()'s callsite, because
				// a field can't be a reference to another field in the same struct. Thus, we have
				// this separate struct to hold the view that refers to the `Array`.
				array_view: strings.view()
			}
		}
	}
}

impl<'t, T> Deref for ViewHolder<'t, T>
where
	T: TensorDataToType
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
