//! Helper traits to extend [`ndarray`] functionality.

use core::ops::{DivAssign, SubAssign};

use ndarray::{Array, ArrayBase};

/// Trait extending [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)
/// with useful tensor operations.
pub trait ArrayExtensions<S, T, D> {
	/// Calculate the [softmax](https://en.wikipedia.org/wiki/Softmax_function) of the tensor along a given axis.
	fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
	where
		D: ndarray::RemoveAxis,
		S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
		<S as ndarray::RawData>::Elem: Clone,
		T: ndarray::NdFloat + SubAssign + DivAssign;
}

impl<S, T, D> ArrayExtensions<S, T, D> for ArrayBase<S, D>
where
	D: ndarray::RemoveAxis,
	S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
	<S as ndarray::RawData>::Elem: Clone,
	T: ndarray::NdFloat + SubAssign + DivAssign
{
	fn softmax(&self, axis: ndarray::Axis) -> Array<T, D> {
		let mut new_array: Array<T, D> = self.to_owned();
		// FIXME: Change to non-overflowing formula
		// e = np.exp(A - np.sum(A, axis=1, keepdims=True))
		// np.exp(a) / np.sum(np.exp(a))
		new_array.map_inplace(|v| *v = v.exp());
		let sum = new_array.sum_axis(axis).insert_axis(axis);
		new_array /= &sum;

		new_array
	}
}

#[cfg(test)]
mod tests {
	use ndarray::{arr1, arr2, arr3};

	use super::*;

	#[test]
	fn softmax_1d() {
		let array = arr1(&[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]);

		let expected_softmax = arr1(&[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]);

		let softmax = array.softmax(ndarray::Axis(0));

		assert_eq!(softmax.shape(), expected_softmax.shape());

		let diff = softmax - expected_softmax;

		assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
	}

	#[test]
	fn softmax_2d() {
		let array = arr2(&[[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]]);

		let expected_softmax = arr2(&[
			[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
			[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]
		]);

		let softmax = array.softmax(ndarray::Axis(1));

		assert_eq!(softmax.shape(), expected_softmax.shape());

		let diff = softmax - expected_softmax;

		assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
	}

	#[test]
	fn softmax_3d() {
		let array = arr3(&[
			[[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]],
			[[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]],
			[[1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], [1.0_f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]]
		]);

		let expected_softmax = arr3(&[
			[
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]
			],
			[
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]
			],
			[
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813],
				[0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813]
			]
		]);

		let softmax = array.softmax(ndarray::Axis(2));

		assert_eq!(softmax.shape(), expected_softmax.shape());

		let diff = softmax - expected_softmax;

		assert!(diff.iter().all(|d| d.abs() < 1.0e-7));
	}
}
