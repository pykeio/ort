//! Traits and types related to [`Tensor`](crate::value::Tensor)s.

#[cfg(feature = "ndarray")]
mod ndarray;
mod types;

use alloc::{string::String, vec::Vec};
use core::{
	fmt,
	ops::{Deref, DerefMut}
};

use smallvec::{SmallVec, smallvec};

#[cfg(feature = "ndarray")]
pub use self::ndarray::ArrayExtensions;
pub use self::types::{IntoTensorElementType, PrimitiveTensorElementType, TensorElementType, Utf8Data};

#[derive(Default, Clone, PartialEq, Eq)]
pub struct Shape {
	inner: SmallVec<i64, 4>
}

impl Shape {
	pub fn new(dims: impl IntoIterator<Item = i64>) -> Self {
		Self { inner: dims.into_iter().collect() }
	}

	pub fn empty(rank: usize) -> Self {
		Self { inner: smallvec![0; rank] }
	}

	#[doc(alias = "numel")]
	pub fn num_elements(&self) -> usize {
		let mut size = 1usize;
		for dim in &self.inner {
			if *dim < 0 {
				return 0;
			}
			size *= *dim as usize;
		}
		size
	}

	#[cfg(feature = "ndarray")]
	#[cfg_attr(docsrs, doc(cfg(feature = "ndarray")))]
	pub fn to_ixdyn(&self) -> ::ndarray::IxDyn {
		use ::ndarray::IntoDimension;
		self.inner.iter().map(|d| *d as usize).collect::<Vec<usize>>().into_dimension()
	}
}

impl fmt::Debug for Shape {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_list().entries(self.inner.iter()).finish()
	}
}

impl fmt::Display for Shape {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_list().entries(self.inner.iter()).finish()
	}
}

impl From<Vec<usize>> for Shape {
	fn from(value: Vec<usize>) -> Self {
		Shape::new(value.into_iter().map(|x| x as i64))
	}
}

impl From<Vec<i64>> for Shape {
	fn from(value: Vec<i64>) -> Self {
		Self { inner: SmallVec::from(value) }
	}
}

impl From<&[usize]> for Shape {
	fn from(value: &[usize]) -> Self {
		Shape::new(value.iter().map(|x| *x as i64))
	}
}

impl From<&[i64]> for Shape {
	fn from(value: &[i64]) -> Self {
		Self { inner: SmallVec::from(value) }
	}
}

impl<const N: usize> From<[usize; N]> for Shape {
	fn from(value: [usize; N]) -> Self {
		Shape::new(value.into_iter().map(|x| x as i64))
	}
}

impl<const N: usize> From<[i64; N]> for Shape {
	fn from(value: [i64; N]) -> Self {
		Self { inner: SmallVec::from(value) }
	}
}

impl FromIterator<usize> for Shape {
	fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
		Self {
			inner: iter.into_iter().map(|x| x as i64).collect()
		}
	}
}

impl FromIterator<i64> for Shape {
	fn from_iter<T: IntoIterator<Item = i64>>(iter: T) -> Self {
		Self { inner: iter.into_iter().collect() }
	}
}

impl Deref for Shape {
	type Target = [i64];
	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl DerefMut for Shape {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicDimensions(SmallVec<String, 4>);

impl SymbolicDimensions {
	pub fn new(dims: impl IntoIterator<Item = String>) -> Self {
		Self(dims.into_iter().collect())
	}

	pub fn empty(rank: usize) -> Self {
		Self(smallvec![String::default(); rank])
	}
}

impl FromIterator<String> for SymbolicDimensions {
	fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
		Self(iter.into_iter().collect())
	}
}

impl Deref for SymbolicDimensions {
	type Target = [String];
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}
