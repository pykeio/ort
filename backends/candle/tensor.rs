use std::ops::{Deref, DerefMut};

use candle_core::DType;

pub struct Tensor {
	pub inner: candle_core::Tensor,
	pub shape: Vec<i64>
}

impl From<candle_core::Tensor> for Tensor {
	fn from(tensor: candle_core::Tensor) -> Self {
		Self {
			shape: tensor.dims().iter().map(|x| *x as i64).collect(),
			inner: tensor
		}
	}
}

impl Deref for Tensor {
	type Target = candle_core::Tensor;

	fn deref(&self) -> &Self::Target {
		&self.inner
	}
}

impl DerefMut for Tensor {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner
	}
}

pub struct TypeInfo {
	pub dtype: DType,
	pub shape: Vec<i64>
}

impl TypeInfo {
	pub fn new_sys(dtype: DType, shape: Vec<i64>) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(Self { dtype, shape })) as *mut TypeInfo).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		Box::from_raw(ptr.cast::<TypeInfo>())
	}
}
