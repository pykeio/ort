use std::ops::{Deref, DerefMut};

use tract_onnx::prelude::DatumType;

pub struct Tensor {
	pub inner: tract_onnx::prelude::Tensor,
	pub shape: Vec<i64>
}

impl From<tract_onnx::prelude::Tensor> for Tensor {
	fn from(tensor: tract_onnx::prelude::Tensor) -> Self {
		Self {
			shape: tensor.shape().iter().map(|x| *x as i64).collect(),
			inner: tensor
		}
	}
}

impl Deref for Tensor {
	type Target = tract_onnx::prelude::Tensor;

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
	pub dtype: DatumType,
	pub shape: Vec<i64>
}

impl TypeInfo {
	pub fn new_sys(dtype: DatumType, shape: Vec<i64>) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(Self { dtype, shape })) as *mut TypeInfo).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		Box::from_raw(ptr.cast::<TypeInfo>())
	}
}
