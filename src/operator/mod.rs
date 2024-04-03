use std::{
	ffi::CString,
	ptr::{self, NonNull}
};

pub(crate) mod bound;
pub(crate) mod io;
pub(crate) mod kernel;

use self::{
	bound::ErasedBoundOperator,
	io::{OperatorInput, OperatorOutput},
	kernel::{DummyKernel, Kernel, KernelAttributes}
};
use crate::{operator::bound::BoundOperator, ortsys, Error, Result};

pub type InferShapeFn = dyn FnMut(*mut ort_sys::OrtShapeInferContext) -> crate::Result<()>;

pub trait Operator: Send {
	type Kernel: Kernel;

	fn name() -> &'static str;

	fn execution_provider_type() -> Option<&'static str> {
		None
	}

	fn inputs() -> Vec<OperatorInput>;
	fn outputs() -> Vec<OperatorOutput>;

	fn create_kernel(attributes: &KernelAttributes) -> crate::Result<Self::Kernel>;

	fn min_version() -> ort_sys::c_int {
		1
	}
	fn max_version() -> ort_sys::c_int {
		ort_sys::c_int::MAX
	}

	fn get_infer_shape_function() -> Option<Box<InferShapeFn>> {
		None
	}
}

struct DummyOperator;

impl Operator for DummyOperator {
	type Kernel = DummyKernel;

	fn name() -> &'static str {
		unimplemented!()
	}
	fn create_kernel(_: &KernelAttributes) -> crate::Result<Self::Kernel> {
		unimplemented!()
	}
	fn inputs() -> Vec<OperatorInput> {
		unimplemented!()
	}
	fn outputs() -> Vec<OperatorOutput> {
		unimplemented!()
	}
}

pub struct OperatorDomain {
	ptr: NonNull<ort_sys::OrtCustomOpDomain>,
	_name: CString,
	operators: Vec<ErasedBoundOperator>
}

impl OperatorDomain {
	pub fn new(name: impl AsRef<str>) -> Result<Self> {
		let name = CString::new(name.as_ref())?;
		let mut ptr: *mut ort_sys::OrtCustomOpDomain = ptr::null_mut();
		ortsys![unsafe CreateCustomOpDomain(name.as_ptr(), &mut ptr) -> Error::CreateOperatorDomain; nonNull(ptr)];
		Ok(Self {
			_name: name,
			ptr: NonNull::from(unsafe { &mut *ptr }),
			operators: Vec::new()
		})
	}

	pub(crate) fn ptr(&self) -> *mut ort_sys::OrtCustomOpDomain {
		self.ptr.as_ptr()
	}

	#[allow(clippy::should_implement_trait)]
	pub fn add<O: Operator>(mut self, _operator: O) -> Result<Self> {
		let name = O::name();

		let bound = BoundOperator::<O>::new(CString::new(name)?, O::execution_provider_type().map(CString::new).transpose()?);
		let bound = ErasedBoundOperator::new(bound);
		ortsys![unsafe CustomOpDomain_Add(self.ptr.as_ptr(), bound.op_ptr()) -> Error::AddCustomOperator];

		self.operators.push(bound);

		Ok(self)
	}
}

impl Drop for OperatorDomain {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseCustomOpDomain(self.ptr.as_ptr())];
	}
}
