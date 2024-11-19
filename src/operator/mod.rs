//! Contains traits for implementing custom operator domains & kernels.

use std::{
	ffi::CString,
	ptr::{self, NonNull}
};

pub(crate) mod bound;
pub mod io;
pub mod kernel;
#[cfg(test)]
mod tests;

use self::{
	bound::{BoundOperator, ErasedBoundOperator},
	io::{OperatorInput, OperatorOutput},
	kernel::{DummyKernel, Kernel, KernelAttributes}
};
use crate::{
	AsPointer, Error,
	error::Result,
	ortsys,
	value::{ValueType, r#type::extract_data_type_from_tensor_info}
};

/// A custom operator descriptor, which describes the expected inputs & outputs of a graph operator.
///
/// [`Operator`]s are bound to [`OperatorDomain`]s. Multiple operators can have the same name as long as they have
/// different input/output types, in which case the exact operator will be picked depending on the input/output
/// types. If you want to, for example, define a `Sort` operator that can accept either a single `f32` or `i64` tensor
/// input, you'll need to define 2 separate operators (which can be done via a macro); but both of these
/// [`Operator`] structs can return the same name in [`Operator::name`] so that they are usable as simply
/// `my.domain:Sort` in the graph.
pub trait Operator: Send {
	type Kernel: Kernel;

	/// Returns the name of the operator.
	fn name() -> &'static str;

	/// Returns the execution provider this operator runs on, e.g. `CUDAExecutionProvider`.
	///
	/// If the returned type is not `None`, and the execution provider used by the session matches this operator's
	/// EP type, the value will not be copied to the CPU and you may use functions like [`Tensor::data_ptr`] to
	/// access the underlying device memory, and [`KernelContext::compute_stream`] to access the GPU compute
	/// stream.
	///
	/// [`Tensor::data_ptr`]: crate::value::Tensor::data_ptr
	/// [`KernelContext::compute_stream`]: crate::operator::kernel::KernelContext::compute_stream
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

/// Dummy type implementing [`Operator`] used by [`ErasedBoundOperator`] to cheat the type system.
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

pub type InferShapeFn = dyn FnMut(&mut ShapeInferenceContext) -> crate::Result<()> + 'static;

pub struct ShapeInferenceContext {
	ptr: *mut ort_sys::OrtShapeInferContext
}

impl ShapeInferenceContext {
	pub fn inputs(&self) -> Vec<ValueType> {
		let mut count = 0;
		ortsys![unsafe ShapeInferContext_GetInputCount(self.ptr(), &mut count).expect("failed to get input count")];

		let mut tys = Vec::with_capacity(count);
		for i in 0..count {
			let mut ty_info = ptr::null_mut();
			ortsys![unsafe ShapeInferContext_GetInputTypeShape(self.ptr(), i, &mut ty_info).expect("failed to get info type")];
			tys.push(unsafe { extract_data_type_from_tensor_info(ty_info) });
		}
		tys
	}

	pub fn set_output(&mut self, idx: usize, ty: &ValueType) -> Result<()> {
		match ty.to_tensor_type_info() {
			Some(ty_ptr) => {
				ortsys![unsafe ShapeInferContext_SetOutputTypeShape(self.ptr(), idx, ty_ptr)?];
				ortsys![unsafe ReleaseTensorTypeAndShapeInfo(ty_ptr)];
				Ok(())
			}
			None => Err(Error::new("only tensors are supported"))
		}
	}
}

impl AsPointer for ShapeInferenceContext {
	type Sys = ort_sys::OrtShapeInferContext;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr
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
		ortsys![unsafe CreateCustomOpDomain(name.as_ptr(), &mut ptr)?; nonNull(ptr)];
		Ok(Self {
			_name: name,
			ptr: NonNull::from(unsafe { &mut *ptr }),
			operators: Vec::new()
		})
	}

	#[allow(clippy::should_implement_trait)]
	pub fn add<O: Operator>(mut self) -> Result<Self> {
		let name = O::name();

		let bound = BoundOperator::<O>::new(CString::new(name)?, O::execution_provider_type().map(CString::new).transpose()?);
		let bound = ErasedBoundOperator::new(bound);
		ortsys![unsafe CustomOpDomain_Add(self.ptr.as_ptr(), bound.op_ptr())?];

		self.operators.push(bound);

		Ok(self)
	}
}

impl AsPointer for OperatorDomain {
	type Sys = ort_sys::OrtCustomOpDomain;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for OperatorDomain {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseCustomOpDomain(self.ptr.as_ptr())];
	}
}
