//! Contains traits for implementing custom operator domains & kernels.

use alloc::{boxed::Box, ffi::CString, vec::Vec};
use core::{
	ops::RangeInclusive,
	ptr::{self, NonNull}
};

mod attribute;
pub(crate) mod erased;
mod io;
mod kernel;
#[cfg(test)]
mod tests;

use self::erased::ErasedOperator;
pub use self::{
	attribute::{Attribute, FromKernelContext, FromOpAttr, ToAttribute},
	io::{InputOutputCharacteristic, OperatorInput, OperatorOutput},
	kernel::{BoxedKernel, ComputeContext, Kernel, KernelContext, ScratchBuffer}
};
use crate::{
	AsPointer, Error,
	error::Result,
	ortsys,
	util::with_cstr,
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
pub trait Operator: Send + Sync {
	type Kernel<'attr>: Kernel + 'attr;

	/// A list of which inputs can be reused for an output if the shapes are identical; i.e. the operation for
	/// those outputs can be performed inplace, allowing the graph optimizer to save memory.
	///
	/// The tuple is `(input_idx, output_idx)`.
	///
	/// A `Mul` operator with inputs `A` (tensor) and `B` (scalar) would set this to `&[(0, 0)]`
	/// since the `A` tensor can be reused for the output.
	#[cfg(feature = "api-18")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-18")))]
	const INPLACES: &[(u32, u32)] = &[];

	/// A list of which inputs are output without their data being changed, allowing for memory optimizations.
	///
	/// The tuple is `(input_idx, output_idx)`.
	///
	/// The `Reshape` or `Identity` operator would set this to `&[(0, 0)]` because the input's **data** doesn't change
	/// (only the shape in the case of `Reshape`).
	#[cfg(feature = "api-18")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-18")))]
	const ALIASES: &[(u32, u32)] = &[];

	/// Returns the name of the operator.
	fn name(&self) -> &str;

	/// Returns the domain version range supported by this operator.
	fn versions(&self) -> RangeInclusive<u32> {
		1..=u32::MAX
	}

	/// Returns the internal name of the execution provider this operator runs on, e.g. `"CUDAExecutionProvider"` (see
	/// [`ExecutionProvider::name`](crate::ep::ExecutionProvider::name)).
	///
	/// If the returned type is not `None`, and the execution provider used by the session matches this operator's
	/// EP type, the value will not be copied to the CPU and you may use functions like [`Tensor::data_ptr`] to
	/// access the underlying device memory, and [`KernelContext::compute_stream`] to access the GPU compute
	/// stream.
	///
	/// [`Tensor::data_ptr`]: crate::value::Tensor::data_ptr
	/// [`KernelContext::compute_stream`]: crate::operator::KernelContext::compute_stream
	fn execution_provider_type(&self) -> Option<&str> {
		None
	}

	fn inputs(&self) -> impl IntoIterator<Item = OperatorInput>;
	fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput>;

	fn create_kernel<'attr>(&self, kctx: &KernelContext<'attr>) -> crate::Result<Self::Kernel<'attr>>;

	fn infer_shape(&self, ctx: &mut ShapeInferenceContext) -> crate::Result<()> {
		let _ = ctx;
		Ok(())
	}
}

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
			ortsys![unsafe ShapeInferContext_GetInputTypeShape(self.ptr(), i, &mut ty_info).expect("failed to get info type"); nonNull(ty_info)];
			tys.push(unsafe { extract_data_type_from_tensor_info(ty_info) });
		}
		tys
	}

	pub fn attr<T: FromOpAttr>(&self, name: impl AsRef<str>) -> Result<T> {
		let attr = with_cstr(name.as_ref().as_bytes(), &|name| {
			let mut attr = ptr::null();
			ortsys![unsafe ShapeInferContext_GetAttribute(self.ptr(), name.as_ptr(), &mut attr)?];
			Ok(attr)
		})?;

		let mut len = 0;
		// this will always error since the length we pass (0) is too small to hold anything; we don't care about value yet, we
		// just want the real length which will end up in the `len` variable
		let _ = ortsys![@ort: unsafe ReadOpAttr(attr, T::attr_type(), ptr::null_mut(), 0, &mut len) as Result];

		unsafe { T::from_op_attr(attr, len) }
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
	#[allow(clippy::vec_box)]
	operators: Vec<Box<ErasedOperator>>
}

impl OperatorDomain {
	pub fn new(name: impl AsRef<str>) -> Result<Self> {
		let name = CString::new(name.as_ref())?;
		let mut ptr: *mut ort_sys::OrtCustomOpDomain = ptr::null_mut();
		ortsys![unsafe CreateCustomOpDomain(name.as_ptr(), &mut ptr)?; nonNull(ptr)];
		crate::logging::create!(OperatorDomain, ptr);
		Ok(Self {
			ptr,
			_name: name,
			operators: Vec::new()
		})
	}

	#[allow(clippy::should_implement_trait)]
	pub fn add<O: Operator + 'static>(mut self, operator: O) -> Result<Self> {
		// `Box`ing the operator here because we move it into `self` immediately after registering it. Without `Box`,
		// the pointer we pass to `CustomOpDomain_Add` would become invalid.
		let erased = Box::new(erased::erase(operator)?);
		ortsys![unsafe CustomOpDomain_Add(self.ptr.as_ptr(), (&*erased as *const ErasedOperator) as *mut _)?];

		self.operators.push(erased);

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
		crate::logging::drop!(OperatorDomain, self.ptr);
	}
}
