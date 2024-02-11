use std::{
	ffi::CString,
	marker::PhantomData,
	ptr::{self, NonNull}
};

use super::{
	io::InputOutputCharacteristic,
	kernel::{Kernel, KernelAttributes, KernelContext},
	DummyOperator, Operator
};
use crate::error::IntoStatus;

/// A layer between the unsafe [`ort_sys::OrtCustomOp`] structure and the safe [`Operator`] API.
///
/// Does not store any operator-specific data (as [`Operator`]s cannot be `Sized` or have state), but stores the name
/// and execution provider type [`CString`]s since those are defined as `&'static str`s and the ONNX Runtime API expects
/// null-terminated strings that last for as long as the operator.
#[repr(C)]
#[derive(Clone)]
pub(crate) struct BoundOperator<O: Operator> {
	implementation: ort_sys::OrtCustomOp,
	// The ONNX Runtime API only ever passes around [`ort_sys::OrtCustomOp`] as a pointer, so with the right layout
	// (`#[repr(C)]`) we can upcast it to a pointer to `BoundOperator` to get references to the strings.
	name: CString,
	execution_provider_type: Option<CString>,
	_operator: PhantomData<O>
}

#[allow(non_snake_case, clippy::unnecessary_cast)]
impl<O: Operator> BoundOperator<O> {
	pub(crate) fn new(name: CString, execution_provider_type: Option<CString>) -> Self {
		Self {
			implementation: ort_sys::OrtCustomOp {
				version: ort_sys::ORT_API_VERSION as _,
				GetStartVersion: Some(BoundOperator::<O>::GetStartVersion),
				GetEndVersion: Some(BoundOperator::<O>::GetEndVersion),
				CreateKernel: None,
				CreateKernelV2: Some(BoundOperator::<O>::CreateKernelV2),
				GetInputCharacteristic: Some(BoundOperator::<O>::GetInputCharacteristic),
				GetInputMemoryType: Some(BoundOperator::<O>::GetInputMemoryType),
				GetInputType: Some(BoundOperator::<O>::GetInputType),
				GetInputTypeCount: Some(BoundOperator::<O>::GetInputTypeCount),
				GetName: Some(BoundOperator::<O>::GetName),
				GetExecutionProviderType: Some(BoundOperator::<O>::GetExecutionProviderType),
				GetOutputCharacteristic: Some(BoundOperator::<O>::GetOutputCharacteristic),
				GetOutputType: Some(BoundOperator::<O>::GetOutputType),
				GetOutputTypeCount: Some(BoundOperator::<O>::GetOutputTypeCount),
				GetVariadicInputHomogeneity: Some(BoundOperator::<O>::GetVariadicInputHomogeneity),
				GetVariadicInputMinArity: Some(BoundOperator::<O>::GetVariadicInputMinArity),
				GetVariadicOutputHomogeneity: Some(BoundOperator::<O>::GetVariadicOutputHomogeneity),
				GetVariadicOutputMinArity: Some(BoundOperator::<O>::GetVariadicOutputMinArity),
				InferOutputShapeFn: if O::get_infer_shape_function().is_some() {
					Some(BoundOperator::<O>::InferOutputShapeFn)
				} else {
					None
				},
				KernelCompute: None,
				KernelComputeV2: Some(BoundOperator::<O>::ComputeKernelV2),
				KernelDestroy: Some(BoundOperator::<O>::KernelDestroy)
			},
			name,
			execution_provider_type,
			_operator: PhantomData
		}
	}

	/// Upcast an [`ort_sys::OrtCustomOp`] pointer to a safe [`BoundOperator`] reference.
	unsafe fn safe<'a>(op: *const ort_sys::OrtCustomOp) -> &'a BoundOperator<O> {
		&*op.cast()
	}

	pub(crate) unsafe extern "C" fn CreateKernelV2(
		_: *const ort_sys::OrtCustomOp,
		_: *const ort_sys::OrtApi,
		info: *const ort_sys::OrtKernelInfo,
		kernel_ptr: *mut *mut ort_sys::c_void
	) -> *mut ort_sys::OrtStatus {
		let kernel = match O::create_kernel(KernelAttributes::new(info)) {
			Ok(kernel) => kernel,
			e => return e.into_status()
		};
		*kernel_ptr = (Box::leak(Box::new(kernel)) as *mut O::Kernel).cast();
		Ok(()).into_status()
	}

	pub(crate) unsafe extern "C" fn ComputeKernelV2(kernel_ptr: *mut ort_sys::c_void, context: *mut ort_sys::OrtKernelContext) -> *mut ort_sys::OrtStatus {
		let mut context = KernelContext::new(context);
		O::Kernel::compute(unsafe { &mut *kernel_ptr.cast::<O::Kernel>() }, &mut context).into_status()
	}

	pub(crate) unsafe extern "C" fn KernelDestroy(op_kernel: *mut ort_sys::c_void) {
		drop(Box::from_raw(op_kernel.cast::<O::Kernel>()));
	}

	pub(crate) unsafe extern "C" fn GetName(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let safe = Self::safe(op);
		safe.name.as_ptr()
	}
	pub(crate) unsafe extern "C" fn GetExecutionProviderType(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let safe = Self::safe(op);
		safe.execution_provider_type.as_ref().map(|c| c.as_ptr()).unwrap_or_else(ptr::null)
	}

	pub(crate) unsafe extern "C" fn GetStartVersion(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::min_version()
	}
	pub(crate) unsafe extern "C" fn GetEndVersion(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::max_version()
	}

	pub(crate) unsafe extern "C" fn GetInputMemoryType(_: *const ort_sys::OrtCustomOp, index: ort_sys::size_t) -> ort_sys::OrtMemType {
		O::inputs()[index as usize].memory_type.into()
	}
	pub(crate) unsafe extern "C" fn GetInputCharacteristic(
		_: *const ort_sys::OrtCustomOp,
		index: ort_sys::size_t
	) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		O::inputs()[index as usize].characteristic.into()
	}
	pub(crate) unsafe extern "C" fn GetOutputCharacteristic(
		_: *const ort_sys::OrtCustomOp,
		index: ort_sys::size_t
	) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		O::outputs()[index as usize].characteristic.into()
	}
	pub(crate) unsafe extern "C" fn GetInputTypeCount(_: *const ort_sys::OrtCustomOp) -> ort_sys::size_t {
		O::inputs().len() as _
	}
	pub(crate) unsafe extern "C" fn GetOutputTypeCount(_: *const ort_sys::OrtCustomOp) -> ort_sys::size_t {
		O::outputs().len() as _
	}
	pub(crate) unsafe extern "C" fn GetInputType(_: *const ort_sys::OrtCustomOp, index: ort_sys::size_t) -> ort_sys::ONNXTensorElementDataType {
		O::inputs()[index as usize]
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}
	pub(crate) unsafe extern "C" fn GetOutputType(_: *const ort_sys::OrtCustomOp, index: ort_sys::size_t) -> ort_sys::ONNXTensorElementDataType {
		O::outputs()[index as usize]
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}
	pub(crate) unsafe extern "C" fn GetVariadicInputMinArity(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::inputs()
			.into_iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.unwrap()
	}
	pub(crate) unsafe extern "C" fn GetVariadicInputHomogeneity(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::inputs()
			.into_iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}
	pub(crate) unsafe extern "C" fn GetVariadicOutputMinArity(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::outputs()
			.into_iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.unwrap()
	}
	pub(crate) unsafe extern "C" fn GetVariadicOutputHomogeneity(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::outputs()
			.into_iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	pub(crate) unsafe extern "C" fn InferOutputShapeFn(
		_: *const ort_sys::OrtCustomOp,
		infer_ctx: *mut ort_sys::OrtShapeInferContext
	) -> *mut ort_sys::OrtStatus {
		// `unwrap_unchecked()` is safe here because `BoundOperator::new` will only add this function to the `OrtCustomOp`
		// definition if the infer shape function is present.
		O::get_infer_shape_function().unwrap_unchecked()(infer_ctx).into_status()
	}
}

/// A type-erased [`BoundOperator`].
pub(crate) struct ErasedBoundOperator(NonNull<()>);

unsafe impl Send for ErasedBoundOperator {}

impl ErasedBoundOperator {
	pub(crate) fn new<O: Operator>(bound: BoundOperator<O>) -> Self {
		ErasedBoundOperator(NonNull::from(unsafe {
			// I stopped writing C because I didn't like pointer trickery, and yet here I am...
			&mut *(Box::leak(Box::new(bound)) as *mut _ as *mut ())
		}))
	}

	/// Returns the pointer to the contained [`BoundOperator`] as an [`ort_sys::OrtCustomOp`] pointer.
	pub(crate) fn op_ptr(&self) -> *mut ort_sys::OrtCustomOp {
		self.0.as_ptr().cast()
	}
}

impl Drop for ErasedBoundOperator {
	fn drop(&mut self) {
		// [`Operator`]s cannot be `Sized`, so we don't have to call the drop implementation for the operator.
		// We do need something to put in the type parameter though, so we use `DummyOperator`.
		drop(unsafe { Box::from_raw(self.0.as_ptr().cast::<BoundOperator<DummyOperator>>()) });
	}
}
