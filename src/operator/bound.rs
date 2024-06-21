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

#[repr(C)] // <- important! a defined layout allows us to store extra data after the `OrtCustomOp` that we can retrieve later
pub(crate) struct BoundOperator<O: Operator> {
	implementation: ort_sys::OrtCustomOp,
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
				GetAliasMap: None,
				ReleaseAliasMap: None,
				GetMayInplace: None,
				ReleaseMayInplace: None,
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

	unsafe fn safe<'a>(op: *const ort_sys::OrtCustomOp) -> &'a BoundOperator<O> {
		&*op.cast()
	}

	pub(crate) unsafe extern "C" fn CreateKernelV2(
		_: *const ort_sys::OrtCustomOp,
		_: *const ort_sys::OrtApi,
		info: *const ort_sys::OrtKernelInfo,
		kernel_ptr: *mut *mut ort_sys::c_void
	) -> *mut ort_sys::OrtStatus {
		let kernel = match O::create_kernel(&KernelAttributes::new(info)) {
			Ok(kernel) => kernel,
			e => return e.into_status()
		};
		*kernel_ptr = (Box::leak(Box::new(kernel)) as *mut O::Kernel).cast();
		Ok(()).into_status()
	}

	pub(crate) unsafe extern "C" fn ComputeKernelV2(kernel_ptr: *mut ort_sys::c_void, context: *mut ort_sys::OrtKernelContext) -> *mut ort_sys::OrtStatus {
		let context = KernelContext::new(context);
		O::Kernel::compute(unsafe { &mut *kernel_ptr.cast::<O::Kernel>() }, &context).into_status()
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
			.expect("input minimum arity overflows i32")
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
			.expect("output minimum arity overflows i32")
	}
	pub(crate) unsafe extern "C" fn GetVariadicOutputHomogeneity(_: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		O::outputs()
			.into_iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	pub(crate) unsafe extern "C" fn InferOutputShapeFn(_: *const ort_sys::OrtCustomOp, arg1: *mut ort_sys::OrtShapeInferContext) -> *mut ort_sys::OrtStatus {
		O::get_infer_shape_function().expect("missing infer shape function")(arg1).into_status()
	}
}

pub(crate) struct ErasedBoundOperator(NonNull<()>);

unsafe impl Send for ErasedBoundOperator {}

impl ErasedBoundOperator {
	pub(crate) fn new<O: Operator>(bound: BoundOperator<O>) -> Self {
		ErasedBoundOperator(NonNull::from(unsafe {
			// horrible horrible horrible horrible horrible horrible horrible horrible horrible
			&mut *(Box::leak(Box::new(bound)) as *mut _ as *mut ())
		}))
	}

	pub(crate) fn op_ptr(&self) -> *mut ort_sys::OrtCustomOp {
		self.0.as_ptr().cast()
	}
}

impl Drop for ErasedBoundOperator {
	fn drop(&mut self) {
		drop(unsafe { Box::from_raw(self.0.as_ptr().cast::<BoundOperator<DummyOperator>>()) });
	}
}
