use alloc::{boxed::Box, ffi::CString, vec::Vec};
use core::ptr::{self, NonNull};

use super::{
	Operator, ShapeInferenceContext,
	io::{self, InputOutputCharacteristic},
	kernel::{Kernel, KernelAttributes, KernelContext}
};
use crate::{Result, error::IntoStatus};

#[repr(C)] // <- important! a defined layout allows us to store extra data after the `OrtCustomOp` that we can retrieve later
pub(crate) struct BoundOperator {
	implementation: ort_sys::OrtCustomOp,
	name: CString,
	execution_provider_type: Option<CString>,
	inputs: Vec<io::OperatorInput>,
	outputs: Vec<io::OperatorOutput>,
	operator: Box<dyn Operator>
}

unsafe impl Send for BoundOperator {}

#[allow(non_snake_case, clippy::unnecessary_cast)]
impl BoundOperator {
	pub(crate) fn new<O: Operator + 'static>(operator: O) -> Result<Self> {
		let name = CString::new(operator.name())?;
		let execution_provider_type = operator.execution_provider_type().map(CString::new).transpose()?;

		Ok(Self {
			implementation: ort_sys::OrtCustomOp {
				version: ort_sys::ORT_API_VERSION,
				GetStartVersion: Some(BoundOperator::get_min_version),
				GetEndVersion: Some(BoundOperator::get_max_version),
				CreateKernel: None,
				CreateKernelV2: Some(BoundOperator::create_kernel),
				GetInputCharacteristic: Some(BoundOperator::get_input_characteristic),
				GetInputMemoryType: Some(BoundOperator::get_input_memory_type),
				GetInputType: Some(BoundOperator::get_input_type),
				GetInputTypeCount: Some(BoundOperator::get_input_type_count),
				GetName: Some(BoundOperator::get_name),
				GetExecutionProviderType: Some(BoundOperator::get_execution_provider_type),
				GetOutputCharacteristic: Some(BoundOperator::get_output_characteristic),
				GetOutputType: Some(BoundOperator::get_output_type),
				GetOutputTypeCount: Some(BoundOperator::get_output_type_count),
				GetVariadicInputHomogeneity: Some(BoundOperator::get_variadic_input_homogeneity),
				GetVariadicInputMinArity: Some(BoundOperator::get_variadic_input_min_arity),
				GetVariadicOutputHomogeneity: Some(BoundOperator::get_variadic_output_homogeneity),
				GetVariadicOutputMinArity: Some(BoundOperator::get_variadic_output_min_arity),
				GetAliasMap: None,
				ReleaseAliasMap: None,
				GetMayInplace: None,
				ReleaseMayInplace: None,
				InferOutputShapeFn: Some(BoundOperator::infer_output_shape),
				KernelCompute: None,
				KernelComputeV2: Some(BoundOperator::compute_kernel),
				KernelDestroy: Some(BoundOperator::destroy_kernel)
			},
			name,
			execution_provider_type,
			inputs: operator.inputs(),
			outputs: operator.outputs(),
			operator: Box::new(operator)
		})
	}

	fn safe<'a>(op: *const ort_sys::OrtCustomOp) -> &'a BoundOperator {
		unsafe { &*op.cast() }
	}

	pub(crate) extern "system" fn create_kernel(
		op: *const ort_sys::OrtCustomOp,
		_: *const ort_sys::OrtApi,
		info: *const ort_sys::OrtKernelInfo,
		kernel_ptr: *mut *mut ort_sys::c_void
	) -> ort_sys::OrtStatusPtr {
		let safe = Self::safe(op);
		let kernel = match safe
			.operator
			.create_kernel(&KernelAttributes::from_ptr(NonNull::new(info.cast_mut()).expect("infallible"), false))
		{
			Ok(kernel) => kernel,
			e => return e.into_status()
		};
		unsafe { *kernel_ptr = (Box::leak(Box::new(kernel)) as *mut Box<dyn Kernel>).cast() };
		Ok(()).into_status()
	}

	pub(crate) extern "system" fn compute_kernel(kernel_ptr: *mut ort_sys::c_void, context: *mut ort_sys::OrtKernelContext) -> ort_sys::OrtStatusPtr {
		let context = KernelContext::new(context);
		unsafe { &mut *kernel_ptr.cast::<Box<dyn Kernel>>() }.compute(&context).into_status()
	}

	pub(crate) extern "system" fn destroy_kernel(op_kernel: *mut ort_sys::c_void) {
		drop(unsafe { Box::from_raw(op_kernel.cast::<Box<dyn Kernel>>()) });
	}

	pub(crate) extern "system" fn get_name(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let safe = Self::safe(op);
		safe.name.as_ptr()
	}

	pub(crate) extern "system" fn get_execution_provider_type(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let safe = Self::safe(op);
		safe.execution_provider_type.as_ref().map(|c| c.as_ptr()).unwrap_or_else(ptr::null)
	}

	pub(crate) extern "system" fn get_min_version(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.operator.min_version() as _
	}

	pub(crate) extern "system" fn get_max_version(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.operator.max_version() as _
	}

	pub(crate) extern "system" fn get_input_memory_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtMemType {
		let safe = Self::safe(op);
		safe.inputs[index].memory_type.into()
	}

	pub(crate) extern "system" fn get_input_characteristic(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		let safe = Self::safe(op);
		safe.inputs[index].characteristic.into()
	}

	pub(crate) extern "system" fn get_output_characteristic(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		let safe = Self::safe(op);
		safe.outputs[index].characteristic.into()
	}

	pub(crate) extern "system" fn get_input_type_count(op: *const ort_sys::OrtCustomOp) -> usize {
		let safe = Self::safe(op);
		safe.inputs.len()
	}

	pub(crate) extern "system" fn get_output_type_count(op: *const ort_sys::OrtCustomOp) -> usize {
		let safe = Self::safe(op);
		safe.outputs.len()
	}

	pub(crate) extern "system" fn get_input_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::ONNXTensorElementDataType {
		let safe = Self::safe(op);
		safe.inputs[index]
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}

	pub(crate) extern "system" fn get_output_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::ONNXTensorElementDataType {
		let safe = Self::safe(op);
		safe.outputs[index]
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}

	pub(crate) extern "system" fn get_variadic_input_min_arity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.inputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.expect("input minimum arity overflows i32")
	}

	pub(crate) extern "system" fn get_variadic_input_homogeneity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.inputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	pub(crate) extern "system" fn get_variadic_output_min_arity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.outputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.expect("output minimum arity overflows i32")
	}

	pub(crate) extern "system" fn get_variadic_output_homogeneity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let safe = Self::safe(op);
		safe.outputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	pub(crate) extern "system" fn infer_output_shape(op: *const ort_sys::OrtCustomOp, ctx: *mut ort_sys::OrtShapeInferContext) -> ort_sys::OrtStatusPtr {
		let safe = Self::safe(op);
		let mut ctx = ShapeInferenceContext { ptr: ctx };
		safe.operator.infer_shape(&mut ctx).into_status()
	}
}
