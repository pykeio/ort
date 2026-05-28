use alloc::{boxed::Box, ffi::CString};
use core::{
	alloc::Layout,
	ops::RangeInclusive,
	ptr::{self, NonNull}
};

use super::{
	Operator, ShapeInferenceContext,
	io::{self, InputOutputCharacteristic},
	kernel::{ComputeContext, Kernel, KernelContext}
};
use crate::{Result, error::IntoStatus};

// `Operator` itself isn't dyn-compatible, which we'd need for some methods, so we have to manually emulate `dyn` for
// those.
struct OperatorVTable {
	create_kernel: for<'attr> fn(*const (), attributes: &KernelContext<'attr>) -> crate::Result<Box<dyn Kernel + 'attr>>,
	infer_shape: fn(*const (), ctx: &mut ShapeInferenceContext) -> crate::Result<()>,
	versions: fn(*const ()) -> RangeInclusive<u32>,
	drop: fn(*mut ())
}

struct DynOperator {
	op: *mut (),
	vtable: &'static OperatorVTable
}

impl DynOperator {
	pub fn create<O: Operator>(op: O) -> DynOperator {
		fn create_kernel<'attr, O: Operator>(op: *const (), attributes: &KernelContext<'attr>) -> crate::Result<Box<dyn Kernel + 'attr>> {
			let op = unsafe { &*op.cast::<O>() };
			op.create_kernel(attributes).map(|k| Box::new(k) as Box<dyn Kernel>)
		}
		fn infer_shape<O: Operator>(op: *const (), ctx: &mut ShapeInferenceContext) -> crate::Result<()> {
			let op = unsafe { &*op.cast::<O>() };
			op.infer_shape(ctx)
		}
		fn versions<O: Operator>(op: *const ()) -> RangeInclusive<u32> {
			let op = unsafe { &*op.cast::<O>() };
			op.versions()
		}
		fn op_drop<O: Operator>(op: *mut ()) {
			let _ = unsafe { Box::from_raw(op.cast::<O>()) };
		}

		DynOperator {
			op: (Box::leak(Box::new(op)) as *mut O).cast(),
			vtable: &OperatorVTable {
				create_kernel: create_kernel::<O>,
				infer_shape: infer_shape::<O>,
				versions: versions::<O>,
				drop: op_drop::<O>
			}
		}
	}

	#[inline(always)]
	pub fn create_kernel<'attr>(&self, attributes: &KernelContext<'attr>) -> crate::Result<Box<dyn Kernel + 'attr>> {
		(self.vtable.create_kernel)(self.op, attributes)
	}
	#[inline(always)]
	pub fn infer_shape(&self, ctx: &mut ShapeInferenceContext) -> crate::Result<()> {
		(self.vtable.infer_shape)(self.op, ctx)
	}
	#[inline(always)]
	pub fn versions(&self) -> RangeInclusive<u32> {
		(self.vtable.versions)(self.op)
	}
}

impl Drop for DynOperator {
	fn drop(&mut self) {
		(self.vtable.drop)(self.op);
	}
}

#[repr(C)] // <- important! a defined layout allows us to store extra data after the `OrtCustomOp` that we can retrieve later
pub(crate) struct ErasedOperator {
	implementation: ort_sys::OrtCustomOp,
	name: CString,
	execution_provider_type: Option<CString>,
	inputs: Box<[io::OperatorInput]>,
	outputs: Box<[io::OperatorOutput]>,
	operator: DynOperator
}

unsafe impl Send for ErasedOperator {}

pub(super) fn erase<O: Operator + 'static>(operator: O) -> Result<ErasedOperator> {
	let name = CString::new(operator.name())?;
	let execution_provider_type = operator.execution_provider_type().map(CString::new).transpose()?;
	let inputs = operator.inputs().into_iter().collect();
	let outputs = operator.outputs().into_iter().collect();

	unsafe extern "system" fn create_kernel(
		op: *const ort_sys::OrtCustomOp,
		_: *const ort_sys::OrtApi,
		info: *const ort_sys::OrtKernelInfo,
		kernel_ptr: *mut *mut ort_sys::c_void
	) -> ort_sys::OrtStatusPtr {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };

		let attributes = KernelContext::from_ptr(NonNull::new(info.cast_mut()).expect("infallible"), false);
		let kernel = match erased.operator.create_kernel(&attributes) {
			Ok(kernel) => kernel,
			e => return e.into_status()
		};
		unsafe { *kernel_ptr = (Box::leak(Box::new(kernel)) as *mut Box<dyn Kernel>).cast() };
		crate::logging::create!(Kernel, unsafe { *kernel_ptr });

		Ok(()).into_status()
	}

	unsafe extern "system" fn compute_kernel(kernel_ptr: *mut ort_sys::c_void, context: *mut ort_sys::OrtKernelContext) -> ort_sys::OrtStatusPtr {
		let context = ComputeContext::new(context);
		unsafe { &mut *kernel_ptr.cast::<Box<dyn Kernel>>() }.compute(&context).into_status()
	}

	unsafe extern "system" fn destroy_kernel(op_kernel: *mut ort_sys::c_void) {
		drop(unsafe { Box::from_raw(op_kernel.cast::<Box<dyn Kernel>>()) });
		crate::logging::drop!(Kernel, op_kernel);
	}

	unsafe extern "system" fn get_name(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased.name.as_ptr()
	}

	unsafe extern "system" fn get_execution_provider_type(op: *const ort_sys::OrtCustomOp) -> *const ort_sys::c_char {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased.execution_provider_type.as_ref().map(|c| c.as_ptr()).unwrap_or_else(ptr::null)
	}

	unsafe extern "system" fn get_min_version(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		*erased.operator.versions().start() as _
	}

	unsafe extern "system" fn get_max_version(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		*erased.operator.versions().end() as _
	}

	unsafe extern "system" fn get_input_memory_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtMemType {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		// SAFETY: onnxruntime will never request an index > inputs.len
		unsafe { erased.inputs.get_unchecked(index) }.memory_type.into()
	}

	unsafe extern "system" fn get_input_characteristic(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		unsafe { erased.inputs.get_unchecked(index) }.characteristic.into()
	}

	unsafe extern "system" fn get_output_characteristic(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::OrtCustomOpInputOutputCharacteristic {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		unsafe { erased.outputs.get_unchecked(index) }.characteristic.into()
	}

	unsafe extern "system" fn get_input_type_count(op: *const ort_sys::OrtCustomOp) -> usize {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased.inputs.len()
	}

	unsafe extern "system" fn get_output_type_count(op: *const ort_sys::OrtCustomOp) -> usize {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased.outputs.len()
	}

	unsafe extern "system" fn get_input_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::ONNXTensorElementDataType {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		unsafe { erased.inputs.get_unchecked(index) }
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}

	unsafe extern "system" fn get_output_type(op: *const ort_sys::OrtCustomOp, index: usize) -> ort_sys::ONNXTensorElementDataType {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		unsafe { erased.outputs.get_unchecked(index) }
			.r#type
			.map(|c| c.into())
			.unwrap_or(ort_sys::ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
	}

	unsafe extern "system" fn get_variadic_input_min_arity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased
			.inputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.expect("input minimum arity shouldn't overflow i32")
	}

	unsafe extern "system" fn get_variadic_input_homogeneity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased
			.inputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	unsafe extern "system" fn get_variadic_output_min_arity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased
			.outputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_min_arity)
			.unwrap_or(1)
			.try_into()
			.expect("output minimum arity shouldn't overflow i32")
	}

	unsafe extern "system" fn get_variadic_output_homogeneity(op: *const ort_sys::OrtCustomOp) -> ort_sys::c_int {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		erased
			.outputs
			.iter()
			.find(|c| c.characteristic == InputOutputCharacteristic::Variadic)
			.and_then(|c| c.variadic_homogeneity)
			.unwrap_or(false)
			.into()
	}

	unsafe extern "system" fn infer_output_shape(op: *const ort_sys::OrtCustomOp, ctx: *mut ort_sys::OrtShapeInferContext) -> ort_sys::OrtStatusPtr {
		let erased = unsafe { &*op.cast::<ErasedOperator>() };
		let mut ctx = ShapeInferenceContext { ptr: ctx };
		erased.operator.infer_shape(&mut ctx).into_status()
	}

	#[cfg(feature = "api-18")]
	unsafe extern "system" fn get_may_inplace<O: Operator + 'static>(
		input_index: *mut *mut core::ffi::c_int,
		output_index: *mut *mut core::ffi::c_int
	) -> usize {
		let inplaces = O::INPLACES;
		unsafe {
			*input_index = alloc::alloc::alloc(Layout::from_size_align_unchecked(size_of::<u32>() * inplaces.len(), align_of::<u32>())).cast();
			*output_index = alloc::alloc::alloc(Layout::from_size_align_unchecked(size_of::<u32>() * inplaces.len(), align_of::<u32>())).cast();

			let (input_index, output_index) = (*input_index, *output_index);
			for (i, &(input, output)) in inplaces.iter().enumerate() {
				*input_index.add(i) = input as _;
				*output_index.add(i) = output as _;
			}
		}
		inplaces.len()
	}

	#[cfg(feature = "api-18")]
	unsafe extern "system" fn release_may_inplace<O: Operator + 'static>(input_index: *mut core::ffi::c_int, output_index: *mut core::ffi::c_int) {
		unsafe {
			alloc::alloc::dealloc(input_index.cast(), Layout::from_size_align_unchecked(size_of::<u32>() * O::INPLACES.len(), align_of::<u32>()));
			alloc::alloc::dealloc(output_index.cast(), Layout::from_size_align_unchecked(size_of::<u32>() * O::INPLACES.len(), align_of::<u32>()));
		}
	}

	#[cfg(feature = "api-18")]
	unsafe extern "system" fn get_may_alias<O: Operator + 'static>(input_index: *mut *mut core::ffi::c_int, output_index: *mut *mut core::ffi::c_int) -> usize {
		let aliases = O::ALIASES;
		unsafe {
			*input_index = alloc::alloc::alloc(Layout::from_size_align_unchecked(size_of::<u32>() * aliases.len(), align_of::<u32>())).cast();
			*output_index = alloc::alloc::alloc(Layout::from_size_align_unchecked(size_of::<u32>() * aliases.len(), align_of::<u32>())).cast();

			let (input_index, output_index) = (*input_index, *output_index);
			for (i, &(input, output)) in aliases.iter().enumerate() {
				*input_index.add(i) = input as _;
				*output_index.add(i) = output as _;
			}
		}
		aliases.len()
	}

	#[cfg(feature = "api-18")]
	unsafe extern "system" fn release_may_alias<O: Operator + 'static>(input_index: *mut core::ffi::c_int, output_index: *mut core::ffi::c_int) {
		unsafe {
			alloc::alloc::dealloc(input_index.cast(), Layout::from_size_align_unchecked(size_of::<u32>() * O::ALIASES.len(), align_of::<u32>()));
			alloc::alloc::dealloc(output_index.cast(), Layout::from_size_align_unchecked(size_of::<u32>() * O::ALIASES.len(), align_of::<u32>()));
		}
	}

	Ok(ErasedOperator {
		implementation: ort_sys::OrtCustomOp {
			version: ort_sys::ORT_API_VERSION,
			GetStartVersion: Some(get_min_version),
			GetEndVersion: Some(get_max_version),
			CreateKernel: None,
			CreateKernelV2: Some(create_kernel),
			GetInputCharacteristic: Some(get_input_characteristic),
			GetInputMemoryType: Some(get_input_memory_type),
			GetInputType: Some(get_input_type),
			GetInputTypeCount: Some(get_input_type_count),
			GetName: Some(get_name),
			GetExecutionProviderType: Some(get_execution_provider_type),
			GetOutputCharacteristic: Some(get_output_characteristic),
			GetOutputType: Some(get_output_type),
			GetOutputTypeCount: Some(get_output_type_count),
			GetVariadicInputHomogeneity: Some(get_variadic_input_homogeneity),
			GetVariadicInputMinArity: Some(get_variadic_input_min_arity),
			GetVariadicOutputHomogeneity: Some(get_variadic_output_homogeneity),
			GetVariadicOutputMinArity: Some(get_variadic_output_min_arity),
			InferOutputShapeFn: Some(infer_output_shape),
			KernelCompute: None,
			KernelComputeV2: Some(compute_kernel),
			KernelDestroy: Some(destroy_kernel),
			#[cfg(feature = "api-18")]
			GetMayInplace: Some(get_may_inplace::<O>),
			#[cfg(feature = "api-18")]
			ReleaseMayInplace: Some(release_may_inplace::<O>),
			#[cfg(feature = "api-18")]
			GetAliasMap: Some(get_may_alias::<O>),
			#[cfg(feature = "api-18")]
			ReleaseAliasMap: Some(release_may_alias::<O>)
		},
		name,
		execution_provider_type,
		inputs,
		outputs,
		operator: DynOperator::create(operator)
	})
}
