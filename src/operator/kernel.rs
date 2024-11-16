use std::{
	ffi::{CString, c_char, c_void},
	ops::{Deref, DerefMut},
	ptr::{self, NonNull}
};

use crate::{
	AsPointer,
	error::{Error, Result, status_to_result},
	memory::{Allocator, MemoryInfo, MemoryType},
	ortsys,
	session::{Input, Output},
	value::{DowncastableTarget, DynValue, Value, ValueRef, ValueRefMut, ValueType}
};

pub trait Kernel {
	fn compute(&mut self, ctx: &KernelContext) -> crate::Result<()>;
}

pub(crate) struct DummyKernel;

impl Kernel for DummyKernel {
	fn compute(&mut self, _: &KernelContext) -> crate::Result<()> {
		unimplemented!()
	}
}

pub struct KernelAttributes(NonNull<ort_sys::OrtKernelInfo>);

impl KernelAttributes {
	pub(crate) fn new(info: *const ort_sys::OrtKernelInfo) -> Self {
		Self(NonNull::from(unsafe { &*info }))
	}

	#[allow(private_bounds)]
	pub fn get<'s, T: GetKernelAttribute<'s>>(&'s self, name: impl AsRef<str>) -> Option<T> {
		let name = CString::new(name.as_ref()).ok()?;
		T::get_from(self.0.as_ptr(), name.as_ptr())
	}

	pub fn inputs(&self) -> Result<Vec<Input>> {
		let mut num_inputs = 0;
		ortsys![unsafe KernelInfo_GetInputCount(self.0.as_ptr(), &mut num_inputs)?];

		let mut inputs = Vec::with_capacity(num_inputs);
		for idx in 0..num_inputs {
			let mut name_len = 0;
			ortsys![unsafe KernelInfo_GetInputName(self.0.as_ptr(), idx, ptr::null_mut(), &mut name_len)?];
			let mut name = vec![0u8; name_len];
			ortsys![unsafe KernelInfo_GetInputName(self.0.as_ptr(), idx, name.as_mut_ptr().cast::<c_char>(), &mut name_len)?];
			let name = CString::from_vec_with_nul(name)
				.map_err(Error::wrap)?
				.into_string()
				.map_err(Error::wrap)?;
			let mut type_info = ptr::null_mut();
			ortsys![unsafe KernelInfo_GetInputTypeInfo(self.0.as_ptr(), idx, &mut type_info)?; nonNull(type_info)];
			let input_type = ValueType::from_type_info(type_info);
			inputs.push(Input { name, input_type })
		}
		Ok(inputs)
	}

	pub fn outputs(&self) -> Result<Vec<Output>> {
		let mut num_outputs = 0;
		ortsys![unsafe KernelInfo_GetOutputCount(self.0.as_ptr(), &mut num_outputs)?];

		let mut outputs = Vec::with_capacity(num_outputs);
		for idx in 0..num_outputs {
			let mut name_len = 0;
			ortsys![unsafe KernelInfo_GetOutputName(self.0.as_ptr(), idx, ptr::null_mut(), &mut name_len)?];
			let mut name = vec![0u8; name_len];
			ortsys![unsafe KernelInfo_GetOutputName(self.0.as_ptr(), idx, name.as_mut_ptr().cast::<c_char>(), &mut name_len)?];
			let name = CString::from_vec_with_nul(name)
				.map_err(Error::wrap)?
				.into_string()
				.map_err(Error::wrap)?;
			let mut type_info = ptr::null_mut();
			ortsys![unsafe KernelInfo_GetOutputTypeInfo(self.0.as_ptr(), idx, &mut type_info)?; nonNull(type_info)];
			let output_type = ValueType::from_type_info(type_info);
			outputs.push(Output { name, output_type })
		}
		Ok(outputs)
	}

	pub fn node_name(&self) -> Result<String> {
		let mut name_len = 0;
		ortsys![unsafe KernelInfo_GetNodeName(self.0.as_ptr(), ptr::null_mut(), &mut name_len)?];
		let mut name = vec![0u8; name_len];
		ortsys![unsafe KernelInfo_GetNodeName(self.0.as_ptr(), name.as_mut_ptr().cast::<c_char>(), &mut name_len)?];
		CString::from_vec_with_nul(name).map_err(Error::wrap)?.into_string().map_err(Error::wrap)
	}

	pub fn allocator(&self, mem_type: MemoryType) -> Result<Allocator> {
		let mut ptr: *mut ort_sys::OrtAllocator = ptr::null_mut();
		ortsys![unsafe KernelInfoGetAllocator(self.0.as_ptr(), mem_type.into(), &mut ptr)?];
		Ok(unsafe { Allocator::from_raw_unchecked(ptr) })
	}
}

impl AsPointer for KernelAttributes {
	type Sys = ort_sys::OrtKernelInfo;

	fn ptr(&self) -> *const Self::Sys {
		self.0.as_ptr()
	}
}

pub(crate) trait GetKernelAttribute<'s> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized;
}

impl GetKernelAttribute<'_> for f32 {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_float(info, name, &mut value)]).ok()?;
		Some(value)
	}
}

impl GetKernelAttribute<'_> for i64 {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_int64(info, name, &mut value)]).ok()?;
		Some(value)
	}
}

impl GetKernelAttribute<'_> for String {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_string(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0u8; size];
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_string(info, name, out.as_mut_ptr().cast::<c_char>(), &mut size)]).ok()?;
		CString::from_vec_with_nul(out).ok().and_then(|c| c.into_string().ok())
	}
}

impl GetKernelAttribute<'_> for Vec<f32> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0f32; size];
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, out.as_mut_ptr(), &mut size)]).ok()?;
		Some(out)
	}
}

impl GetKernelAttribute<'_> for Vec<i64> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_int64(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0i64; size];
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_int64(info, name, out.as_mut_ptr(), &mut size)]).ok()?;
		Some(out)
	}
}

impl<'s, T: DowncastableTarget> GetKernelAttribute<'s> for ValueRef<'s, T> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		// TODO: This should probably be customizable - docs say the allocator is required for "internal tensor state", but it's
		// not clear if this also includes tensor data (and thus it should instead be allocated on an appropriate device).
		let allocator = Allocator::default();

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_tensor(info, name, allocator.ptr().cast_mut(), &mut value_ptr)]).ok()?;
		unsafe { ValueRef::new(DynValue::from_ptr(NonNull::new(value_ptr)?, None)) }
			.downcast()
			.ok()
	}
}

pub struct ScratchBuffer<T> {
	allocator: Allocator,
	buffer: *mut T,
	size: usize
}

impl<T> Deref for ScratchBuffer<T> {
	type Target = [T];

	fn deref(&self) -> &Self::Target {
		unsafe { std::slice::from_raw_parts(self.buffer.cast_const(), self.size) }
	}
}
impl<T> DerefMut for ScratchBuffer<T> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		unsafe { std::slice::from_raw_parts_mut(self.buffer, self.size) }
	}
}

impl<T> Drop for ScratchBuffer<T> {
	fn drop(&mut self) {
		unsafe {
			self.allocator.free(self.buffer);
		}
	}
}

pub struct KernelContext {
	ptr: NonNull<ort_sys::OrtKernelContext>
}

impl KernelContext {
	pub(crate) fn new(ctx: *mut ort_sys::OrtKernelContext) -> Self {
		Self {
			ptr: NonNull::from(unsafe { &mut *ctx })
		}
	}

	pub fn input(&self, idx: usize) -> Result<Option<ValueRef<'_>>> {
		let mut value_ptr: *const ort_sys::OrtValue = ptr::null();
		ortsys![unsafe KernelContext_GetInput(self.ptr.as_ptr(), idx, &mut value_ptr)?];
		Ok(NonNull::new(value_ptr.cast_mut()).map(|c| ValueRef::new(unsafe { Value::from_ptr_nodrop(c, None) })))
	}

	pub fn output(&self, idx: usize, shape: impl IntoIterator<Item = i64>) -> Result<Option<ValueRefMut<'_>>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		let shape = shape.into_iter().collect::<Vec<i64>>();
		ortsys![unsafe KernelContext_GetOutput(self.ptr.as_ptr(), idx, shape.as_ptr(), shape.len(), &mut value_ptr)?];
		Ok(NonNull::new(value_ptr).map(|c| ValueRefMut::new(unsafe { Value::from_ptr_nodrop(c, None) })))
	}

	pub fn num_inputs(&self) -> Result<usize> {
		let mut num = 0;
		ortsys![unsafe KernelContext_GetInputCount(self.ptr.as_ptr(), &mut num)?];
		Ok(num)
	}

	pub fn num_outputs(&self) -> Result<usize> {
		let mut num = 0;
		ortsys![unsafe KernelContext_GetOutputCount(self.ptr.as_ptr(), &mut num)?];
		Ok(num)
	}

	pub fn allocator(&self, memory_info: &MemoryInfo) -> Result<Allocator> {
		let mut allocator_ptr = ptr::null_mut();
		ortsys![unsafe KernelContext_GetAllocator(self.ptr.as_ptr(), memory_info.ptr(), &mut allocator_ptr)?];
		Ok(unsafe { Allocator::from_raw_unchecked(allocator_ptr) })
	}

	pub fn get_resource(&self, id: ort_sys::c_int, version: ort_sys::c_int) -> Result<Option<NonNull<ort_sys::c_void>>> {
		let mut resource_ptr: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe KernelContext_GetResource(self.ptr.as_ptr(), version, id, &mut resource_ptr)?];
		Ok(NonNull::new(resource_ptr))
	}

	pub fn par_for<F>(&self, total: usize, max_num_batches: usize, f: F) -> Result<()>
	where
		F: Fn(usize) + Sync + Send
	{
		let executor = Box::new(f) as Box<dyn Fn(usize) + Sync + Send>;
		ortsys![unsafe KernelContext_ParallelFor(self.ptr.as_ptr(), Some(parallel_for_cb), total, max_num_batches, &executor as *const _ as *mut c_void)?];
		Ok(())
	}

	// TODO: STATUS_ACCESS_VIOLATION inside `KernelContext_GetScratchBuffer`. gonna assume this one is just an internal ONNX
	// Runtime bug.
	//
	// pub fn allocate<T>(&self, memory_info: &MemoryInfo, len: usize) -> Result<ScratchBuffer<T>> {
	// 	let mut buffer = ptr::null_mut();
	// 	let allocator = self.allocator(memory_info)?;
	// 	ortsys![
	// 		unsafe KernelContext_GetScratchBuffer(
	// 			self.ptr.as_ptr(),
	// 			memory_info.ptr.as_ptr(),
	// 			len * std::mem::size_of::<T>(),
	// 			&mut buffer
	// 		)?;
	// 		nonNull(buffer)
	// 	];
	// 	Ok(ScratchBuffer {
	// 		allocator,
	// 		buffer: buffer.cast::<T>(),
	// 		size: len
	// 	})
	// }

	/// Returns a pointer to the GPU compute stream (i.e. `cudaStream_t`) used by the execution provider, if this
	/// kernel's operator was configured to use said execution provider (see
	/// [`super::Operator::execution_provider_type`]).
	pub fn compute_stream(&self) -> Result<Option<NonNull<ort_sys::c_void>>> {
		let mut stream_ptr: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe KernelContext_GetGPUComputeStream(self.ptr.as_ptr(), &mut stream_ptr)?];
		Ok(NonNull::new(stream_ptr))
	}
}

impl AsPointer for KernelContext {
	type Sys = ort_sys::OrtKernelContext;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

extern "C" fn parallel_for_cb(user_data: *mut c_void, iterator: usize) {
	let executor = unsafe { &*user_data.cast::<Box<dyn Fn(usize) + Sync + Send>>() };
	executor(iterator)
}
