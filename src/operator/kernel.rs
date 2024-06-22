use std::{
	ffi::{c_char, CString},
	ptr::{self, NonNull}
};

use crate::{error::status_to_result, ortsys, value::ValueRefMut, Allocator, DowncastableTarget, DynValue, Error, Result, Value, ValueRef};

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
}

pub trait GetKernelAttribute<'s> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized;
}

impl<'s> GetKernelAttribute<'s> for f32 {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_float(info, name, &mut value)]).ok()?;
		Some(value)
	}
}

impl<'s> GetKernelAttribute<'s> for i64 {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_int64(info, name, &mut value)]).ok()?;
		Some(value)
	}
}

impl<'s> GetKernelAttribute<'s> for String {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = ort_sys::size_t::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_string(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0u8; size as _];
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_string(info, name, out.as_mut_ptr().cast::<c_char>(), &mut size)]).ok()?;
		CString::from_vec_with_nul(out).ok().and_then(|c| c.into_string().ok())
	}
}

impl<'s> GetKernelAttribute<'s> for Vec<f32> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = ort_sys::size_t::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0f32; size as _];
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, out.as_mut_ptr(), &mut size)]).ok()?;
		Some(out)
	}
}

impl<'s> GetKernelAttribute<'s> for Vec<i64> {
	fn get_from(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Option<Self>
	where
		Self: Sized
	{
		let mut size = ort_sys::size_t::default();
		status_to_result(ortsys![unsafe KernelInfoGetAttributeArray_int64(info, name, ptr::null_mut(), &mut size)]).ok()?;
		let mut out = vec![0i64; size as _];
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
		status_to_result(ortsys![unsafe KernelInfoGetAttribute_tensor(info, name, allocator.ptr.as_ptr(), &mut value_ptr)]).ok()?;
		unsafe { ValueRef::new(DynValue::from_ptr(NonNull::new(value_ptr)?, None)) }
			.downcast()
			.ok()
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
		ortsys![unsafe KernelContext_GetInput(self.ptr.as_ptr(), idx as ort_sys::size_t, &mut value_ptr) -> Error::GetOperatorInput];
		Ok(NonNull::new(value_ptr.cast_mut()).map(|c| ValueRef::new(unsafe { Value::from_ptr_nodrop(c, None) })))
	}

	pub fn output(&self, idx: usize, shape: impl IntoIterator<Item = i64>) -> Result<Option<ValueRefMut<'_>>> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		let shape = shape.into_iter().collect::<Vec<i64>>();
		ortsys![unsafe KernelContext_GetOutput(self.ptr.as_ptr(), idx as ort_sys::size_t, shape.as_ptr(), shape.len() as _, &mut value_ptr) -> Error::GetOperatorOutput];
		Ok(NonNull::new(value_ptr).map(|c| ValueRefMut::new(unsafe { Value::from_ptr_nodrop(c, None) })))
	}

	/// Returns a pointer to the GPU compute stream (i.e. `cudaStream_t`) used by the execution provider, if this
	/// kernel's operator was configured to use said execution provider (see
	/// [`super::Operator::execution_provider_type`]).
	pub fn compute_stream(&self) -> Result<Option<NonNull<ort_sys::c_void>>> {
		let mut stream_ptr: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe KernelContext_GetGPUComputeStream(self.ptr.as_ptr(), &mut stream_ptr) -> Error::GetOperatorGPUComputeStream];
		Ok(NonNull::new(stream_ptr))
	}
}
