use alloc::{ffi::CString, string::String, vec, vec::Vec};
use core::{
	ffi::c_char,
	ptr::{self, NonNull}
};

use crate::{
	AsPointer, Error, Result,
	memory::Allocator,
	ortsys,
	util::{with_cstr, with_cstr_ptr_array},
	value::{DowncastableTarget, DynValue, ValueRef}
};

#[derive(Debug)]
#[repr(transparent)] // required for `editor::Node::new`
pub struct Attribute(NonNull<ort_sys::OrtOpAttr>);

impl Attribute {
	pub fn new(name: impl AsRef<str>, value: impl ToAttribute) -> Result<Self> {
		with_cstr(name.as_ref().as_bytes(), &|name| unsafe { value.to_attribute(name.as_ptr()) }.map(Self))
	}
}

impl Drop for Attribute {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseOpAttr(self.0.as_ptr())];
	}
}

pub trait FromKernelAttributes<'s> {
	/// Reads the value of the attribute from an [`ort_sys::OrtKernelInfo`] given its C name.
	#[doc(hidden)]
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized;

	private_trait!();
}

pub trait FromOpAttr {
	#[doc(hidden)]
	fn attr_type() -> ort_sys::OrtOpAttrType;

	/// Reads the value of the attribute via [`ort_sys::OrtApi::ReadOpAttr`] using the known `len`gth of the value.
	#[doc(hidden)]
	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, len: usize) -> Result<Self>
	where
		Self: Sized;

	private_trait!();
}

pub trait ToAttribute {
	#[doc(hidden)]
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized;

	private_trait!();
}

impl FromKernelAttributes<'_> for f32 {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		ortsys![unsafe KernelInfoGetAttribute_float(info, name, &mut value)?];
		Ok(value)
	}

	private_impl!();
}

impl FromOpAttr for f32 {
	fn attr_type() -> ort_sys::OrtOpAttrType {
		ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOAT
	}

	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, mut len: usize) -> Result<Self>
	where
		Self: Sized
	{
		let mut out = 0.0_f32;
		ortsys![unsafe ReadOpAttr(attr, ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOAT, (&mut out as *mut f32).cast(), size_of::<f32>(), &mut len)?];
		assert_eq!(len, size_of::<f32>());
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for f32 {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		let mut out = ptr::null_mut();
		ortsys![
			unsafe CreateOpAttr(
				name,
				(self as *const f32).cast(),
				1,
				ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOAT,
				&mut out
			)?;
			nonNull(out)
		];
		Ok(out)
	}

	private_impl!();
}

impl FromKernelAttributes<'_> for i64 {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		let mut value = Self::default();
		ortsys![unsafe KernelInfoGetAttribute_int64(info, name, &mut value)?];
		Ok(value)
	}

	private_impl!();
}

impl FromOpAttr for i64 {
	fn attr_type() -> ort_sys::OrtOpAttrType {
		ort_sys::OrtOpAttrType::ORT_OP_ATTR_INT
	}

	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, mut len: usize) -> Result<Self>
	where
		Self: Sized
	{
		let mut out = 0_i64;
		ortsys![unsafe ReadOpAttr(attr, ort_sys::OrtOpAttrType::ORT_OP_ATTR_INT, (&mut out as *mut i64).cast(), size_of::<i64>(), &mut len)?];
		assert_eq!(len, size_of::<i64>());
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for i64 {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		let mut out = ptr::null_mut();
		ortsys![
			unsafe CreateOpAttr(
				name,
				(self as *const i64).cast(),
				1,
				ort_sys::OrtOpAttrType::ORT_OP_ATTR_INT,
				&mut out
			)?;
			nonNull(out)
		];
		Ok(out)
	}

	private_impl!();
}

impl FromKernelAttributes<'_> for String {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		ortsys![unsafe KernelInfoGetAttribute_string(info, name, ptr::null_mut(), &mut size)?];
		let mut out = vec![0u8; size];
		ortsys![unsafe KernelInfoGetAttribute_string(info, name, out.as_mut_ptr().cast::<c_char>(), &mut size)?];
		let string = CString::from_vec_with_nul(out)?;
		Ok(string.into_string()?)
	}

	private_impl!();
}

impl FromOpAttr for String {
	fn attr_type() -> ort_sys::OrtOpAttrType {
		ort_sys::OrtOpAttrType::ORT_OP_ATTR_STRING
	}

	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, mut len: usize) -> Result<Self>
	where
		Self: Sized
	{
		let mut out = vec![0_u8; len / size_of::<u8>()];
		ortsys![unsafe ReadOpAttr(attr, ort_sys::OrtOpAttrType::ORT_OP_ATTR_STRING, out.as_mut_ptr().cast(), len, &mut len)?];
		assert_eq!(out.len(), len / size_of::<u8>());
		CString::from_vec_with_nul(out)
			.map_err(|_| Error::new("invalid string"))
			.and_then(|f| f.into_string().map_err(|_| Error::new("invalid string")))
	}

	private_impl!();
}

impl ToAttribute for String {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		with_cstr(self.as_bytes(), &|contents| {
			let mut out = ptr::null_mut();
			ortsys![
				unsafe CreateOpAttr(
					name,
					contents.as_ptr().cast(),
					1,
					ort_sys::OrtOpAttrType::ORT_OP_ATTR_STRING,
					&mut out
				)?;
				nonNull(out)
			];
			Ok(out)
		})
	}

	private_impl!();
}

impl FromKernelAttributes<'_> for Vec<f32> {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, ptr::null_mut(), &mut size)?];
		let mut out = vec![0f32; size];
		ortsys![unsafe KernelInfoGetAttributeArray_float(info, name, out.as_mut_ptr(), &mut size)?];
		Ok(out)
	}

	private_impl!();
}

impl FromOpAttr for Vec<f32> {
	fn attr_type() -> ort_sys::OrtOpAttrType {
		ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOATS
	}

	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, mut len: usize) -> Result<Self>
	where
		Self: Sized
	{
		let mut out = vec![0.0_f32; len / size_of::<f32>()];
		ortsys![unsafe ReadOpAttr(attr, ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOATS, out.as_mut_ptr().cast(), len, &mut len)?];
		assert_eq!(out.len(), len / size_of::<f32>());
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for &[f32] {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		let mut out = ptr::null_mut();
		ortsys![
			unsafe CreateOpAttr(
				name,
				self.as_ptr().cast(),
				self.len() as _,
				ort_sys::OrtOpAttrType::ORT_OP_ATTR_FLOATS,
				&mut out
			)?;
			nonNull(out)
		];
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for Vec<f32> {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		self.as_slice().to_attribute(name)
	}

	private_impl!();
}

impl FromKernelAttributes<'_> for Vec<i64> {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		let mut size = 0;
		ortsys![unsafe KernelInfoGetAttributeArray_int64(info, name, ptr::null_mut(), &mut size)?];
		let mut out = vec![0i64; size];
		ortsys![unsafe KernelInfoGetAttributeArray_int64(info, name, out.as_mut_ptr(), &mut size)?];
		Ok(out)
	}

	private_impl!();
}

impl FromOpAttr for Vec<i64> {
	fn attr_type() -> ort_sys::OrtOpAttrType {
		ort_sys::OrtOpAttrType::ORT_OP_ATTR_INTS
	}

	unsafe fn from_op_attr(attr: *const ort_sys::OrtOpAttr, mut len: usize) -> Result<Self>
	where
		Self: Sized
	{
		let mut out = vec![0_i64; len / size_of::<i64>()];
		ortsys![unsafe ReadOpAttr(attr, ort_sys::OrtOpAttrType::ORT_OP_ATTR_INTS, out.as_mut_ptr().cast(), len, &mut len)?];
		assert_eq!(out.len(), len / size_of::<i64>());
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for &[i64] {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		let mut out = ptr::null_mut();
		ortsys![
			unsafe CreateOpAttr(
				name,
				self.as_ptr().cast(),
				self.len() as _,
				ort_sys::OrtOpAttrType::ORT_OP_ATTR_INTS,
				&mut out
			)?;
			nonNull(out)
		];
		Ok(out)
	}

	private_impl!();
}

impl ToAttribute for Vec<i64> {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		self.as_slice().to_attribute(name)
	}

	private_impl!();
}

impl ToAttribute for &[String] {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		with_cstr_ptr_array(self, &|strings| {
			let mut out = ptr::null_mut();
			ortsys![
				unsafe CreateOpAttr(
					name,
					strings.as_ptr().cast(),
					strings.len() as _,
					ort_sys::OrtOpAttrType::ORT_OP_ATTR_STRINGS,
					&mut out
				)?;
				nonNull(out)
			];
			Ok(out)
		})
	}

	private_impl!();
}

impl ToAttribute for &[&str] {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		with_cstr_ptr_array(self, &|strings| {
			let mut out = ptr::null_mut();
			ortsys![
				unsafe CreateOpAttr(
					name,
					strings.as_ptr().cast(),
					strings.len() as _,
					ort_sys::OrtOpAttrType::ORT_OP_ATTR_STRINGS,
					&mut out
				)?;
				nonNull(out)
			];
			Ok(out)
		})
	}

	private_impl!();
}

impl ToAttribute for Vec<String> {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		self.as_slice().to_attribute(name)
	}

	private_impl!();
}

impl ToAttribute for Vec<&str> {
	unsafe fn to_attribute(&self, name: *const ort_sys::c_char) -> Result<NonNull<ort_sys::OrtOpAttr>>
	where
		Self: Sized
	{
		self.as_slice().to_attribute(name)
	}

	private_impl!();
}

impl<'s, T: DowncastableTarget> FromKernelAttributes<'s> for ValueRef<'s, T> {
	unsafe fn from_info(info: *mut ort_sys::OrtKernelInfo, name: *const ort_sys::c_char) -> Result<Self>
	where
		Self: Sized
	{
		// TODO: This should probably be customizable - docs say the allocator is required for "internal tensor state", but it's
		// not clear if this also includes tensor data (and thus it should instead be allocated on an appropriate device).
		let allocator = Allocator::default();

		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();
		ortsys![unsafe KernelInfoGetAttribute_tensor(info, name, allocator.ptr().cast_mut(), &mut value_ptr)?; nonNull(value_ptr)];
		unsafe { ValueRef::new(DynValue::from_ptr(value_ptr, None)) }.downcast()
	}

	private_impl!();
}
