use alloc::{
	string::{String, ToString},
	vec::Vec
};
use core::{
	ffi::c_char,
	marker::PhantomData,
	ptr::{self, NonNull},
	slice
};

use crate::{
	AsPointer,
	error::Result,
	memory::Allocator,
	ortsys,
	util::{AllocatedString, char_p_to_string, with_cstr}
};

/// Container for model metadata, including name & producer information.
pub struct ModelMetadata<'s> {
	ptr: NonNull<ort_sys::OrtModelMetadata>,
	allocator: Allocator,
	_p: PhantomData<&'s ()>
}

impl ModelMetadata<'_> {
	pub(crate) unsafe fn new(ptr: NonNull<ort_sys::OrtModelMetadata>) -> Self {
		crate::logging::create!(ModelMetadata, ptr);
		ModelMetadata {
			ptr,
			allocator: Allocator::default(),
			_p: PhantomData
		}
	}

	/// Gets the model description, returning an error if no description is present.
	pub fn description(&self) -> Option<String> {
		let mut str_bytes: *mut c_char = ptr::null_mut();
		ortsys![@ort: unsafe ModelMetadataGetDescription(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut str_bytes) as Result].ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	/// Gets the description of the graph.
	pub fn graph_description(&self) -> Option<String> {
		let mut str_bytes: *mut c_char = ptr::null_mut();
		ortsys![@ort: unsafe ModelMetadataGetGraphDescription(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut str_bytes) as Result].ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	/// Gets the model producer name, returning an error if no producer name is present.
	pub fn producer(&self) -> Option<String> {
		let mut str_bytes: *mut c_char = ptr::null_mut();
		ortsys![@ort: unsafe ModelMetadataGetProducerName(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut str_bytes) as Result].ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	/// Gets the model name, returning an error if no name is present.
	pub fn name(&self) -> Option<String> {
		let mut str_bytes: *mut c_char = ptr::null_mut();
		ortsys![@ort: unsafe ModelMetadataGetGraphName(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut str_bytes) as Result].ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	/// Returns the model's domain, returning an error if no name is present.
	pub fn domain(&self) -> Option<String> {
		let mut str_bytes: *mut c_char = ptr::null_mut();
		ortsys![@ort: unsafe ModelMetadataGetDomain(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut str_bytes) as Result].ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	/// Gets the model version, returning an error if no version is present.
	pub fn version(&self) -> Option<i64> {
		let mut ver = 0i64;
		ortsys![@ort: unsafe ModelMetadataGetVersion(self.ptr.as_ptr(), &mut ver) as Result].ok()?;
		Some(ver)
	}

	/// Fetch the value of a custom metadata key. Returns `Ok(None)` if the key is not found.
	pub fn custom(&self, key: &str) -> Option<String> {
		let str_bytes = with_cstr(key.as_bytes(), &|key| {
			let mut str_bytes: *mut c_char = ptr::null_mut();
			ortsys![unsafe ModelMetadataLookupCustomMetadataMap(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), key.as_ptr(), &mut str_bytes)?];
			Ok(str_bytes)
		})
		.ok()?;
		unsafe { AllocatedString::from_ptr(str_bytes, &self.allocator) }
			.ok()
			.map(|s| s.to_string())
	}

	pub fn custom_keys(&self) -> Result<Vec<String>> {
		let mut keys: *mut *mut c_char = ptr::null_mut();
		let mut key_len = 0;
		ortsys![unsafe ModelMetadataGetCustomMetadataMapKeys(self.ptr.as_ptr(), self.allocator.ptr().cast_mut(), &mut keys, &mut key_len)?];
		if key_len != 0 && !keys.is_null() {
			let res = unsafe { slice::from_raw_parts(keys, key_len as usize) }
				.iter()
				.map(|c| {
					let res = char_p_to_string(*c);
					unsafe { self.allocator.free(*c) };
					res
				})
				.collect();
			unsafe { self.allocator.free(keys) };
			res
		} else {
			Ok(Vec::new())
		}
	}
}

impl AsPointer for ModelMetadata<'_> {
	type Sys = ort_sys::OrtModelMetadata;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for ModelMetadata<'_> {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseModelMetadata(self.ptr.as_ptr())];
		crate::logging::drop!(ModelMetadata, self.ptr);
	}
}
