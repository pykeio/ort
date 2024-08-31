use std::{ffi::CString, os::raw::c_char, ptr::NonNull};

use crate::{char_p_to_string, error::Result, memory::Allocator, ortsys};

/// Container for model metadata, including name & producer information.
pub struct ModelMetadata<'s> {
	metadata_ptr: NonNull<ort_sys::OrtModelMetadata>,
	allocator: &'s Allocator
}

impl<'s> ModelMetadata<'s> {
	pub(crate) fn new(metadata_ptr: NonNull<ort_sys::OrtModelMetadata>, allocator: &'s Allocator) -> Self {
		ModelMetadata { metadata_ptr, allocator }
	}

	/// Gets the model description, returning an error if no description is present.
	pub fn description(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetDescription(self.metadata_ptr.as_ptr(), self.allocator.ptr.as_ptr(), &mut str_bytes)?; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				unsafe { self.allocator.free(str_bytes) };
				return Err(e);
			}
		};
		unsafe { self.allocator.free(str_bytes) };
		Ok(value)
	}

	/// Gets the model producer name, returning an error if no producer name is present.
	pub fn producer(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetProducerName(self.metadata_ptr.as_ptr(), self.allocator.ptr.as_ptr(), &mut str_bytes)?; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				unsafe { self.allocator.free(str_bytes) };
				return Err(e);
			}
		};
		unsafe { self.allocator.free(str_bytes) };
		Ok(value)
	}

	/// Gets the model name, returning an error if no name is present.
	pub fn name(&self) -> Result<String> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		ortsys![unsafe ModelMetadataGetGraphName(self.metadata_ptr.as_ptr(), self.allocator.ptr.as_ptr(), &mut str_bytes)?; nonNull(str_bytes)];

		let value = match char_p_to_string(str_bytes) {
			Ok(value) => value,
			Err(e) => {
				unsafe { self.allocator.free(str_bytes) };
				return Err(e);
			}
		};
		unsafe { self.allocator.free(str_bytes) };
		Ok(value)
	}

	/// Gets the model version, returning an error if no version is present.
	pub fn version(&self) -> Result<i64> {
		let mut ver = 0i64;
		ortsys![unsafe ModelMetadataGetVersion(self.metadata_ptr.as_ptr(), &mut ver)?];
		Ok(ver)
	}

	/// Fetch the value of a custom metadata key. Returns `Ok(None)` if the key is not found.
	pub fn custom(&self, key: &str) -> Result<Option<String>> {
		let mut str_bytes: *mut c_char = std::ptr::null_mut();
		let key_str = CString::new(key)?;
		ortsys![unsafe ModelMetadataLookupCustomMetadataMap(self.metadata_ptr.as_ptr(), self.allocator.ptr.as_ptr(), key_str.as_ptr(), &mut str_bytes)?];
		if !str_bytes.is_null() {
			let value = match char_p_to_string(str_bytes) {
				Ok(value) => value,
				Err(e) => {
					unsafe { self.allocator.free(str_bytes) };
					return Err(e);
				}
			};
			unsafe { self.allocator.free(str_bytes) };
			Ok(Some(value))
		} else {
			Ok(None)
		}
	}

	pub fn custom_keys(&self) -> Result<Vec<String>> {
		let mut keys: *mut *mut c_char = std::ptr::null_mut();
		let mut key_len = 0;
		ortsys![unsafe ModelMetadataGetCustomMetadataMapKeys(self.metadata_ptr.as_ptr(), self.allocator.ptr.as_ptr(), &mut keys, &mut key_len)?];
		if key_len != 0 && !keys.is_null() {
			let res = unsafe { std::slice::from_raw_parts(keys, key_len as usize) }
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
			Ok(vec![])
		}
	}
}

impl<'s> Drop for ModelMetadata<'s> {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseModelMetadata(self.metadata_ptr.as_ptr())];
	}
}
