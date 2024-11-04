use std::{
	path::Path,
	ptr::{self, NonNull},
	sync::Arc
};

use crate::{Allocator, Result, ortsys, util};

#[derive(Debug)]
pub(crate) struct AdapterInner {
	pub(crate) ptr: NonNull<ort_sys::OrtLoraAdapter>
}

impl Drop for AdapterInner {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseLoraAdapter(self.ptr.as_ptr())];
	}
}

#[derive(Debug, Clone)]
pub struct Adapter {
	pub(crate) inner: Arc<AdapterInner>
}

impl Adapter {
	pub fn from_file(path: impl AsRef<Path>, allocator: Option<&Allocator>) -> Result<Self> {
		let path = util::path_to_os_char(path);
		let allocator_ptr = allocator.map(|c| c.ptr()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapter(path.as_ptr(), allocator_ptr, &mut ptr)?];
		Ok(Adapter {
			inner: Arc::new(AdapterInner {
				ptr: unsafe { NonNull::new_unchecked(ptr) }
			})
		})
	}

	pub fn from_memory(bytes: &[u8], allocator: Option<&Allocator>) -> Result<Self> {
		let allocator_ptr = allocator.map(|c| c.ptr()).unwrap_or_else(ptr::null_mut);
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateLoraAdapterFromArray(bytes.as_ptr().cast(), bytes.len(), allocator_ptr, &mut ptr)?];
		Ok(Adapter {
			inner: Arc::new(AdapterInner {
				ptr: unsafe { NonNull::new_unchecked(ptr) }
			})
		})
	}

	pub fn ptr(&self) -> *mut ort_sys::OrtLoraAdapter {
		self.inner.ptr.as_ptr()
	}
}
