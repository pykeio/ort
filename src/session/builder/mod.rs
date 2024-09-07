use std::{
	ffi::CString,
	ptr::{self, NonNull},
	rc::Rc,
	sync::Arc
};

use crate::{
	error::{assert_non_null_pointer, status_to_result, Result},
	memory::MemoryInfo,
	operator::OperatorDomain,
	ortsys
};

mod impl_commit;
mod impl_config_keys;
mod impl_options;

pub use self::impl_options::GraphOptimizationLevel;

/// Creates a session using the builder pattern.
///
/// Once configured, use the [`SessionBuilder::commit_from_file`](crate::SessionBuilder::commit_from_file)
/// method to 'commit' the builder configuration into a [`Session`].
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.commit_from_file("tests/data/upsample.onnx")?;
/// # Ok(())
/// # }
/// ```
pub struct SessionBuilder {
	pub(crate) session_options_ptr: NonNull<ort_sys::OrtSessionOptions>,
	memory_info: Option<Rc<MemoryInfo>>,
	operator_domains: Vec<Arc<OperatorDomain>>
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		status_to_result(ortsys![unsafe CloneSessionOptions(self.session_options_ptr.as_ptr(), ptr::addr_of_mut!(session_options_ptr))])
			.expect("error cloning session options");
		assert_non_null_pointer(session_options_ptr, "OrtSessionOptions").expect("Cloned session option pointer is null");
		Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: self.memory_info.clone(),
			operator_domains: self.operator_domains.clone()
		}
	}
}

impl Drop for SessionBuilder {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseSessionOptions(self.session_options_ptr.as_ptr())];
	}
}

impl SessionBuilder {
	/// Creates a new session builder.
	///
	/// ```
	/// # use ort::{GraphOptimizationLevel, Session};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
	/// 	.with_intra_threads(1)?
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new() -> Result<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = std::ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr)?; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: None,
			operator_domains: Vec::new()
		})
	}

	pub(crate) fn add_config_entry(&mut self, key: &str, value: &str) -> Result<()> {
		let key = CString::new(key)?;
		let value = CString::new(value)?;
		ortsys![unsafe AddSessionConfigEntry(self.session_options_ptr.as_ptr(), key.as_ptr(), value.as_ptr())?];
		Ok(())
	}
}
