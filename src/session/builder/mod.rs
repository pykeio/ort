use alloc::{borrow::Cow, rc::Rc, sync::Arc, vec::Vec};
use core::{
	any::Any,
	ptr::{self, NonNull}
};

use crate::{AsPointer, error::Result, memory::MemoryInfo, operator::OperatorDomain, ortsys, util::with_cstr, value::DynValue};

mod impl_commit;
mod impl_config_keys;
mod impl_options;

pub use self::impl_options::{GraphOptimizationLevel, PrepackedWeights};

/// Creates a session using the builder pattern.
///
/// Once configured, use the
/// [`SessionBuilder::commit_from_file`](crate::session::builder::SessionBuilder::commit_from_file) method to 'commit'
/// the builder configuration into a [`Session`].
///
/// ```
/// # use ort::session::{builder::GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?
/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
/// 	.with_intra_threads(1)?
/// 	.commit_from_file("tests/data/upsample.onnx")?;
/// # Ok(())
/// # }
/// ```
///
/// [`Session`]: crate::session::Session
pub struct SessionBuilder {
	session_options_ptr: NonNull<ort_sys::OrtSessionOptions>,
	memory_info: Option<Rc<MemoryInfo>>,
	operator_domains: Vec<Arc<OperatorDomain>>,
	external_initializers: Vec<Rc<DynValue>>,
	external_initializer_buffers: Vec<Cow<'static, [u8]>>,
	prepacked_weights: Option<PrepackedWeights>,
	thread_manager: Option<Rc<dyn Any>>,
	no_global_thread_pool: bool,
	no_env_eps: bool
}

impl Clone for SessionBuilder {
	fn clone(&self) -> Self {
		let mut session_options_ptr = ptr::null_mut();
		ortsys![
			unsafe CloneSessionOptions(self.ptr(), ptr::addr_of_mut!(session_options_ptr))
				.expect("error cloning session options");
			nonNull(session_options_ptr)
		];
		Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: self.memory_info.clone(),
			operator_domains: self.operator_domains.clone(),
			external_initializers: self.external_initializers.clone(),
			external_initializer_buffers: self.external_initializer_buffers.clone(),
			prepacked_weights: self.prepacked_weights.clone(),
			thread_manager: self.thread_manager.clone(),
			no_global_thread_pool: self.no_global_thread_pool,
			no_env_eps: self.no_env_eps
		}
	}
}

impl Drop for SessionBuilder {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseSessionOptions(self.ptr_mut())];
	}
}

impl SessionBuilder {
	/// Creates a new session builder.
	///
	/// ```
	/// # use ort::session::{builder::GraphOptimizationLevel, Session};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
	/// 	.with_intra_threads(1)?
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new() -> Result<Self> {
		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr)?; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr: unsafe { NonNull::new_unchecked(session_options_ptr) },
			memory_info: None,
			operator_domains: Vec::new(),
			external_initializers: Vec::new(),
			external_initializer_buffers: Vec::new(),
			prepacked_weights: None,
			thread_manager: None,
			no_global_thread_pool: false,
			no_env_eps: false
		})
	}

	pub(crate) fn add_config_entry(&mut self, key: &str, value: &str) -> Result<()> {
		let ptr = self.ptr_mut();
		with_cstr(key.as_bytes(), &|key| {
			with_cstr(value.as_bytes(), &|value| {
				ortsys![unsafe AddSessionConfigEntry(ptr, key.as_ptr(), value.as_ptr())?];
				Ok(())
			})
		})
	}

	/// Adds a custom configuration entry to the session.
	pub fn with_config_entry(mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Result<Self> {
		self.add_config_entry(key.as_ref(), value.as_ref())?;
		Ok(self)
	}
}

impl AsPointer for SessionBuilder {
	type Sys = ort_sys::OrtSessionOptions;

	fn ptr(&self) -> *const Self::Sys {
		self.session_options_ptr.as_ptr()
	}
}
