use alloc::{
	borrow::Cow,
	sync::{Arc, Weak}
};
use core::{
	any::Any,
	ptr::{self, NonNull}
};

use smallvec::SmallVec;

use crate::{
	AsPointer,
	environment::{Environment, get_environment},
	error::Result,
	logging::LoggerFunction,
	memory::MemoryInfo,
	operator::OperatorDomain,
	ortsys,
	util::with_cstr,
	value::DynValue
};

#[cfg(feature = "api-22")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
mod editable;
mod impl_commit;
mod impl_config_keys;
mod impl_options;

#[cfg(feature = "api-22")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
pub use self::editable::*;
pub use self::impl_options::*;

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
	session_options_ptr: Arc<SessionOptionsPointer>,
	memory_info: Option<Arc<MemoryInfo>>,
	operator_domains: SmallVec<[Arc<OperatorDomain>; 4]>,
	initializers: SmallVec<[Arc<DynValue>; 4]>,
	external_initializer_buffers: SmallVec<[Cow<'static, [u8]>; 4]>,
	prepacked_weights: Option<PrepackedWeights>,
	thread_manager: Option<Arc<dyn Any>>,
	logger: Option<Arc<LoggerFunction>>,
	no_global_thread_pool: bool,
	no_env_eps: bool,
	pub(crate) environment: Arc<Environment>
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
			session_options_ptr: Arc::new(SessionOptionsPointer::new(session_options_ptr)),
			memory_info: self.memory_info.clone(),
			operator_domains: self.operator_domains.clone(),
			initializers: self.initializers.clone(),
			external_initializer_buffers: self.external_initializer_buffers.clone(),
			prepacked_weights: self.prepacked_weights.clone(),
			thread_manager: self.thread_manager.clone(),
			logger: self.logger.clone(),
			no_global_thread_pool: self.no_global_thread_pool,
			no_env_eps: self.no_env_eps,
			environment: self.environment.clone()
		}
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
		let _environment = get_environment()?;

		let mut session_options_ptr: *mut ort_sys::OrtSessionOptions = ptr::null_mut();
		ortsys![unsafe CreateSessionOptions(&mut session_options_ptr)?; nonNull(session_options_ptr)];

		Ok(Self {
			session_options_ptr: Arc::new(SessionOptionsPointer::new(session_options_ptr)),
			memory_info: None,
			operator_domains: SmallVec::new(),
			initializers: SmallVec::new(),
			external_initializer_buffers: SmallVec::new(),
			prepacked_weights: None,
			thread_manager: None,
			logger: None,
			no_global_thread_pool: false,
			no_env_eps: false,
			environment: _environment
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

	/// Creates a signaler that can be used from another thread to cancel any in-progress commits.
	///
	/// ```
	/// # use ort::session::{builder::GraphOptimizationLevel, Session};
	/// # use std::{thread, time::Duration};
	/// # fn main() -> ort::Result<()> {
	/// let builder = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
	/// 	.with_intra_threads(1)?;
	///
	/// let canceler = builder.canceler();
	/// thread::spawn(move || {
	/// 	thread::sleep(Duration::from_millis(500));
	/// 	// timeout if model hasn't loaded in 500ms
	/// 	let _ = canceler.cancel();
	/// });
	///
	/// let session = builder.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	#[cfg(feature = "api-22")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn canceler(&self) -> LoadCanceler {
		LoadCanceler(Arc::downgrade(&self.session_options_ptr))
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

/// A handle which can be used to remotely terminate an in-progress session load.
///
/// See [`SessionBuilder::canceler`].
#[derive(Debug, Clone)]
#[cfg(feature = "api-22")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
pub struct LoadCanceler(Weak<SessionOptionsPointer>);

unsafe impl Send for LoadCanceler {}
unsafe impl Sync for LoadCanceler {}

#[cfg(feature = "api-22")]
impl LoadCanceler {
	/// Cancels any active session commits.
	///
	/// ```
	/// # use ort::session::{builder::GraphOptimizationLevel, Session};
	/// # use std::{thread, time::Duration};
	/// # fn main() -> ort::Result<()> {
	/// let builder = Session::builder()?
	/// 	.with_optimization_level(GraphOptimizationLevel::Level1)?
	/// 	.with_intra_threads(1)?;
	///
	/// let canceler = builder.canceler();
	/// thread::spawn(move || {
	/// 	thread::sleep(Duration::from_millis(500));
	/// 	// timeout if model hasn't loaded in 500ms
	/// 	let _ = canceler.cancel();
	/// });
	///
	/// let session = builder.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	#[cfg(feature = "api-22")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn cancel(&self) -> Result<()> {
		if let Some(ptr) = self.0.upgrade() {
			ortsys![unsafe SessionOptionsSetLoadCancellationFlag(ptr.as_ptr(), true)?];
		}
		Ok(())
	}
}

#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SessionOptionsPointer(NonNull<ort_sys::OrtSessionOptions>);

impl SessionOptionsPointer {
	#[inline]
	pub(crate) fn new(ptr: NonNull<ort_sys::OrtSessionOptions>) -> Self {
		crate::logging::create!(SessionBuilder, ptr);
		Self(ptr)
	}

	#[inline]
	pub(crate) fn as_ptr(&self) -> *mut ort_sys::OrtSessionOptions {
		self.0.as_ptr()
	}
}

impl Drop for SessionOptionsPointer {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseSessionOptions(self.0.as_ptr())];
		crate::logging::drop!(SessionBuilder, self.0.as_ptr());
	}
}
