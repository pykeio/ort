use std::{
	cell::UnsafeCell,
	ffi::CString,
	sync::{
		atomic::{AtomicPtr, Ordering},
		Arc
	}
};

use tracing::debug;

use super::{
	custom_logger,
	error::{Error, Result},
	ortsys, ExecutionProviderDispatch
};
#[cfg(feature = "load-dynamic")]
use crate::G_ORT_DYLIB_PATH;

struct EnvironmentSingleton {
	cell: UnsafeCell<Option<Arc<Environment>>>
}

unsafe impl Sync for EnvironmentSingleton {}

static G_ENV: EnvironmentSingleton = EnvironmentSingleton { cell: UnsafeCell::new(None) };

#[derive(Debug)]
pub struct Environment {
	pub(crate) execution_providers: Vec<ExecutionProviderDispatch>,
	pub(crate) env_ptr: AtomicPtr<ort_sys::OrtEnv>,
	pub(crate) has_global_threadpool: bool
}

impl Environment {
	/// Loads the underlying [`ort_sys::OrtEnv`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtEnv {
		self.env_ptr.load(Ordering::Relaxed)
	}
}

impl Drop for Environment {
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!("Releasing environment");

		let env_ptr: *mut ort_sys::OrtEnv = *self.env_ptr.get_mut();

		assert_ne!(env_ptr, std::ptr::null_mut());
		ortsys![unsafe ReleaseEnv(env_ptr)];
	}
}

/// Gets a reference to the global environment, creating one if an environment has been committed yet.
pub fn get_environment() -> Result<&'static Arc<Environment>> {
	if let Some(c) = unsafe { &*G_ENV.cell.get() } {
		Ok(c)
	} else {
		debug!("Environment not yet initialized, creating a new one");
		EnvironmentBuilder::default().commit()?;

		Ok(unsafe { (*G_ENV.cell.get()).as_ref().unwrap_unchecked() })
	}
}

#[derive(Debug, Default, Clone)]
pub struct EnvironmentGlobalThreadPoolOptions {
	pub inter_op_parallelism: Option<i32>,
	pub intra_op_parallelism: Option<i32>,
	pub spin_control: Option<bool>,
	pub intra_op_thread_affinity: Option<String>
}

/// Struct used to build an `Environment`.
pub struct EnvironmentBuilder {
	name: String,
	execution_providers: Vec<ExecutionProviderDispatch>,
	global_thread_pool_options: Option<EnvironmentGlobalThreadPoolOptions>
}

impl Default for EnvironmentBuilder {
	fn default() -> Self {
		EnvironmentBuilder {
			name: "default".to_string(),
			execution_providers: vec![],
			global_thread_pool_options: None
		}
	}
}

impl EnvironmentBuilder {
	/// Configure the environment with a given name for logging purposes.
	#[must_use]
	pub fn with_name<S>(mut self, name: S) -> Self
	where
		S: Into<String>
	{
		self.name = name.into();
		self
	}

	/// Sets a list of execution providers which all sessions created in this environment will register.
	///
	/// If a session is created in this environment with [`crate::SessionBuilder::with_execution_providers`], those EPs
	/// will take precedence over the environment's EPs.
	///
	/// Execution providers will only work if the corresponding Cargo feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled will emit a warning.
	#[must_use]
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> Self {
		self.execution_providers = execution_providers.as_ref().to_vec();
		self
	}

	/// Enables the global thread pool for this environment.
	///
	/// Sessions will only use the global thread pool if they are created with
	/// [`SessionBuilder::with_disable_per_session_threads`](crate::SessionBuilder::with_disable_per_session_threads).
	#[must_use]
	pub fn with_global_thread_pool(mut self, options: EnvironmentGlobalThreadPoolOptions) -> Self {
		self.global_thread_pool_options = Some(options);
		self
	}

	/// Commit the environment configuration and set the global environment.
	pub fn commit(self) -> Result<()> {
		// drop global reference to previous environment
		if let Some(env_arc) = unsafe { (*G_ENV.cell.get()).take() } {
			drop(env_arc);
		}

		let (env_ptr, has_global_threadpool) = if let Some(global_thread_pool) = self.global_thread_pool_options {
			let mut env_ptr: *mut ort_sys::OrtEnv = std::ptr::null_mut();
			let logging_function: ort_sys::OrtLoggingFunction = Some(custom_logger);
			let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();
			let cname = CString::new(self.name.clone()).unwrap();

			let mut thread_options: *mut ort_sys::OrtThreadingOptions = std::ptr::null_mut();
			ortsys![unsafe CreateThreadingOptions(&mut thread_options) -> Error::CreateEnvironment; nonNull(thread_options)];
			if let Some(inter_op_parallelism) = global_thread_pool.inter_op_parallelism {
				ortsys![unsafe SetGlobalInterOpNumThreads(thread_options, inter_op_parallelism) -> Error::CreateEnvironment];
			}
			if let Some(intra_op_parallelism) = global_thread_pool.intra_op_parallelism {
				ortsys![unsafe SetGlobalIntraOpNumThreads(thread_options, intra_op_parallelism) -> Error::CreateEnvironment];
			}
			if let Some(spin_control) = global_thread_pool.spin_control {
				ortsys![unsafe SetGlobalSpinControl(thread_options, i32::from(spin_control)) -> Error::CreateEnvironment];
			}
			if let Some(intra_op_thread_affinity) = global_thread_pool.intra_op_thread_affinity {
				let cstr = CString::new(intra_op_thread_affinity).unwrap();
				ortsys![unsafe SetGlobalIntraOpThreadAffinity(thread_options, cstr.as_ptr()) -> Error::CreateEnvironment];
			}

			ortsys![unsafe CreateEnvWithCustomLoggerAndGlobalThreadPools(
					logging_function,
					logger_param,
					ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
					cname.as_ptr(),
					thread_options,
					&mut env_ptr
				) -> Error::CreateEnvironment; nonNull(env_ptr)];
			ortsys![unsafe ReleaseThreadingOptions(thread_options)];
			(env_ptr, true)
		} else {
			let mut env_ptr: *mut ort_sys::OrtEnv = std::ptr::null_mut();
			let logging_function: ort_sys::OrtLoggingFunction = Some(custom_logger);
			// FIXME: What should go here?
			let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();
			let cname = CString::new(self.name.clone()).unwrap();
			ortsys![unsafe CreateEnvWithCustomLogger(
					logging_function,
					logger_param,
					ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
					cname.as_ptr(),
					&mut env_ptr
				) -> Error::CreateEnvironment; nonNull(env_ptr)];
			(env_ptr, false)
		};
		debug!(env_ptr = format!("{env_ptr:?}").as_str(), "Environment created");

		unsafe {
			*G_ENV.cell.get() = Some(Arc::new(Environment {
				execution_providers: self.execution_providers,
				env_ptr: AtomicPtr::new(env_ptr),
				has_global_threadpool
			}));
		};

		Ok(())
	}
}

/// Creates an ONNX Runtime environment.
///
/// # Notes
/// - It is not required to call this function. If this is not called by the time any other `ort` APIs are used, a
///   default environment will be created.
/// - Library crates that use `ort` shouldn't create their own environment. Let downstream applications create it.
/// - In order for environment settings to apply, this must be called **before** you use other APIs like
///   [`crate::Session`], and you *must* call `.commit()` on the builder returned by this function.
#[must_use]
pub fn init() -> EnvironmentBuilder {
	EnvironmentBuilder::default()
}

/// Creates an ONNX Runtime environment, dynamically loading ONNX Runtime from the library file (`.dll`/`.so`/`.dylib`)
/// specified by `path`.
///
/// This must be called before any other `ort` APIs are used in order for the correct dynamic library to be loaded.
///
/// # Notes
/// - In order for environment settings to apply, this must be called **before** you use other APIs like
///   [`crate::Session`], and you *must* call `.commit()` on the builder returned by this function.
#[cfg(feature = "load-dynamic")]
#[cfg_attr(docsrs, doc(cfg(feature = "load-dynamic")))]
#[must_use]
pub fn init_from(path: impl ToString) -> EnvironmentBuilder {
	let _ = G_ORT_DYLIB_PATH.set(Arc::new(path.to_string()));
	EnvironmentBuilder::default()
}

#[cfg(test)]
mod tests {
	use std::sync::{atomic::Ordering, Arc, OnceLock, RwLock, RwLockWriteGuard};

	use test_log::test;

	use super::*;

	fn is_env_initialized() -> bool {
		unsafe { (*G_ENV.cell.get()).as_ref() }.is_some() && !unsafe { (*G_ENV.cell.get()).as_ref() }.unwrap().env_ptr.load(Ordering::Relaxed).is_null()
	}

	fn env_ptr() -> Option<*mut ort_sys::OrtEnv> {
		unsafe { (*G_ENV.cell.get()).as_ref() }.map(|f| f.env_ptr.load(Ordering::Relaxed))
	}

	struct ConcurrentTestRun {
		lock: Arc<RwLock<()>>
	}

	static CONCURRENT_TEST_RUN: OnceLock<ConcurrentTestRun> = OnceLock::new();

	fn single_test_run() -> RwLockWriteGuard<'static, ()> {
		CONCURRENT_TEST_RUN
			.get_or_init(|| ConcurrentTestRun { lock: Arc::new(RwLock::new(())) })
			.lock
			.write()
			.unwrap()
	}

	#[test]
	fn env_is_initialized() {
		let _run_lock = single_test_run();

		assert!(!is_env_initialized());
		assert_eq!(env_ptr(), None);

		EnvironmentBuilder::default().with_name("env_is_initialized").commit().unwrap();
		assert!(is_env_initialized());
		assert_ne!(env_ptr(), None);
	}
}
