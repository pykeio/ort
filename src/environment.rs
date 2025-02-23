//! An [`Environment`] is a process-global structure, under which [`Session`](crate::session::Session)s are created.
//!
//! Environments can be configured via [`ort::init`](init):
//! ```
//! # use ort::execution_providers::CUDAExecutionProvider;
//! # fn main() -> ort::Result<()> {
//! ort::init()
//! 	.with_execution_providers([CUDAExecutionProvider::default().build()])
//! 	.commit()?;
//! # Ok(())
//! # }
//! ```

use alloc::{boxed::Box, ffi::CString, string::String, vec::Vec};
use core::{
	any::Any,
	ffi::c_void,
	ptr::{self, NonNull}
};

#[cfg(feature = "load-dynamic")]
use crate::G_ORT_DYLIB_PATH;
use crate::{AsPointer, error::Result, execution_providers::ExecutionProviderDispatch, ortsys, util::OnceLock};

static G_ENV: OnceLock<Environment> = OnceLock::new();

/// An `Environment` is a process-global structure, under which [`Session`](crate::session::Session)s are created.
///
/// Environments can be used to [configure global thread pools](EnvironmentBuilder::with_global_thread_pool), in
/// which all sessions share threads from the environment's pool, and configuring [default execution
/// providers](EnvironmentBuilder::with_execution_providers) for all sessions. In the context of `ort` specifically,
/// environments are also used to configure ONNX Runtime to send log messages through the [`tracing`] crate in Rust.
///
/// For ease of use, and since sessions require an environment to be created, `ort` will automatically create an
/// environment if one is not configured via [`init`] (or [`init_from`]).
#[derive(Debug)]
pub struct Environment {
	pub(crate) execution_providers: Vec<ExecutionProviderDispatch>,
	ptr: NonNull<ort_sys::OrtEnv>,
	pub(crate) has_global_threadpool: bool,
	_thread_manager: Option<Box<dyn Any>>
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl AsPointer for Environment {
	type Sys = ort_sys::OrtEnv;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for Environment {
	fn drop(&mut self) {
		crate::debug!(ptr = ?self.ptr(), "Releasing environment");
		ortsys![unsafe ReleaseEnv(self.ptr_mut())];
	}
}

/// Gets a reference to the global environment, creating one if an environment has not been
/// [`commit`](EnvironmentBuilder::commit)ted yet.
pub fn get_environment() -> Result<&'static Environment> {
	G_ENV.get_or_try_init(|| {
		crate::debug!("Environment not yet initialized, creating a new one");
		EnvironmentBuilder::new().commit_internal()
	})
}

#[derive(Debug)]
pub struct GlobalThreadPoolOptions {
	ptr: *mut ort_sys::OrtThreadingOptions,
	thread_manager: Option<Box<dyn Any>>
}

impl Default for GlobalThreadPoolOptions {
	fn default() -> Self {
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateThreadingOptions(&mut ptr).expect("failed to create threading options")];
		Self { ptr, thread_manager: None }
	}
}

impl GlobalThreadPoolOptions {
	pub fn with_inter_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetGlobalInterOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	pub fn with_intra_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetGlobalIntraOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	pub fn with_spin_control(mut self, spin_control: bool) -> Result<Self> {
		ortsys![unsafe SetGlobalSpinControl(self.ptr_mut(), if spin_control { 1 } else { 0 })?];
		Ok(self)
	}

	pub fn with_intra_affinity(mut self, affinity: impl AsRef<str>) -> Result<Self> {
		let affinity = CString::new(affinity.as_ref())?;
		ortsys![unsafe SetGlobalIntraOpThreadAffinity(self.ptr_mut(), affinity.as_ptr())?];
		Ok(self)
	}

	pub fn with_flush_to_zero(mut self) -> Result<Self> {
		ortsys![unsafe SetGlobalDenormalAsZero(self.ptr_mut())?];
		Ok(self)
	}

	pub fn with_thread_manager<T: ThreadManager + Any + 'static>(mut self, manager: T) -> Result<Self> {
		let mut manager = Box::new(manager);
		ortsys![unsafe SetGlobalCustomThreadCreationOptions(self.ptr_mut(), (&mut *manager as *mut T).cast())?];
		ortsys![unsafe SetGlobalCustomCreateThreadFn(self.ptr_mut(), Some(thread_create::<T>))?];
		ortsys![unsafe SetGlobalCustomJoinThreadFn(self.ptr_mut(), Some(thread_join::<T>))?];
		self.thread_manager = Some(manager as Box<dyn Any>);
		Ok(self)
	}
}

impl AsPointer for GlobalThreadPoolOptions {
	type Sys = ort_sys::OrtThreadingOptions;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr
	}
}

impl Drop for GlobalThreadPoolOptions {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseThreadingOptions(self.ptr)];
	}
}

pub struct ThreadWorker {
	data: *mut c_void,
	worker: ort_sys::OrtThreadWorkerFn
}

unsafe impl Send for ThreadWorker {}

impl ThreadWorker {
	pub fn work(self) {
		unsafe { self.worker.unwrap_unchecked()(self.data) }
	}
}

pub trait ThreadManager {
	type Thread;

	fn create(&mut self, worker: ThreadWorker) -> crate::Result<Self::Thread>;

	fn join(thread: Self::Thread) -> crate::Result<()>;
}

pub(crate) unsafe extern "system" fn thread_create<T: ThreadManager + Any>(
	ort_custom_thread_creation_options: *mut c_void,
	ort_thread_worker_fn: ort_sys::OrtThreadWorkerFn,
	ort_worker_fn_param: *mut c_void
) -> ort_sys::OrtCustomThreadHandle {
	let thread_worker = ThreadWorker {
		data: ort_worker_fn_param,
		worker: ort_thread_worker_fn
	};

	let runner = || {
		let manager = unsafe { &mut *ort_custom_thread_creation_options.cast::<T>() };
		<T as ThreadManager>::create(manager, thread_worker)
	};
	#[cfg(not(feature = "std"))]
	let res = Result::<_, crate::Error>::Ok(runner()); // dumb hack
	#[cfg(feature = "std")]
	let res = std::panic::catch_unwind(runner);
	match res {
		Ok(Ok(thread)) => (Box::leak(Box::new(thread)) as *mut <T as ThreadManager>::Thread)
			.cast_const()
			.cast::<ort_sys::OrtCustomHandleType>(),
		Ok(Err(e)) => {
			crate::error!("Failed to create thread using manager: {e}");
			let _ = e;
			ptr::null()
		}
		Err(e) => {
			crate::error!("Thread manager panicked: {e:?}");
			let _ = e;
			ptr::null()
		}
	}
}

pub(crate) unsafe extern "system" fn thread_join<T: ThreadManager + Any>(ort_custom_thread_handle: ort_sys::OrtCustomThreadHandle) {
	let handle = unsafe { Box::from_raw(ort_custom_thread_handle.cast_mut().cast::<<T as ThreadManager>::Thread>()) };
	if let Err(e) = <T as ThreadManager>::join(*handle) {
		crate::error!("Failed to join thread using manager: {e}");
		let _ = e;
	}
}

/// Struct used to build an [`Environment`]; see [`crate::init`].
pub struct EnvironmentBuilder {
	name: String,
	telemetry: bool,
	execution_providers: Vec<ExecutionProviderDispatch>,
	global_thread_pool_options: Option<GlobalThreadPoolOptions>
}

impl EnvironmentBuilder {
	pub(crate) fn new() -> Self {
		EnvironmentBuilder {
			name: String::from("default"),
			telemetry: true,
			execution_providers: Vec::new(),
			global_thread_pool_options: None
		}
	}

	/// Configure the environment with a given name for logging purposes.
	#[must_use = "commit() must be called in order for the environment to take effect"]
	pub fn with_name<S>(mut self, name: S) -> Self
	where
		S: Into<String>
	{
		self.name = name.into();
		self
	}

	/// Enable or disable sending telemetry events to Microsoft.
	///
	/// Typically, only Windows builds of ONNX Runtime provided by Microsoft will have telemetry enabled.
	/// Pre-built binaries provided by pyke, or binaries compiled from source, won't have telemetry enabled.
	///
	/// The exact kind of telemetry data sent can be found [here](https://github.com/microsoft/onnxruntime/blob/v1.20.2/onnxruntime/core/platform/windows/telemetry.cc).
	/// Currently, this includes (but is not limited to): ONNX graph version, model producer name & version, whether or
	/// not FP16 is used, operator domains & versions, model graph name & custom metadata, execution provider names,
	/// error messages, and the total number & time of session inference runs. The ONNX Runtime team uses this data to
	/// better understand how customers use ONNX Runtime and where performance can be improved.
	#[must_use = "commit() must be called in order for the environment to take effect"]
	pub fn with_telemetry(mut self, enable: bool) -> Self {
		self.telemetry = enable;
		self
	}

	/// Sets a list of execution providers which all sessions created in this environment will register.
	///
	/// If a session is created in this environment with [`SessionBuilder::with_execution_providers`], those EPs
	/// will take precedence over the environment's EPs.
	///
	/// Execution providers will only work if the corresponding Cargo feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled will emit a warning.
	///
	/// [`SessionBuilder::with_execution_providers`]: crate::session::builder::SessionBuilder::with_execution_providers
	#[must_use = "commit() must be called in order for the environment to take effect"]
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> Self {
		self.execution_providers = execution_providers.as_ref().to_vec();
		self
	}

	/// Enables the global thread pool for this environment.
	#[must_use = "commit() must be called in order for the environment to take effect"]
	pub fn with_global_thread_pool(mut self, options: GlobalThreadPoolOptions) -> Self {
		self.global_thread_pool_options = Some(options);
		self
	}

	pub(crate) fn commit_internal(self) -> Result<Environment> {
		let (env_ptr, thread_manager, has_global_threadpool) = if let Some(mut thread_pool_options) = self.global_thread_pool_options {
			let mut env_ptr: *mut ort_sys::OrtEnv = ptr::null_mut();
			let cname = CString::new(self.name.clone()).unwrap_or_else(|_| unreachable!());

			#[cfg(feature = "tracing")]
			ortsys![
				unsafe CreateEnvWithCustomLoggerAndGlobalThreadPools(
					Some(crate::logging::custom_logger),
					ptr::null_mut(),
					ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
					cname.as_ptr(),
					thread_pool_options.ptr(),
					&mut env_ptr
				)?;
				nonNull(env_ptr)
			];
			#[cfg(not(feature = "tracing"))]
			ortsys![
				unsafe CreateEnvWithGlobalThreadPools(
					crate::logging::default_log_level(),
					cname.as_ptr(),
					thread_pool_options.ptr(),
					&mut env_ptr
				)?;
				nonNull(env_ptr)
			];

			let thread_manager = thread_pool_options.thread_manager.take();
			(env_ptr, thread_manager, true)
		} else {
			let mut env_ptr: *mut ort_sys::OrtEnv = ptr::null_mut();
			let cname = CString::new(self.name.clone()).unwrap_or_else(|_| unreachable!());

			#[cfg(feature = "tracing")]
			ortsys![
				unsafe CreateEnvWithCustomLogger(
					Some(crate::logging::custom_logger),
					ptr::null_mut(),
					ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
					cname.as_ptr(),
					&mut env_ptr
				)?;
				nonNull(env_ptr)
			];
			#[cfg(not(feature = "tracing"))]
			ortsys![
				unsafe CreateEnv(
					crate::logging::default_log_level(),
					cname.as_ptr(),
					&mut env_ptr
				)?;
				nonNull(env_ptr)
			];

			(env_ptr, None, false)
		};
		crate::debug!(env_ptr = alloc::format!("{env_ptr:?}").as_str(), "Environment created");

		if self.telemetry {
			ortsys![unsafe EnableTelemetryEvents(env_ptr)?];
		} else {
			ortsys![unsafe DisableTelemetryEvents(env_ptr)?];
		}

		Ok(Environment {
			execution_providers: self.execution_providers,
			// we already asserted the env pointer is non-null in the `CreateEnvWithCustomLogger` call
			ptr: unsafe { NonNull::new_unchecked(env_ptr) },
			has_global_threadpool,
			_thread_manager: thread_manager
		})
	}

	/// Commit the environment configuration.
	pub fn commit(self) -> Result<bool> {
		let env = self.commit_internal()?;
		Ok(G_ENV.try_insert(env))
	}
}

/// Creates an ONNX Runtime environment.
///
/// ```
/// # use ort::execution_providers::CUDAExecutionProvider;
/// # fn main() -> ort::Result<()> {
/// ort::init()
/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])
/// 	.commit()?;
/// # Ok(())
/// # }
/// ```
///
/// # Notes
/// - It is not required to call this function. If this is not called by the time any other `ort` APIs are used, a
///   default environment will be created.
/// - **Library crates that use `ort` shouldn't create their own environment.** Let downstream applications create it.
/// - In order for environment settings to apply, this must be called **before** you use other APIs like [`Session`],
///   and you *must* call `.commit()` on the builder returned by this function.
///
/// [`Session`]: crate::session::Session
#[must_use = "commit() must be called in order for the environment to take effect"]
pub fn init() -> EnvironmentBuilder {
	EnvironmentBuilder::new()
}

/// Creates an ONNX Runtime environment, dynamically loading ONNX Runtime from the library file (`.dll`/`.so`/`.dylib`)
/// specified by `path`.
///
/// This must be called before any other `ort` APIs are used in order for the correct dynamic library to be loaded.
///
/// ```no_run
/// # use ort::execution_providers::CUDAExecutionProvider;
/// # fn main() -> ort::Result<()> {
/// let lib_path = std::env::current_exe().unwrap().parent().unwrap().join("lib");
/// ort::init_from(lib_path.join("onnxruntime.dll"))
/// 	.with_execution_providers([CUDAExecutionProvider::default().build()])
/// 	.commit()?;
/// # Ok(())
/// # }
/// ```
///
/// # Notes
/// - In order for environment settings to apply, this must be called **before** you use other APIs like [`Session`],
///   and you *must* call `.commit()` on the builder returned by this function.
///
/// [`Session`]: crate::session::Session
#[cfg(feature = "load-dynamic")]
#[cfg_attr(docsrs, doc(cfg(feature = "load-dynamic")))]
#[must_use = "commit() must be called in order for the environment to take effect"]
pub fn init_from(path: impl ToString) -> EnvironmentBuilder {
	let _ = G_ORT_DYLIB_PATH.get_or_init(|| alloc::sync::Arc::new(path.to_string()));
	EnvironmentBuilder::new()
}
