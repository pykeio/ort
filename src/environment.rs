//! The [`Environment`] is a process-global configuration under which [`Session`](crate::session::Session)s are created.
//!
//! With it, you can configure [default execution providers], enable/disable [telemetry], share a [global thread pool]
//! across all sessions, or add a [custom logger].
//!
//! Environments can be set up via [`ort::init`](init):
//! ```
//! # use ort::ep;
//! # fn main() -> ort::Result<()> {
//! ort::init().with_execution_providers([ep::CUDA::default().build()]).commit();
//!
//! // ... do other ort things now that our environment is set up...
//! # Ok(())
//! # }
//! ```
//!
//! With the `load-dynamic` feature, you can also load the runtime from a direct path to a DLL with
//! [`ort::init_from`](init_from):
//!
//! ```ignore
//! # use ort::ep;
//! # fn main() -> ort::Result<()> {
//! let lib_path = std::env::current_exe().unwrap().parent().unwrap().join("lib");
//! ort::init_from(lib_path.join("onnxruntime.dll"))?
//! 	.with_execution_providers([ep::CUDA::default().build()])
//! 	.commit();
//! # Ok(())
//! # }
//! ```
//!
//! If you don't configure an environment, one will be created with default settings at the first creation of a session.
//! The environment can't be re-configured after one is committed, so it's important `ort::init` come before any other
//! `ort` API for the config to take effect. Authors of libraries using `ort` should **never** have the library
//! configure the environment itself; allow the application developer to do that themselves if they wish.
//!
//! [default execution providers]: EnvironmentBuilder::with_execution_providers
//! [telemetry]: EnvironmentBuilder::with_telemetry
//! [global thread pool]: EnvironmentBuilder::with_global_thread_pool
//! [custom logger]: EnvironmentBuilder::with_logger

use alloc::{
	boxed::Box,
	string::String,
	sync::{Arc, Weak}
};
use core::{
	any::Any,
	ffi::c_void,
	fmt,
	mem::forget,
	ptr::{self, NonNull}
};

use smallvec::SmallVec;

use crate::{
	AsPointer,
	ep::ExecutionProviderDispatch,
	error::Result,
	logging::{LogLevel, LoggerFunction},
	ortsys,
	util::{Mutex, OnceLock, STACK_EXECUTION_PROVIDERS, run_on_drop, with_cstr}
};

/// Hold onto a weak reference here so that the environment is dropped when all sessions under it are. Statics don't run
/// destructors; holding a strong reference to the environment here thus leads to issues, so instead of holding the
/// environment for the entire duration of the process, we keep `Arc` references of this weak `Environment` in sessions.
/// That way, the environment is actually dropped when it is no longer needed.
static G_ENV: Mutex<Option<Weak<Environment>>> = Mutex::new(None);

static G_ENV_OPTIONS: OnceLock<EnvironmentBuilder> = OnceLock::new();

/// Holds shared global configuration for all [`Session`](crate::session::Session)s in the process.
///
/// See the [module-level documentation][self] for more information on environments. To create an environment, see
/// [`ort::init`](init) & [`ort::init_from`](init_from).
pub struct Environment {
	execution_providers: SmallVec<[ExecutionProviderDispatch; STACK_EXECUTION_PROVIDERS]>,
	ptr: NonNull<ort_sys::OrtEnv>,
	_thread_manager: Option<Arc<dyn Any>>,
	_logger: Option<LoggerFunction>
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl Environment {
	/// Sets the global log level.
	///
	/// ```
	/// # fn main() -> ort::Result<()> {
	/// # use ort::logging::LogLevel;
	/// let env = ort::environment::get_environment()?;
	///
	/// env.set_log_level(LogLevel::Warning);
	/// # Ok(())
	/// # }
	/// ```
	pub fn set_log_level(&self, level: LogLevel) {
		// technically this method should take `&mut self`, but it isn't enough of an issue to warrant putting
		// environments behind a mutex and the performance hit that comes with that
		ortsys![unsafe UpdateEnvWithCustomLogLevel(self.ptr().cast_mut(), level.into()).expect("infallible")];
	}

	/// Returns the execution providers configured by [`EnvironmentBuilder::with_execution_providers`].
	pub fn execution_providers(&self) -> &[ExecutionProviderDispatch] {
		&self.execution_providers
	}

	#[inline]
	pub(crate) fn has_global_threadpool(&self) -> bool {
		self._thread_manager.is_some()
	}
}

impl fmt::Debug for Environment {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Environment").field("ptr", &self.ptr).finish_non_exhaustive()
	}
}

impl AsPointer for Environment {
	type Sys = ort_sys::OrtEnv;

	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for Environment {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseEnv(self.ptr_mut())];
		crate::logging::drop!(Environment, self.ptr());
	}
}

/// Returns a reference to the currently active `Environment`. If one has not yet been committed (or an old environment
/// has fallen out of usage), a new environment will be created & committed.
pub fn get_environment() -> Result<Arc<Environment>> {
	let mut env_lock = G_ENV.lock();
	if let Some(env) = env_lock.as_ref()
		&& let Some(upgraded) = Weak::upgrade(env)
	{
		return Ok(upgraded);
	}

	let options = G_ENV_OPTIONS.get_or_init(EnvironmentBuilder::new);
	let env = options.create_environment().map(Arc::new)?;
	*env_lock = Some(Arc::downgrade(&env));
	Ok(env)
}

#[derive(Debug)]
pub struct GlobalThreadPoolOptions {
	ptr: *mut ort_sys::OrtThreadingOptions,
	thread_manager: Option<Arc<dyn Any>>
}

unsafe impl Send for GlobalThreadPoolOptions {}
unsafe impl Sync for GlobalThreadPoolOptions {}

impl Default for GlobalThreadPoolOptions {
	fn default() -> Self {
		let mut ptr = ptr::null_mut();
		ortsys![unsafe CreateThreadingOptions(&mut ptr).expect("failed to create threading options")];
		crate::logging::create!(GlobalThreadPoolOptions, ptr);
		Self { ptr, thread_manager: None }
	}
}

impl GlobalThreadPoolOptions {
	/// Configure the number of threads used for parallelization *between operations*.
	///
	/// This only affects sessions created with [`with_parallel_execution(true)`][wpe], and models with
	/// parallelizable branches.
	///
	/// [wpe]: crate::session::builder::SessionBuilder::with_parallel_execution
	pub fn with_inter_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetGlobalInterOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	/// Configure the number of threads used for parallelization *within a single operation*.
	///
	/// A value of `0` will use the default thread count (likely determined by the logical core count of the system).
	pub fn with_intra_threads(mut self, num_threads: usize) -> Result<Self> {
		ortsys![unsafe SetGlobalIntraOpNumThreads(self.ptr_mut(), num_threads as _)?];
		Ok(self)
	}

	/// Allow/disallow threads in the pool to [spin](https://en.wikipedia.org/wiki/Busy_waiting) when their work queues
	/// are empty.
	///
	/// If there is always work to do (i.e. if sessions are constantly running inference non-stop), allowing spinning is
	/// faster. Otherwise, spinning increases CPU usage, so it is recommended to disable it when use is infrequent.
	pub fn with_spin_control(mut self, spin_control: bool) -> Result<Self> {
		ortsys![unsafe SetGlobalSpinControl(self.ptr_mut(), if spin_control { 1 } else { 0 })?];
		Ok(self)
	}

	pub fn with_intra_affinity(mut self, affinity: impl AsRef<str>) -> Result<Self> {
		let ptr = self.ptr_mut();
		with_cstr(affinity.as_ref().as_bytes(), &|affinity| {
			ortsys![unsafe SetGlobalIntraOpThreadAffinity(ptr, affinity.as_ptr())?];
			Ok(())
		})?;
		Ok(self)
	}

	/// Disables subnormal floats by enabling the denormals-are-zero and flush-to-zero flags for all threads in the
	/// pool.
	///
	/// [Subnormal floats](https://en.wikipedia.org/wiki/Subnormal_number) are extremely small numbers very close to zero.
	/// Operations involving subnormal numbers can be very slow; enabling this flag will instead treat them as `0.0`,
	/// giving faster & more consistent performance, but lower accuracy (in cases where subnormals are involved).
	pub fn with_flush_to_zero(mut self) -> Result<Self> {
		ortsys![unsafe SetGlobalDenormalAsZero(self.ptr_mut())?];
		Ok(self)
	}

	/// Use a custom [thread manager](ThreadManager) to spawn threads for the global thread pool.
	pub fn with_thread_manager<T: ThreadManager + Any + 'static>(mut self, manager: T) -> Result<Self> {
		let manager = Arc::new(manager);
		ortsys![unsafe SetGlobalCustomThreadCreationOptions(self.ptr_mut(), (&*manager as *const T as *mut T).cast())?];
		ortsys![unsafe SetGlobalCustomCreateThreadFn(self.ptr_mut(), Some(thread_create::<T>))?];
		ortsys![unsafe SetGlobalCustomJoinThreadFn(self.ptr_mut(), Some(thread_join::<T>))?];
		self.thread_manager = Some(manager as Arc<dyn Any>);
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
		crate::logging::drop!(GlobalThreadPoolOptions, self.ptr);
	}
}

/// Used for customizing the thread spawning process of a [global thread pool](GlobalThreadPoolOptions) or [session
/// thread pool][session]. Could be used to add additional initialization/cleanup code to inference threads for
/// better debugging/error handling.
///
/// Threads spawned by `ThreadManager` should be *real* threads, spawned directly via the operating system; they
/// shouldn't be spawned in another thread pool like [`rayon`](https://crates.io/crates/rayon) because sessions have
/// their own (interfering) thread pool logic.
///
/// A very simple thread manager would be:
/// ```
/// use std::thread::{self, JoinHandle};
///
/// use ort::environment::ThreadManager;
///
/// struct StdThreadManager;
///
/// impl ThreadManager for StdThreadManager {
/// 	type Thread = JoinHandle<()>;
///
/// 	fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
/// 		Ok(thread::spawn(move || {
/// 			// ... maybe optional initialization code ...
///
/// 			// threads must call work() to actually do the work the runtime needs
/// 			work();
///
/// 			// ... maybe optional destructor code ...
/// 		}))
/// 	}
///
/// 	fn join(thread: Self::Thread) -> ort::Result<()> {
/// 		let _ = thread.join();
/// 		Ok(())
/// 	}
/// }
/// ```
///
/// [session]: crate::session::builder::SessionBuilder::with_thread_manager
pub trait ThreadManager {
	/// A handle to a spawned thread; used to [`join`](ThreadManager::join) it later.
	type Thread;

	/// Spawns a thread.
	///
	/// The newly spawned thread must call `work()`.
	fn create(&self, work: impl FnOnce() + Send + 'static) -> crate::Result<Self::Thread>;

	/// Wait for the thread to finish, like [`std::thread::JoinHandle::join`].
	fn join(thread: Self::Thread) -> crate::Result<()>;
}

pub(crate) unsafe extern "system" fn thread_create<T: ThreadManager + Any>(
	ort_custom_thread_creation_options: *mut c_void,
	ort_thread_worker_fn: ort_sys::OrtThreadWorkerFn,
	ort_worker_fn_param: *mut c_void
) -> ort_sys::OrtCustomThreadHandle {
	struct SendablePtr(*mut c_void);
	unsafe impl Send for SendablePtr {}

	let ort_worker_fn_param = SendablePtr(ort_worker_fn_param);

	let runner = || {
		let manager = unsafe { &mut *ort_custom_thread_creation_options.cast::<T>() };
		<T as ThreadManager>::create(manager, move || {
			let p = ort_worker_fn_param;
			unsafe { (ort_thread_worker_fn)(p.0) }
		})
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
	execution_providers: SmallVec<[ExecutionProviderDispatch; STACK_EXECUTION_PROVIDERS]>,
	global_thread_pool_options: Option<GlobalThreadPoolOptions>,
	logger: Option<LoggerFunction>
}

impl EnvironmentBuilder {
	pub(crate) fn new() -> Self {
		EnvironmentBuilder {
			name: String::from("default"),
			telemetry: true,
			execution_providers: SmallVec::new(),
			global_thread_pool_options: None,
			logger: None
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

	/// Enable or disable sending telemetry data.
	///
	/// Typically, only Windows builds of ONNX Runtime provided by Microsoft will have telemetry enabled.
	/// Pre-built binaries provided by pyke, binaries compiled from source, and most alternative backends won't have
	/// telemetry enabled.
	///
	/// The exact kind of telemetry data sent by ONNX Runtime can be found [here][etw].
	/// Currently, this includes (but is not limited to): ONNX graph version, model producer name & version, whether or
	/// not FP16 is used, operator domains & versions, model graph name & custom metadata, execution provider names,
	/// error messages, and the total number & time of session inference runs. The ONNX Runtime team uses this data to
	/// better understand how customers use ONNX Runtime and where performance can be improved.
	///
	/// ## `ort-web`
	///
	/// The `ort-web` alternative backend collects telemetry data by default. This telemetry data is sent to pyke.
	/// More details can be found in the `_telemetry.js` file in the root of the `ort-web` crate.
	///
	/// [etw]: https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/platform/windows/telemetry.cc
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
		self.execution_providers = execution_providers.as_ref().into();
		self
	}

	/// Enables the global thread pool for this environment.
	#[must_use = "commit() must be called in order for the environment to take effect"]
	pub fn with_global_thread_pool(mut self, options: GlobalThreadPoolOptions) -> Self {
		self.global_thread_pool_options = Some(options);
		self
	}

	/// Configures the environment to use a custom logger function.
	///
	/// ```
	/// # fn main() -> ort::Result<()> {
	/// use std::sync::Arc;
	///
	/// ort::init()
	/// 	.with_logger(Arc::new(
	/// 		|level: ort::logging::LogLevel, category: &str, id: &str, code_location: &str, message: &str| {
	/// 			// ...
	/// 		}
	/// 	))
	/// 	.commit();
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn with_logger(mut self, logger: LoggerFunction) -> Self {
		self.logger = Some(logger);
		self
	}

	pub(crate) fn create_environment(&self) -> Result<Environment> {
		let logger = self
			.logger
			.as_ref()
			.map(|c| (crate::logging::custom_logger as ort_sys::OrtLoggingFunction, c as *const _ as *mut c_void));
		#[cfg(feature = "tracing")]
		let logger = logger.or(Some((crate::logging::tracing_logger, ptr::null_mut())));

		let env_ptr = with_cstr(self.name.as_bytes(), &|name| {
			let mut env_ptr: *mut ort_sys::OrtEnv = ptr::null_mut();
			#[allow(clippy::collapsible_else_if)]
			if let Some(thread_pool_options) = self.global_thread_pool_options.as_ref() {
				if let Some((log_fn, log_ptr)) = logger {
					ortsys![
						unsafe CreateEnvWithCustomLoggerAndGlobalThreadPools(
							log_fn,
							log_ptr,
							ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
							name.as_ptr(),
							thread_pool_options.ptr(),
							&mut env_ptr
						)?;
						nonNull(env_ptr)
					];
					Ok(env_ptr)
				} else {
					ortsys![
						unsafe CreateEnvWithGlobalThreadPools(
							crate::logging::default_log_level(),
							name.as_ptr(),
							thread_pool_options.ptr(),
							&mut env_ptr
						)?;
						nonNull(env_ptr)
					];
					Ok(env_ptr)
				}
			} else {
				if let Some((log_fn, log_ptr)) = logger {
					ortsys![
						unsafe CreateEnvWithCustomLogger(
							log_fn,
							log_ptr,
							ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
							name.as_ptr(),
							&mut env_ptr
						)?;
						nonNull(env_ptr)
					];
					Ok(env_ptr)
				} else {
					ortsys![
						unsafe CreateEnv(
							crate::logging::default_log_level(),
							name.as_ptr(),
							&mut env_ptr
						)?;
						nonNull(env_ptr)
					];
					Ok(env_ptr)
				}
			}
		})?;

		let _guard = run_on_drop(|| ortsys![unsafe ReleaseEnv(env_ptr.as_ptr())]);

		if self.telemetry {
			ortsys![unsafe EnableTelemetryEvents(env_ptr.as_ptr())?];
		} else {
			ortsys![unsafe DisableTelemetryEvents(env_ptr.as_ptr())?];
		}

		forget(_guard);

		crate::logging::create!(Environment, env_ptr);
		Ok(Environment {
			execution_providers: self.execution_providers.clone(),
			ptr: env_ptr,
			_thread_manager: self
				.global_thread_pool_options
				.as_ref()
				.and_then(|options| options.thread_manager.clone()),
			_logger: self.logger.clone()
		})
	}

	/// Commit the environment configuration.
	///
	/// Returns `true` if the environment configuration was successfully committed; returns `false` if an environment
	/// has already been configured, indicating this config will not take effect.
	pub fn commit(self) -> bool {
		G_ENV_OPTIONS.try_insert_with(|| self)
	}
}

/// Creates an ONNX Runtime environment.
///
/// ```
/// # use ort::ep;
/// # fn main() -> ort::Result<()> {
/// ort::init().with_execution_providers([ep::CUDA::default().build()]).commit();
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
/// specified by `path`. Returns an error if the dylib fails to load.
///
/// This must be called before any other `ort` APIs are used in order for the correct dynamic library to be loaded.
///
/// ```no_run
/// # use ort::ep;
/// # fn main() -> ort::Result<()> {
/// let lib_path = std::env::current_exe().unwrap().parent().unwrap().join("lib");
/// ort::init_from(lib_path.join("onnxruntime.dll"))?
/// 	.with_execution_providers([ep::CUDA::default().build()])
/// 	.commit();
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
pub fn init_from<P: AsRef<std::path::Path>>(path: P) -> Result<EnvironmentBuilder> {
	crate::load_dylib_from_path(path.as_ref())?;
	Ok(EnvironmentBuilder::new())
}
