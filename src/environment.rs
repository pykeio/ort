//! The [`Environment`] is a process-global configuration under which [`Session`](crate::session::Session)s are created.
//!
//! With it, you can configure [default execution providers], enable/disable [telemetry], share a [global thread pool]
//! across all sessions, or add a [custom logger].
//!
//! Environments can be set up via [`ort::init`](init):
//! ```no_run
//! # use ort::ep;
//! # fn main() -> ort::Result<()> {
//! let env = ort::init()
//! 	.with_execution_providers([
//! 		#[cfg(feature = "cuda")]
//! 		ep::CUDA::default().build()
//! 	])
//! 	.build()?;
//!
//! // create sessions, etc...
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
//! let env = ort::init_from(lib_path.join("onnxruntime.dll"))?
//! 	.with_execution_providers([
//! 		#[cfg(feature = "cuda")]
//! 		ep::CUDA::default().build()
//! 	])
//! 	.build()?;
//! # Ok(())
//! # }
//! ```
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
#[cfg(all(feature = "api-22", feature = "std"))]
use std::path::Path;

use smallvec::SmallVec;

#[cfg(feature = "api-22")]
use crate::ep::ExecutionProviderLibrary;
use crate::{
	AsPointer,
	ep::ExecutionProviderDispatch,
	error::{Error, Result},
	logging::{LogLevel, LoggerFunction},
	ortsys,
	util::{OnceLock, STACK_EXECUTION_PROVIDERS, run_on_drop, with_cstr}
};

static CURRENT_ENV: OnceLock<Weak<EnvironmentInner>> = OnceLock::new();

pub(crate) struct EnvironmentInner {
	pub(crate) ptr: NonNull<ort_sys::OrtEnv>,
	execution_providers: SmallVec<[ExecutionProviderDispatch; STACK_EXECUTION_PROVIDERS]>,
	has_global_threadpool: bool,
	_thread_manager: Option<Arc<dyn Any>>,
	_logger: Option<LoggerFunction>
}

unsafe impl Send for EnvironmentInner {}
unsafe impl Sync for EnvironmentInner {}

/// Holds a handle to an **environment**, a shared global configuration for all [`Session`](crate::session::Session)s in
/// the process.
///
/// See the [module-level documentation][self] for more information on environments. To create an environment, see
/// [`EnvironmentBuilder`].
#[derive(Clone)]
pub struct Environment(Arc<EnvironmentInner>);

impl Environment {
	pub(crate) fn current() -> Option<Environment> {
		CURRENT_ENV.get().and_then(Weak::upgrade).map(Environment)
	}

	pub(crate) fn weak(&self) -> Weak<EnvironmentInner> {
		Arc::downgrade(&self.0)
	}

	/// Sets the global log level.
	///
	/// ```
	/// # use ort::logging::LogLevel;
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
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
		&self.0.execution_providers
	}

	/// Registers an execution provider library from the given `path`. Can be used to customize the path of a provider
	/// library, or load new ones ONNX Runtime was not initially compiled with.
	///
	/// `name` is semi-arbitrary - it should be unique per EP library. Adding the suffix `.virtual` to `name` allows the
	/// EP library to create virtual [devices](crate::device).
	///
	/// Returns a handle that can be used to [unregister](ExecutionProviderLibrary::unregister) the library, should it
	/// no longer be needed.
	///
	/// ```
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let _ = env.register_ep_library("CUDA", "/path/to/onnxruntime_providers_cuda.dll");
	/// # Ok(())
	/// # }
	/// ```
	#[cfg(all(feature = "api-22", feature = "std"))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "api-22", feature = "std"))))]
	pub fn register_ep_library<P: AsRef<Path>>(&self, name: impl Into<String>, path: P) -> Result<ExecutionProviderLibrary> {
		let name = name.into();
		let path = crate::util::path_to_os_char(path);
		with_cstr(name.as_bytes(), &|name| {
			ortsys![unsafe RegisterExecutionProviderLibrary(self.ptr().cast_mut(), name.as_ptr(), path.as_ptr())?];
			Ok(())
		})?;
		Ok(ExecutionProviderLibrary::new(name, self))
	}

	/// Returns an iterator over all automatically discovered [hardware device](crate::device::Device)s.
	///
	/// ```
	/// # use ort::environment::Environment;
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// for device in env.devices() {
	/// 	let hardware_device = device.hardware_device();
	/// 	println!(
	/// 		"{ep}: {vendor} {ty:?} ({id})",
	/// 		id = hardware_device.id(),
	/// 		vendor = hardware_device.vendor()?,
	/// 		ty = hardware_device.ty(),
	/// 		ep = device.ep()?
	/// 	);
	/// 	// CPUExecutionProvider: Intel CPU (0)
	/// }
	/// # Ok(())
	/// # }
	/// ```
	#[cfg(feature = "api-22")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn devices(&self) -> impl DoubleEndedIterator<Item = crate::device::Device<'_>> + '_ {
		let mut ptrs = ptr::dangling();
		let mut len = 0;
		// returns an error in minimal build because its unsupported. ignore & return empty iterator in that case
		let _ = ortsys![@ort: unsafe GetEpDevices(self.ptr().cast_mut(), &mut ptrs, &mut len) as Result];
		unsafe { core::slice::from_raw_parts(ptrs, len) }
			.iter()
			.filter_map(|c| NonNull::new(c.cast_mut()))
			.map(crate::device::Device::new)
	}

	#[inline]
	pub(crate) fn has_global_threadpool(&self) -> bool {
		self.0.has_global_threadpool
	}
}

impl fmt::Debug for Environment {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Environment").field("ptr", &self.0.ptr).finish_non_exhaustive()
	}
}

impl AsPointer for Environment {
	type Sys = ort_sys::OrtEnv;

	fn ptr(&self) -> *const Self::Sys {
		self.0.ptr.as_ptr()
	}
}

impl Drop for EnvironmentInner {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseEnv(self.ptr.as_ptr())];
		crate::logging::drop!(Environment, self.ptr);
	}
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
		#[cfg(all(feature = "std", target_arch = "x86_64"))]
		{
			if ort_sys::USING_PYKE_BINARIES && !std::is_x86_feature_detected!("avx2") {
				eprintln!(
					"WARNING: This CPU does not support AVX2, which is required by ort's prebuilt ONNX Runtime binaries. The app will likely crash with an illegal instruction error; use a custom build of ONNX Runtime to fix."
				);
			}
		}

		EnvironmentBuilder {
			name: String::from("default"),
			telemetry: true,
			execution_providers: SmallVec::new(),
			global_thread_pool_options: None,
			logger: None
		}
	}

	/// Configure the environment with a given name for logging purposes.
	#[must_use]
	pub fn with_name<S>(mut self, name: S) -> Self
	where
		S: Into<String>
	{
		self.name = name.into();
		self
	}

	/// Enable or disable sending telemetry data.
	///
	/// This is enabled by default in Microsoft-provided builds of ONNX Runtime. Pre-built binaries provided by pyke
	/// (the default downloaded by `ort`), binaries compiled from source, and most alternative backends won't have
	/// telemetry enabled by default.
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
	/// [etw]: https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/platform/windows/telemetry.cc
	#[must_use]
	pub fn with_telemetry(mut self, enable: bool) -> Self {
		self.telemetry = enable;
		self
	}

	/// Sets a list of execution providers which all sessions created in this environment will register.
	///
	/// If a session is created in this environment with [`SessionBuilder::with_execution_providers`], those EPs
	/// will be registered first, before the environment's EPs.
	///
	/// Execution providers will only work if the corresponding Cargo feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled will emit a warning.
	///
	/// [`SessionBuilder::with_execution_providers`]: crate::session::builder::SessionBuilder::with_execution_providers
	#[must_use]
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> Self {
		self.execution_providers = execution_providers.as_ref().into();
		self
	}

	/// Enables the global thread pool for this environment.
	#[must_use]
	pub fn with_global_thread_pool(mut self, options: GlobalThreadPoolOptions) -> Self {
		self.global_thread_pool_options = Some(options);
		self
	}

	/// Configures the environment to use a custom logger function.
	///
	/// ```no_run
	/// # fn main() -> ort::Result<()> {
	/// use std::sync::Arc;
	///
	/// let env = ort::init()
	/// 	.with_logger(Arc::new(
	/// 		|level: ort::logging::LogLevel, category: &str, id: &str, code_location: &str, message: &str| {
	/// 			// ...
	/// 		}
	/// 	))
	/// 	.build()?;
	/// # 	Ok(())
	/// # }
	/// ```
	#[must_use]
	pub fn with_logger(mut self, logger: LoggerFunction) -> Self {
		self.logger = Some(logger);
		self
	}

	/// Builds the environment.
	pub fn build(self) -> Result<Environment> {
		if CURRENT_ENV.get().is_some() {
			return Err(Error::new("only one environment is allowed per process"));
		}

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

		let inner = Arc::new(EnvironmentInner {
			execution_providers: self.execution_providers.clone(),
			ptr: env_ptr,
			has_global_threadpool: self.global_thread_pool_options.is_some(),
			_thread_manager: self
				.global_thread_pool_options
				.as_ref()
				.and_then(|options| options.thread_manager.clone()),
			_logger: self.logger.clone()
		});
		CURRENT_ENV.try_insert_with(|| Arc::downgrade(&inner));
		Ok(Environment(inner))
	}
}

/// Creates an ONNX Runtime environment.
///
/// ```no_run
/// # use ort::ep;
/// # fn main() -> ort::Result<()> {
/// let env = ort::init()
/// 	.with_execution_providers([
/// 		#[cfg(feature = "cuda")]
/// 		ep::CUDA::default().build()
/// 	])
/// 	.build()?;
/// # Ok(())
/// # }
/// ```
///
/// [`Session`]: crate::session::Session
#[must_use = "an environment is required to create sessions"]
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
/// # fn main() -> Result<(), ort::LoadDynamicError> {
/// let lib_path = std::env::current_exe().unwrap().parent().unwrap().join("lib");
/// let env = ort::init_from(lib_path.join("onnxruntime.dll"))?
/// 	.with_execution_providers([
/// 		#[cfg(feature = "cuda")]
/// 		ep::CUDA::default().build()
/// 	])
/// 	.build()?;
/// # Ok(())
/// # }
/// ```
///
/// [`Session`]: crate::session::Session
#[cfg(all(feature = "load-dynamic", not(target_arch = "wasm32")))]
#[cfg_attr(docsrs, doc(cfg(feature = "load-dynamic")))]
#[must_use = "an environment is required to create sessions"]
pub fn init_from<P: AsRef<std::path::Path>>(path: P) -> Result<EnvironmentBuilder, crate::LoadDynamicError> {
	crate::load_dynamic::init(path.as_ref())?;
	Ok(EnvironmentBuilder::new())
}
