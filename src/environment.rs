use std::{
	ffi::CString,
	sync::{atomic::AtomicPtr, Arc, Mutex}
};

use once_cell::sync::Lazy;
use tracing::{debug, error, warn};

use super::{
	custom_logger,
	error::{status_to_result, Error, Result},
	ort, ortsys, ExecutionProvider, LoggingLevel
};

static G_ENV: Lazy<Arc<Mutex<EnvironmentSingleton>>> = Lazy::new(|| {
	Arc::new(Mutex::new(EnvironmentSingleton {
		name: String::from("uninitialized"),
		env_ptr: AtomicPtr::new(std::ptr::null_mut())
	}))
});

#[derive(Debug)]
struct EnvironmentSingleton {
	name: String,
	env_ptr: AtomicPtr<ort_sys::OrtEnv>
}

#[derive(Debug, Default, Clone)]
pub struct EnvironmentGlobalThreadPoolOptions {
	pub inter_op_parallelism: Option<i32>,
	pub intra_op_parallelism: Option<i32>,
	pub spin_control: Option<bool>,
	pub intra_op_thread_affinity: Option<String>
}

/// An [`Environment`] is the main entry point of the ONNX Runtime.
///
/// Only one ONNX environment can be created per process. A singleton is used to enforce this.
///
/// Once an environment is created, a [`super::Session`] can be obtained from it.
///
/// **NOTE**: While the [`Environment`] constructor takes a `name` parameter to name the environment, only the first
/// name will be considered if many environments are created.
///
/// # Example
///
/// ```no_run
/// # use std::error::Error;
/// # use ort::{Environment, LoggingLevel};
/// # fn main() -> Result<(), Box<dyn Error>> {
/// let environment = Environment::builder().with_name("test").with_log_level(LoggingLevel::Verbose).build()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Environment {
	env: Arc<Mutex<EnvironmentSingleton>>,
	pub(crate) execution_providers: Vec<ExecutionProvider>
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl Environment {
	/// Create a new environment builder using default values
	/// (name: `default`, log level: [`LoggingLevel::Warning`])
	pub fn builder() -> EnvironmentBuilder {
		EnvironmentBuilder {
			name: "default".into(),
			log_level: LoggingLevel::Warning,
			execution_providers: Vec::new(),
			global_thread_pool_options: None
		}
	}

	/// Return the name of the current environment
	pub fn name(&self) -> String {
		self.env.lock().unwrap().name.to_string()
	}

	/// Wraps this environment in an `Arc` for use with `SessionBuilder`.
	pub fn into_arc(self) -> Arc<Environment> {
		Arc::new(self)
	}

	pub(crate) fn ptr(&self) -> *const ort_sys::OrtEnv {
		*self.env.lock().unwrap().env_ptr.get_mut()
	}
}

impl Default for Environment {
	fn default() -> Self {
		// NOTE: Because 'G_ENV' is `Lazy`, locking it will, initially, create
		//      a new Arc<Mutex<EnvironmentSingleton>> with a strong count of 1.
		//      Cloning it to embed it inside the 'Environment' to return
		//      will thus increase the strong count to 2.
		let mut environment_guard = G_ENV.lock().expect("Failed to acquire global environment lock: another thread panicked?");
		let g_env_ptr = environment_guard.env_ptr.get_mut();
		if g_env_ptr.is_null() {
			debug!("Environment not yet initialized, creating a new one");

			let mut env_ptr: *mut ort_sys::OrtEnv = std::ptr::null_mut();

			let logging_function: ort_sys::OrtLoggingFunction = Some(custom_logger);
			// FIXME: What should go here?
			let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();

			let cname = CString::new("default".to_string()).unwrap();

			status_to_result(
				ortsys![unsafe CreateEnvWithCustomLogger(logging_function, logger_param, LoggingLevel::Warning.into(), cname.as_ptr(), &mut env_ptr); nonNull(env_ptr)]
			)
			.map_err(Error::CreateEnvironment)
			.unwrap();

			debug!(env_ptr = format!("{:?}", env_ptr).as_str(), "Environment created");

			*g_env_ptr = env_ptr;
			environment_guard.name = "default".to_string();

			// NOTE: Cloning the `Lazy` 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one `Lazy` 'G_ENV'
			//          * one inside the 'Environment' returned
			Environment {
				env: G_ENV.clone(),
				execution_providers: vec![]
			}
		} else {
			// NOTE: Cloning the `Lazy` 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one `Lazy` 'G_ENV'
			//          * one inside the 'Environment' returned
			Environment {
				env: G_ENV.clone(),
				execution_providers: vec![]
			}
		}
	}
}

impl Drop for Environment {
	#[tracing::instrument]
	fn drop(&mut self) {
		debug!(global_arc_count = Arc::strong_count(&G_ENV), "Dropping environment");

		let mut environment_guard = self.env.lock().expect("Failed to acquire lock: another thread panicked?");

		// NOTE: If we drop an 'Environment' we (obviously) have _at least_
		//       one 'G_ENV' strong count (the one in the 'env' member).
		//       There is also the "original" 'G_ENV' which is a the `Lazy` global.
		//       If there is no other environment, the strong count should be two and we
		//       can properly free the sys::OrtEnv pointer.
		if Arc::strong_count(&G_ENV) == 2 {
			let release_env = ort().ReleaseEnv.unwrap();
			let env_ptr: *mut ort_sys::OrtEnv = *environment_guard.env_ptr.get_mut();

			debug!(global_arc_count = Arc::strong_count(&G_ENV), "Releasing environment");

			assert_ne!(env_ptr, std::ptr::null_mut());
			if env_ptr.is_null() {
				error!("Environment pointer is null, not dropping!");
			} else {
				unsafe { release_env(env_ptr) };
			}

			environment_guard.env_ptr = AtomicPtr::new(std::ptr::null_mut());
			environment_guard.name = String::from("uninitialized");
		}
	}
}

/// Struct used to build an environment [`Environment`].
///
/// This is ONNX Runtime's main entry point. An environment _must_ be created as the first step. An [`Environment`] can
/// only be built using `EnvironmentBuilder` to configure it.
///
/// Libraries using `ort` should **not** create an environment, as only one is allowed per process. Instead, allow the
/// user to pass their own environment to the library.
///
/// **NOTE**: If the same configuration method (for example [`EnvironmentBuilder::with_name()`] is called multiple
/// times, the last value will have precedence.
pub struct EnvironmentBuilder {
	name: String,
	log_level: LoggingLevel,
	execution_providers: Vec<ExecutionProvider>,
	global_thread_pool_options: Option<EnvironmentGlobalThreadPoolOptions>
}

impl EnvironmentBuilder {
	/// Configure the environment with a given name
	///
	/// **NOTE**: Since ONNX can only define one environment per process, creating multiple environments using multiple
	/// [`EnvironmentBuilder`]s will end up re-using the same environment internally; a new one will _not_ be created.
	/// New parameters will be ignored.
	pub fn with_name<S>(mut self, name: S) -> EnvironmentBuilder
	where
		S: Into<String>
	{
		self.name = name.into();
		self
	}

	/// Configure the environment with a given log level
	///
	/// **NOTE**: Since ONNX can only define one environment per process, creating multiple environments using multiple
	/// [`EnvironmentBuilder`]s will end up re-using the same environment internally; a new one will _not_ be created.
	/// New parameters will be ignored.
	pub fn with_log_level(mut self, log_level: LoggingLevel) -> EnvironmentBuilder {
		self.log_level = log_level;
		self
	}

	/// Configures a list of execution providers sessions created under this environment will use by default. Sessions
	/// may override these via
	/// [`SessionBuilder::with_execution_providers`](crate::SessionBuilder::with_execution_providers).
	///
	/// Execution providers are loaded in the order they are provided until a suitable execution provider is found. Most
	/// execution providers will silently fail if they are unavailable or misconfigured (see notes below), however, some
	/// may log to the console, which is sadly unavoidable. The CPU execution provider is always available, so always
	/// put it last in the list (though it is not required).
	///
	/// Execution providers will only work if the corresponding `onnxep-*` feature is enabled and ONNX Runtime was built
	/// with support for the corresponding execution provider. Execution providers that do not have their corresponding
	/// feature enabled are currently ignored.
	///
	/// Execution provider options can be specified in the second argument. Refer to ONNX Runtime's
	/// [execution provider docs](https://onnxruntime.ai/docs/execution-providers/) for configuration options. In most
	/// cases, passing `None` to configure with no options is suitable.
	///
	/// It is recommended to enable the `cuda` EP for x86 platforms and the `acl` EP for ARM platforms for the best
	/// performance, though this does mean you'll have to build ONNX Runtime for these targets. Microsoft's prebuilt
	/// binaries are built with CUDA and TensorRT support, if you built `ort` with the `onnxep-cuda` or
	/// `onnxep-tensorrt` features enabled.
	///
	/// Supported execution providers:
	/// - `cpu`: Default CPU/MLAS execution provider. Available on all platforms.
	/// - `acl`: Arm Compute Library
	/// - `cuda`: NVIDIA CUDA/cuDNN
	/// - `tensorrt`: NVIDIA TensorRT
	///
	/// ## Notes
	///
	/// - Using the CUDA/TensorRT execution providers **can terminate the process if the CUDA/TensorRT installation is
	///   misconfigured**. Configuring the execution provider will seem to work, but when you attempt to run a session,
	///   it will hard crash the process with a "stack buffer overrun" error. This can occur when CUDA/TensorRT is
	///   missing a DLL such as `zlibwapi.dll`. To prevent your app from crashing, you can check to see if you can load
	///   `zlibwapi.dll` before enabling the CUDA/TensorRT execution providers.
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProvider]>) -> EnvironmentBuilder {
		self.execution_providers = execution_providers.as_ref().to_vec();
		self
	}

	/// Enables the global thread pool for this environment.
	///
	/// Sessions will only use the global thread pool if they are created with
	/// [`SessionBuilder::with_disable_per_session_threads`](crate::SessionBuilder::with_disable_per_session_threads).
	pub fn with_global_thread_pool(mut self, options: EnvironmentGlobalThreadPoolOptions) -> EnvironmentBuilder {
		self.global_thread_pool_options = Some(options);
		self
	}

	/// Commit the configuration to a new [`Environment`].
	pub fn build(self) -> Result<Environment> {
		// NOTE: Because 'G_ENV' is a `Lazy`, locking it will, initially, create
		//      a new Arc<Mutex<EnvironmentSingleton>> with a strong count of 1.
		//      Cloning it to embed it inside the 'Environment' to return
		//      will thus increase the strong count to 2.
		let mut environment_guard = G_ENV.lock().expect("Failed to acquire global environment lock: another thread panicked?");
		let g_env_ptr = environment_guard.env_ptr.get_mut();
		if g_env_ptr.is_null() {
			debug!("Environment not yet initialized, creating a new one");

			let env_ptr = if let Some(global_thread_pool) = self.global_thread_pool_options {
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
					ortsys![unsafe SetGlobalSpinControl(thread_options, if spin_control { 1 } else { 0 }) -> Error::CreateEnvironment];
				}
				if let Some(intra_op_thread_affinity) = global_thread_pool.intra_op_thread_affinity {
					let cstr = CString::new(intra_op_thread_affinity).unwrap();
					ortsys![unsafe SetGlobalIntraOpThreadAffinity(thread_options, cstr.as_ptr()) -> Error::CreateEnvironment];
				}

				ortsys![unsafe CreateEnvWithCustomLoggerAndGlobalThreadPools(logging_function, logger_param, self.log_level.into(), cname.as_ptr(), thread_options, &mut env_ptr) -> Error::CreateEnvironment; nonNull(env_ptr)];
				ortsys![unsafe ReleaseThreadingOptions(thread_options)];
				env_ptr
			} else {
				let mut env_ptr: *mut ort_sys::OrtEnv = std::ptr::null_mut();
				let logging_function: ort_sys::OrtLoggingFunction = Some(custom_logger);
				// FIXME: What should go here?
				let logger_param: *mut std::ffi::c_void = std::ptr::null_mut();
				let cname = CString::new(self.name.clone()).unwrap();
				ortsys![unsafe CreateEnvWithCustomLogger(logging_function, logger_param, self.log_level.into(), cname.as_ptr(), &mut env_ptr) -> Error::CreateEnvironment; nonNull(env_ptr)];
				env_ptr
			};
			debug!(env_ptr = format!("{:?}", env_ptr).as_str(), "Environment created");

			*g_env_ptr = env_ptr;
			environment_guard.name = self.name;

			// NOTE: Cloning the `Lazy` 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one `Lazy` 'G_ENV'
			//          * one inside the 'Environment' returned
			Ok(Environment {
				env: G_ENV.clone(),
				execution_providers: self.execution_providers
			})
		} else {
			warn!(
				name = environment_guard.name.as_str(),
				env_ptr = format!("{:?}", environment_guard.env_ptr).as_str(),
				"Environment already initialized for this thread, reusing it",
			);

			// NOTE: Cloning the `Lazy` 'G_ENV' will increase its strong count by one.
			//       If this 'Environment' is the only one in the process, the strong count
			//       will be 2:
			//          * one `Lazy` 'G_ENV'
			//          * one inside the 'Environment' returned
			Ok(Environment {
				env: G_ENV.clone(),
				execution_providers: self.execution_providers.clone()
			})
		}
	}
}

#[cfg(test)]
mod tests {
	use std::sync::{RwLock, RwLockWriteGuard};

	use once_cell::sync::Lazy;
	use test_log::test;

	use super::*;

	fn is_env_initialized() -> bool {
		Arc::strong_count(&G_ENV) >= 2
	}

	fn env_ptr() -> *const ort_sys::OrtEnv {
		*G_ENV.lock().unwrap().env_ptr.get_mut()
	}

	struct ConcurrentTestRun {
		lock: Arc<RwLock<()>>
	}

	static CONCURRENT_TEST_RUN: Lazy<ConcurrentTestRun> = Lazy::new(|| ConcurrentTestRun { lock: Arc::new(RwLock::new(())) });

	fn single_test_run() -> RwLockWriteGuard<'static, ()> {
		CONCURRENT_TEST_RUN.lock.write().unwrap()
	}

	#[test]
	fn env_is_initialized() {
		let _run_lock = single_test_run();

		assert!(!is_env_initialized());
		assert_eq!(env_ptr(), std::ptr::null_mut());

		let env = Environment::builder()
			.with_name("env_is_initialized")
			.with_log_level(LoggingLevel::Warning)
			.build()
			.unwrap();
		assert!(is_env_initialized());
		assert_ne!(env_ptr(), std::ptr::null_mut());

		drop(env);
		assert!(!is_env_initialized());
		assert_eq!(env_ptr(), std::ptr::null_mut());
	}

	#[ignore]
	#[test]
	fn sequential_environment_creation() {
		let _concurrent_run_lock_guard = single_test_run();

		let mut prev_env_ptr = env_ptr();

		for i in 0..10 {
			let name = format!("sequential_environment_creation: {}", i);
			let env = Environment::builder()
				.with_name(name.clone())
				.with_log_level(LoggingLevel::Warning)
				.build()
				.unwrap();
			let next_env_ptr = env_ptr();
			assert_ne!(next_env_ptr, prev_env_ptr);
			prev_env_ptr = next_env_ptr;

			assert_eq!(env.name(), name);
		}
	}

	#[test]
	fn concurrent_environment_creations() {
		let _concurrent_run_lock_guard = single_test_run();

		let initial_name = String::from("concurrent_environment_creation");
		let main_env = Environment::builder()
			.with_name(&initial_name)
			.with_log_level(LoggingLevel::Warning)
			.build()
			.unwrap();
		let main_env_ptr = main_env.ptr() as usize;

		assert_eq!(main_env.name(), initial_name);
		assert_eq!(main_env.ptr() as usize, main_env_ptr);

		assert!(
			(0..10)
				.map(|t| {
					let initial_name_cloned = initial_name.clone();
					std::thread::spawn(move || {
						let name = format!("concurrent_environment_creation: {}", t);
						let env = Environment::builder()
							.with_name(name)
							.with_log_level(LoggingLevel::Warning)
							.build()
							.unwrap();

						assert_eq!(env.name(), initial_name_cloned);
						assert_eq!(env.ptr() as usize, main_env_ptr);
					})
				})
				.map(|child| child.join())
				.all(|r| Result::is_ok(&r))
		);
	}
}
