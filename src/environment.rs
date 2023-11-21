use std::{
	ffi::CString,
	sync::{atomic::AtomicPtr, OnceLock}
};

use tracing::debug;

use super::{
	custom_logger,
	error::{Error, Result},
	ortsys, ExecutionProviderDispatch, LoggingLevel
};

static G_ENV: OnceLock<EnvironmentSingleton> = OnceLock::new();

#[derive(Debug)]
pub(crate) struct EnvironmentSingleton {
	pub(crate) execution_providers: Vec<ExecutionProviderDispatch>,
	pub(crate) env_ptr: AtomicPtr<ort_sys::OrtEnv>
}

pub(crate) fn get_environment() -> Result<&'static EnvironmentSingleton> {
	if G_ENV.get().is_none() {
		EnvironmentBuilder::default().commit()?;
		Ok(G_ENV.get().unwrap())
	} else {
		Ok(unsafe { G_ENV.get().unwrap_unchecked() })
	}
}

#[derive(Debug, Default, Clone)]
pub struct EnvironmentGlobalThreadPoolOptions {
	pub inter_op_parallelism: Option<i32>,
	pub intra_op_parallelism: Option<i32>,
	pub spin_control: Option<bool>,
	pub intra_op_thread_affinity: Option<String>
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
	execution_providers: Vec<ExecutionProviderDispatch>,
	global_thread_pool_options: Option<EnvironmentGlobalThreadPoolOptions>
}

impl Default for EnvironmentBuilder {
	fn default() -> Self {
		EnvironmentBuilder {
			name: "default".to_string(),
			log_level: LoggingLevel::Error,
			execution_providers: vec![],
			global_thread_pool_options: None
		}
	}
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
	pub fn with_execution_providers(mut self, execution_providers: impl AsRef<[ExecutionProviderDispatch]>) -> EnvironmentBuilder {
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
	pub fn commit(self) -> Result<()> {
		if G_ENV.get().is_none() {
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

			let _ = G_ENV.set(EnvironmentSingleton {
				execution_providers: self.execution_providers,
				env_ptr: AtomicPtr::new(env_ptr)
			});
		}
		Ok(())
	}
}

pub fn init() -> EnvironmentBuilder {
	EnvironmentBuilder::default()
}

#[cfg(test)]
mod tests {
	use std::sync::{atomic::Ordering, Arc, RwLock, RwLockWriteGuard};

	use once_cell::sync::Lazy;
	use test_log::test;

	use super::*;

	fn is_env_initialized() -> bool {
		G_ENV.get().is_some() && !G_ENV.get().unwrap().env_ptr.load(Ordering::Relaxed).is_null()
	}

	fn env_ptr() -> Option<*mut ort_sys::OrtEnv> {
		G_ENV.get().map(|f| f.env_ptr.load(Ordering::Relaxed))
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
		assert_eq!(env_ptr(), None);

		EnvironmentBuilder::default()
			.with_name("env_is_initialized")
			.with_log_level(LoggingLevel::Warning)
			.commit()
			.unwrap();
		assert!(is_env_initialized());
		assert_ne!(env_ptr(), None);
	}
}
