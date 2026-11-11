use alloc::{sync::Arc, vec::Vec};
use core::{
	any::Any,
	ffi::c_void,
	marker::PhantomData,
	mem,
	ptr::{self, NonNull}
};
#[cfg(feature = "std")]
use std::path::Path;

use smallvec::SmallVec;

#[cfg(feature = "api-22")]
use super::EditableSession;
use super::SessionBuilder;
#[cfg(any(all(target_family = "wasm", target_os = "unknown"), feature = "std"))]
use crate::error::{Error, ErrorCode};
#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
use crate::util::OsCharArray;
use crate::{
	AsPointer,
	ep::apply_execution_providers,
	error::Result,
	memory::Allocator,
	ortsys,
	session::{InMemorySession, Outlet, Session, SharedSessionInner, io},
	util::run_on_drop
};

impl SessionBuilder {
	/// Loads an ONNX model from a file and builds the session.
	#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn commit_from_file<P>(&mut self, model_path: P) -> Result<Session>
	where
		P: AsRef<Path>
	{
		let model_path = model_path.as_ref();
		if !model_path.exists() {
			return Err(Error::new_with_code(ErrorCode::NoSuchFile, format!("File at `{}` does not exist", model_path.display())));
		}

		let model_path = crate::util::path_to_os_char(model_path);
		self.commit_from_file_inner(model_path.as_ref())
	}

	#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
	fn commit_from_file_inner(&mut self, model_path: &<OsCharArray as core::ops::Deref>::Target) -> Result<Session> {
		self.pre_commit()?;

		let session_ptr = if let Some(prepacked_weights) = self.prepacked_weights.as_ref() {
			let mut session_ptr = ptr::null_mut();
			ortsys![unsafe CreateSessionWithPrepackedWeightsContainer(self.environment.ptr(), model_path.as_ptr(), self.ptr(), prepacked_weights.ptr().cast_mut(), &mut session_ptr)?; nonNull(session_ptr)];
			session_ptr
		} else {
			let mut session_ptr = ptr::null_mut();
			ortsys![unsafe CreateSession(self.environment.ptr(), model_path.as_ptr(), self.ptr(), &mut session_ptr)?; nonNull(session_ptr)];
			session_ptr
		};

		self.commit_finalize(session_ptr)
	}

	/// Load an ONNX graph directly from memory and commit the session.
	///
	/// For models in the `.ort` format, `session.use_ort_model_bytes_directly` is enabled, so no copy is performed. For
	/// more information, see [the ONNX Runtime docs](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#load-ort-format-model-from-an-in-memory-byte-array).
	///
	/// If you need to store the model bytes and the [`InMemorySession`] in the same struct, there are crates that can
	/// facilitate creating self-referential structs, such as [`ouroboros`](https://github.com/joshua-maros/ouroboros).
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn commit_from_memory_directly<'m>(&mut self, model_bytes: &'m [u8]) -> Result<InMemorySession<'m>> {
		const KEYS: [&str; 2] = ["session.use_ort_model_bytes_directly", "session.use_ort_model_bytes_for_initializers"];

		// Enable zero-copy deserialization for models in `.ort` format, only for this commit; if left on, later
		// `commit_from_memory` calls would borrow the model bytes without a lifetime tying the session to them.
		for key in KEYS {
			let _ = self.add_config_entry(key, "1");
		}
		let session = self.commit_from_memory(model_bytes);
		for key in KEYS {
			let _ = self.add_config_entry(key, "0");
		}

		Ok(InMemorySession {
			session: session?,
			phantom: PhantomData
		})
	}

	/// Load an ONNX graph from memory and commit the session.
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn commit_from_memory(&mut self, model_bytes: &[u8]) -> Result<Session> {
		self.pre_commit()?;

		let model_data = model_bytes.as_ptr().cast::<c_void>();
		let model_data_length = model_bytes.len();
		let session_ptr = if let Some(prepacked_weights) = self.prepacked_weights.as_ref() {
			let mut session_ptr = ptr::null_mut();
			ortsys![
				unsafe CreateSessionFromArrayWithPrepackedWeightsContainer(self.environment.ptr(), model_data, model_data_length, self.ptr(), prepacked_weights.ptr().cast_mut(), &mut session_ptr)?;
				nonNull(session_ptr)
			];
			session_ptr
		} else {
			let mut session_ptr = ptr::null_mut();
			ortsys![
				unsafe CreateSessionFromArray(self.environment.ptr(), model_data, model_data_length, self.ptr(), &mut session_ptr)?;
				nonNull(session_ptr)
			];
			session_ptr
		};

		self.commit_finalize(session_ptr)
	}

	/// Downloads a pre-trained ONNX model from the given URL and builds the session.
	#[cfg(all(target_family = "wasm", target_os = "unknown"))]
	pub async fn commit_from_url(&mut self, model_url: impl AsRef<str>) -> Result<Session> {
		self.pre_commit()?;

		let mut session_ptr = ptr::null_mut();
		let status = ortsys![unsafe CreateSession(self.environment.ptr(), model_url.as_ref(), self.ptr(), &mut session_ptr)].await;
		unsafe { Error::result_from_status(status) }?;

		let Some(session_ptr) = NonNull::new(session_ptr) else {
			return Err(Error::new(alloc::format!("Session creation failed with unknown error")));
		};

		self.commit_finalize(session_ptr)
	}

	/// Load an ONNX graph from memory and commit the session.
	#[cfg(all(target_family = "wasm", target_os = "unknown"))]
	pub async fn commit_from_memory(&mut self, model_bytes: &[u8]) -> Result<Session> {
		self.pre_commit()?;

		let mut session_ptr = ptr::null_mut();
		let status = ortsys![unsafe CreateSessionFromArray(self.environment.ptr(), model_bytes.as_ref(), self.ptr(), &mut session_ptr)].await;
		unsafe { Error::result_from_status(status) }?;

		let Some(session_ptr) = NonNull::new(session_ptr) else {
			return Err(Error::new(alloc::format!("Session creation failed with unknown error")));
		};

		self.commit_finalize(session_ptr)
	}

	pub(crate) fn pre_commit(&mut self) -> Result<()> {
		if !self.no_env_eps {
			let env = self.environment.clone();
			apply_execution_providers(self, env.execution_providers(), "environment")?;
		}

		if self.environment.has_global_threadpool() && !self.no_global_thread_pool {
			ortsys![unsafe DisablePerSessionThreads(self.ptr_mut())?];
		}

		Ok(())
	}

	pub(crate) fn commit_finalize(&self, ptr: NonNull<ort_sys::OrtSession>) -> Result<Session> {
		// Release the session if anything below fails; `SharedSessionInner` takes over once it's built.
		let session_guard = run_on_drop(|| ortsys![unsafe ReleaseSession(ptr.as_ptr())]);

		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = ptr::null_mut();
				ortsys![unsafe CreateAllocator(ptr.as_ptr(), info.ptr(), &mut allocator_ptr)?; nonNull(allocator_ptr)];
				unsafe { Allocator::from_raw(allocator_ptr, false) }
			}
			None => Allocator::default()
		};

		// Extract input and output properties
		let num_inputs = io::extract_io_count(ortsys![SessionGetInputCount], ptr)?;
		let num_outputs = io::extract_io_count(ortsys![SessionGetOutputCount], ptr)?;
		let inputs = (0..num_inputs)
			.map(|i| io::extract_input(ptr, &allocator, i))
			.collect::<Result<Vec<Outlet>>>()?;
		let outputs = (0..num_outputs)
			.map(|i| io::extract_output(ptr, &allocator, i))
			.collect::<Result<Vec<Outlet>>>()?;

		let extras = self.extras();

		mem::forget(session_guard);
		crate::logging::create!(Session, ptr);

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr: ptr,
				allocator,
				_extras: extras,
				_environment: self.environment.clone()
			}),
			inputs,
			outputs
		})
	}

	pub(crate) fn extras(&self) -> SmallVec<[Arc<dyn Any>; 4]> {
		let mut extras: SmallVec<[Arc<dyn Any>; 4]> = SmallVec::new();
		for op_domain in self.operator_domains.iter().cloned() {
			extras.push(op_domain as Arc<dyn Any>);
		}
		for initializer in self.initializers.iter().cloned() {
			extras.push(initializer as Arc<dyn Any>);
		}
		if let Some(prepacked_weights) = self.prepacked_weights.clone() {
			extras.push(prepacked_weights.inner as Arc<dyn Any>);
		}
		if let Some(thread_manager) = self.thread_manager.clone() {
			extras.push(thread_manager as Arc<dyn Any>);
		}
		if let Some(logger) = self.logger.clone() {
			extras.push(logger as Arc<dyn Any>);
		}
		extras
	}

	#[cfg(all(feature = "std", feature = "api-22"))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "std", feature = "api-22"))))]
	pub fn edit_from_file<P>(&mut self, model_filepath: P) -> Result<EditableSession<'_>>
	where
		P: AsRef<Path>
	{
		let mut session_ptr: *mut ort_sys::OrtSession = ptr::null_mut();
		let model_path = crate::util::path_to_os_char(model_filepath);

		ortsys![@editor:
			unsafe CreateModelEditorSession(
				self.environment.ptr(),
				model_path.as_ptr(),
				self.session_options_ptr.as_ptr(),
				&mut session_ptr
			)?;
			nonNull(session_ptr)
		];

		EditableSession::new(session_ptr, self)
	}

	#[cfg(feature = "api-22")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn edit_from_memory(&mut self, model_bytes: &[u8]) -> Result<EditableSession<'_>> {
		let mut session_ptr: *mut ort_sys::OrtSession = ptr::null_mut();

		ortsys![@editor:
			unsafe CreateModelEditorSessionFromArray(
				self.environment.ptr(),
				model_bytes.as_ptr().cast(),
				model_bytes.len() as _,
				self.session_options_ptr.as_ptr(),
				&mut session_ptr
			)?;
			nonNull(session_ptr)
		];

		EditableSession::new(session_ptr, self)
	}
}
