use alloc::{boxed::Box, sync::Arc, vec::Vec};
#[cfg(feature = "fetch-models")]
use core::fmt::Write;
use core::{
	any::Any,
	ffi::c_void,
	marker::PhantomData,
	mem::replace,
	ptr::{self, NonNull}
};
#[cfg(feature = "std")]
use std::path::Path;
#[cfg(feature = "fetch-models")]
use std::path::PathBuf;

use smallvec::SmallVec;

use super::{EditableSession, SessionBuilder};
#[cfg(any(target_arch = "wasm32", feature = "std"))]
use crate::error::{Error, ErrorCode};
#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
use crate::util::OsCharArray;
use crate::{
	AsPointer,
	ep::apply_execution_providers,
	error::Result,
	memory::Allocator,
	ortsys,
	session::{InMemorySession, Outlet, Session, SharedSessionInner, io}
};

impl SessionBuilder {
	/// Downloads a pre-trained ONNX model from the given URL and builds the session.
	#[cfg(all(feature = "fetch-models", feature = "std", not(target_arch = "wasm32")))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "fetch-models", feature = "std"))))]
	pub fn commit_from_url(self, model_url: impl AsRef<str>) -> Result<Session> {
		let downloaded_path = SessionBuilder::download(model_url.as_ref())?;
		self.commit_from_file(downloaded_path)
	}

	#[cfg(all(feature = "fetch-models", feature = "std", not(target_arch = "wasm32")))]
	fn download(url: &str) -> Result<PathBuf> {
		use ureq::{
			config::Config,
			tls::{RootCerts, TlsConfig, TlsProvider}
		};

		let mut download_dir = ort_sys::internal::dirs::cache_dir()
			.expect("could not determine cache directory")
			.join("models");
		if std::fs::create_dir_all(&download_dir).is_err() {
			download_dir = std::env::current_dir().expect("Failed to obtain current working directory");
		}

		let model_filename = <sha2::Sha256 as sha2::Digest>::digest(url).into_iter().fold(String::new(), |mut s, b| {
			let _ = write!(&mut s, "{:02x}", b);
			s
		});
		let model_filepath = download_dir.join(&model_filename);
		if model_filepath.exists() {
			crate::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), "Model already exists, skipping download");
			Ok(model_filepath)
		} else {
			crate::info!(model_filepath = format!("{}", model_filepath.display()).as_str(), url = format!("{url:?}").as_str(), "Downloading model");

			let agent = Config::builder()
				.tls_config(
					TlsConfig::builder()
						.root_certs(RootCerts::WebPki)
						.provider(if cfg!(any(feature = "tls-rustls", feature = "tls-rustls-no-provider")) {
							TlsProvider::Rustls
						} else if cfg!(any(feature = "tls-native", feature = "tls-native-vendored")) {
							TlsProvider::NativeTls
						} else {
							return Err(Error::new(
								"No TLS provider configured. When using `fetch-models` with HTTPS URLs, a `tls-*` feature must be enabled."
							));
						})
						.build()
				)
				.build()
				.new_agent();

			let resp = agent.get(url).call().map_err(|e| Error::new(format!("Error downloading to file: {e}")))?;

			let len = resp
				.headers()
				.get("Content-Length")
				.and_then(|h| h.to_str().ok())
				.and_then(|s| s.parse::<usize>().ok())
				.expect("Missing Content-Length header");
			crate::info!(len, "Downloading {} bytes", len);

			let mut reader = resp.into_body().into_with_config().limit(u64::MAX).reader();
			let temp_filepath = download_dir.join(format!("tmp_{}.{model_filename}", ort_sys::internal::random_identifier()));

			let f = std::fs::File::create(&temp_filepath).expect("Failed to create model file");
			let mut writer = std::io::BufWriter::new(f);

			let bytes_io_count = std::io::copy(&mut reader, &mut writer).map_err(Error::wrap)?;
			if bytes_io_count != len as u64 {
				return Err(Error::new(format!("Failed to download entire model; file only has {bytes_io_count} bytes, expected {len}")));
			}

			drop(writer);

			match std::fs::rename(&temp_filepath, &model_filepath) {
				Ok(()) => Ok(model_filepath),
				Err(e) => {
					if model_filepath.exists() {
						let _ = std::fs::remove_file(temp_filepath);
						Ok(model_filepath)
					} else {
						Err(Error::new(format!("Failed to download model: {e}")))
					}
				}
			}
		}
	}

	/// Loads an ONNX model from a file and builds the session.
	#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn commit_from_file<P>(self, model_path: P) -> Result<Session>
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

	#[cfg(all(feature = "std", not(target_arch = "wasm32")))]
	fn commit_from_file_inner(mut self, model_path: &<OsCharArray as core::ops::Deref>::Target) -> Result<Session> {
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

	/// Load an ONNX graph from memory and commit the session
	/// For `.ort` models, we enable `session.use_ort_model_bytes_directly`.
	/// For more information, check [Load ORT format model from an in-memory byte array](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html#load-ort-format-model-from-an-in-memory-byte-array).
	///
	/// If you wish to store the model bytes and the [`InMemorySession`] in the same struct, look for crates that
	/// facilitate creating self-referential structs, such as [`ouroboros`](https://github.com/joshua-maros/ouroboros).
	#[cfg(not(target_arch = "wasm32"))]
	pub fn commit_from_memory_directly(mut self, model_bytes: &[u8]) -> Result<InMemorySession<'_>> {
		// Enable zero-copy deserialization for models in `.ort` format.
		self.add_config_entry("session.use_ort_model_bytes_directly", "1")?;
		self.add_config_entry("session.use_ort_model_bytes_for_initializers", "1")?;

		let session = self.commit_from_memory(model_bytes)?;
		Ok(InMemorySession { session, phantom: PhantomData })
	}

	/// Load an ONNX graph from memory and commit the session.
	#[cfg(not(target_arch = "wasm32"))]
	pub fn commit_from_memory(mut self, model_bytes: &[u8]) -> Result<Session> {
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
	#[cfg(target_arch = "wasm32")]
	pub async fn commit_from_url(mut self, model_url: impl AsRef<str>) -> Result<Session> {
		self.pre_commit()?;

		let mut session_ptr = ptr::null_mut();
		let status = ortsys![unsafe CreateSession(self.environment.ptr(), model_url.as_ref(), self.ptr(), &mut session_ptr)].await;
		unsafe { crate::error::status_to_result(status) }?;

		let Some(session_ptr) = NonNull::new(session_ptr) else {
			return Err(Error::new(alloc::format!("Session creation failed with unknown error")));
		};

		self.commit_finalize(session_ptr)
	}

	/// Load an ONNX graph from memory and commit the session.
	#[cfg(target_arch = "wasm32")]
	pub async fn commit_from_memory(mut self, model_bytes: &[u8]) -> Result<Session> {
		self.pre_commit()?;

		let mut session_ptr = ptr::null_mut();
		let status = ortsys![unsafe CreateSessionFromArray(self.environment.ptr(), model_bytes.as_ref(), self.ptr(), &mut session_ptr)].await;
		unsafe { crate::error::status_to_result(status) }?;

		let Some(session_ptr) = NonNull::new(session_ptr) else {
			return Err(Error::new(alloc::format!("Session creation failed with unknown error")));
		};

		self.commit_finalize(session_ptr)
	}

	pub(crate) fn pre_commit(&mut self) -> Result<()> {
		if !self.no_env_eps {
			let env = Arc::clone(&self.environment); // dumb borrowck hack
			apply_execution_providers(self, env.execution_providers(), "environment")?;
		}

		if self.environment.has_global_threadpool() && !self.no_global_thread_pool {
			ortsys![unsafe DisablePerSessionThreads(self.ptr_mut())?];
		}

		Ok(())
	}

	pub(crate) fn commit_finalize(&mut self, ptr: NonNull<ort_sys::OrtSession>) -> Result<Session> {
		let allocator = match &self.memory_info {
			Some(info) => {
				let mut allocator_ptr: *mut ort_sys::OrtAllocator = ptr::null_mut();
				ortsys![unsafe CreateAllocator(ptr.as_ptr(), info.ptr(), &mut allocator_ptr)?; nonNull(allocator_ptr)];
				unsafe { Allocator::from_raw(allocator_ptr) }
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

		let mut extras: SmallVec<[Box<dyn Any>; 4]> = self.operator_domains.drain(..).map(|d| Box::new(d) as Box<dyn Any>).collect();
		if let Some(prepacked_weights) = self.prepacked_weights.take() {
			extras.push(Box::new(prepacked_weights) as Box<dyn Any>);
		}
		if let Some(thread_manager) = self.thread_manager.take() {
			extras.push(Box::new(thread_manager) as Box<dyn Any>);
		}
		if let Some(logger) = self.logger.take() {
			extras.push(Box::new(logger) as Box<dyn Any>); // Box<Arc<Box<dyn ...>>>!
		}

		crate::logging::create!(Session, ptr);

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr: ptr,
				allocator,
				_initializers: replace(&mut self.initializers, SmallVec::new()),
				_extras: extras,
				_environment: self.environment.clone()
			}),
			inputs,
			outputs
		})
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn edit_from_file<P>(self, model_filepath: P) -> Result<EditableSession>
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

	pub fn edit_from_memory(self, model_bytes: &[u8]) -> Result<EditableSession> {
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
