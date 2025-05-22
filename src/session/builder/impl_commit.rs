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
#[cfg(feature = "std")]
use crate::error::{Error, ErrorCode};
use crate::{
	AsPointer,
	environment::get_environment,
	error::Result,
	execution_providers::apply_execution_providers,
	memory::Allocator,
	ortsys,
	session::{InMemorySession, Input, Output, Session, SharedSessionInner, dangerous}
};

impl SessionBuilder {
	/// Downloads a pre-trained ONNX model from the given URL and builds the session.
	#[cfg(all(feature = "fetch-models", feature = "std"))]
	#[cfg_attr(docsrs, doc(cfg(all(feature = "fetch-models", feature = "std"))))]
	pub fn commit_from_url(self, model_url: impl AsRef<str>) -> Result<Session> {
		let downloaded_path = SessionBuilder::download(model_url.as_ref())?;
		self.commit_from_file(downloaded_path)
	}

	#[cfg(all(feature = "fetch-models", feature = "std"))]
	fn download(url: &str) -> Result<PathBuf> {
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

			let resp = ureq::get(url).call().map_err(|e| Error::new(format!("Error downloading to file: {e}")))?;

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
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn commit_from_file<P>(self, model_filepath: P) -> Result<Session>
	where
		P: AsRef<Path>
	{
		self.commit_from_file_inner(model_filepath.as_ref())
	}

	#[cfg(feature = "std")]
	fn commit_from_file_inner(mut self, model_filepath: &Path) -> Result<Session> {
		if !model_filepath.exists() {
			return Err(Error::new_with_code(ErrorCode::NoSuchFile, format!("File at `{}` does not exist", model_filepath.display())));
		}

		let model_path = crate::util::path_to_os_char(model_filepath);

		let env = get_environment()?;
		if !self.no_env_eps {
			apply_execution_providers(&mut self, &env.execution_providers, "environment")?;
		}

		if env.has_global_threadpool && !self.no_global_thread_pool {
			ortsys![unsafe DisablePerSessionThreads(self.ptr_mut())?];
		}

		let session_ptr = if let Some(prepacked_weights) = self.prepacked_weights.as_ref() {
			let mut session_ptr = ptr::null_mut();
			ortsys![unsafe CreateSessionWithPrepackedWeightsContainer(env.ptr(), model_path.as_ptr(), self.ptr(), prepacked_weights.ptr().cast_mut(), &mut session_ptr)?; nonNull(session_ptr)];
			session_ptr
		} else {
			let mut session_ptr = ptr::null_mut();
			ortsys![unsafe CreateSession(env.ptr(), model_path.as_ptr(), self.ptr(), &mut session_ptr)?; nonNull(session_ptr)];
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
	pub fn commit_from_memory_directly(mut self, model_bytes: &[u8]) -> Result<InMemorySession<'_>> {
		// Enable zero-copy deserialization for models in `.ort` format.
		self.add_config_entry("session.use_ort_model_bytes_directly", "1")?;
		self.add_config_entry("session.use_ort_model_bytes_for_initializers", "1")?;

		let session = self.commit_from_memory(model_bytes)?;
		Ok(InMemorySession { session, phantom: PhantomData })
	}

	/// Load an ONNX graph from memory and commit the session.
	pub fn commit_from_memory(mut self, model_bytes: &[u8]) -> Result<Session> {
		let env = get_environment()?;
		if !self.no_env_eps {
			apply_execution_providers(&mut self, &env.execution_providers, "environment")?;
		}

		if env.has_global_threadpool && !self.no_global_thread_pool {
			ortsys![unsafe DisablePerSessionThreads(self.ptr_mut())?];
		}

		let model_data = model_bytes.as_ptr().cast::<c_void>();
		let model_data_length = model_bytes.len();
		let session_ptr = if let Some(prepacked_weights) = self.prepacked_weights.as_ref() {
			let mut session_ptr = ptr::null_mut();
			ortsys![
				unsafe CreateSessionFromArrayWithPrepackedWeightsContainer(env.ptr(), model_data, model_data_length, self.ptr(), prepacked_weights.ptr().cast_mut(), &mut session_ptr)?;
				nonNull(session_ptr)
			];
			session_ptr
		} else {
			let mut session_ptr = ptr::null_mut();
			ortsys![
				unsafe CreateSessionFromArray(env.ptr(), model_data, model_data_length, self.ptr(), &mut session_ptr)?;
				nonNull(session_ptr)
			];
			session_ptr
		};

		self.commit_finalize(session_ptr)
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
		let num_input_nodes = dangerous::extract_inputs_count(ptr)?;
		let num_output_nodes = dangerous::extract_outputs_count(ptr)?;
		let inputs = (0..num_input_nodes)
			.map(|i| dangerous::extract_input(ptr, &allocator, i))
			.collect::<Result<Vec<Input>>>()?;
		let outputs = (0..num_output_nodes)
			.map(|i| dangerous::extract_output(ptr, &allocator, i))
			.collect::<Result<Vec<Output>>>()?;

		let mut extras: SmallVec<Box<dyn Any>, 4> = self.operator_domains.drain(..).map(|d| Box::new(d) as Box<dyn Any>).collect();
		if let Some(prepacked_weights) = self.prepacked_weights.take() {
			extras.push(Box::new(prepacked_weights) as Box<dyn Any>);
		}
		if let Some(thread_manager) = self.thread_manager.take() {
			extras.push(Box::new(thread_manager) as Box<dyn Any>);
		}
		if let Some(logger) = self.logger.take() {
			extras.push(Box::new(logger) as Box<dyn Any>); // Box<Arc<Box<dyn ...>>>!
		}

		Ok(Session {
			inner: Arc::new(SharedSessionInner {
				session_ptr: ptr,
				allocator,
				_initializers: replace(&mut self.initializers, SmallVec::new()),
				_extras: extras
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

		let env = get_environment()?;

		ortsys![@editor:
			unsafe CreateModelEditorSession(
				env.ptr(),
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

		let env = get_environment()?;

		ortsys![@editor:
			unsafe CreateModelEditorSessionFromArray(
				env.ptr(),
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
