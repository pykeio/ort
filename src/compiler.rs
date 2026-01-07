//! Provides [`ModelCompiler`], which optimizes and creates EP graphs for an ONNX model ahead-of-time to greatly reduce
//! startup time.
//!
//! Many [execution providers](crate::ep) use a third-party graph-based neural network API (like
//! CoreML, DirectML, CUDA graphs, TensorRT, etc.). The EPs must convert the ONNX graph to these device-native graphs,
//! which can take a lot of work - especially when you consider that both ONNX Runtime and the EP backend are doing
//! their own optimizations on their respective graphs! The end result is that [sessions](crate::session) can often take
//! a very long time to initialize - often multiple seconds and, in the case of more complex models, upwards of minutes.
//! This is unacceptable for many applications, hence the need for [`ModelCompiler`].
//!
//! [`ModelCompiler`] is created like a normal [`Session`](crate::session::Session), except instead of ever performing
//! inference on the model, [`ModelCompiler`] only:
//! - optimizes the ONNX model;
//! - creates the EP graph(s);
//! - saves EP states in a new model.
//!
//! Loading a compiled model like one would with any other `.onnx` model (using the same EPs & options used to compile
//! it) will greatly reduce the time required for the EP graph step. The model must be compiled on the end user's
//! system; you can't distribute pre-compiled models.
//!
//! ```no_run
//! # use std::path::PathBuf;
//! # use ort::{compiler::ModelCompiler, session::Session, ep};
//! # fn main() -> ort::Result<()> {
//! let session_options = Session::builder()?.with_execution_providers([ep::CoreML::default()
//! 	.with_model_format(ep::coreml::ModelFormat::MLProgram)
//! 	.build()])?;
//!
//! let compiled_path = PathBuf::from("model-coreml.compiled.onnx");
//! if !compiled_path.exists() {
//! 	ModelCompiler::new(session_options.clone())?
//! 		.with_model_from_file("model.onnx")?
//! 		.compile_to_file(&compiled_path)?;
//! }
//!
//! let session = session_options.commit_from_file(&compiled_path)?;
//! # Ok(())
//! # }
//! ```

use core::{
	ffi::c_void,
	marker::PhantomData,
	mem,
	ops::Deref,
	ptr::{self, NonNull}
};
#[cfg(feature = "std")]
use std::path::Path;

use crate::{
	AsPointer,
	error::{Error, Result},
	memory::Allocator,
	ortsys,
	session::builder::SessionBuilder,
	util::OnceLock
};

/// Returns a pointer to the global [`ort_sys::OrtCompileApi`] object, or errors if the Compile API is not
/// supported by this backend.
pub fn compile_api() -> Result<&'static ort_sys::OrtCompileApi> {
	struct CompileApiPointer(*const ort_sys::OrtCompileApi);
	unsafe impl Send for CompileApiPointer {}
	unsafe impl Sync for CompileApiPointer {}

	static COMPILE_API: OnceLock<CompileApiPointer> = OnceLock::new();

	let ptr = NonNull::new(
		COMPILE_API
			.get_or_init(|| {
				let api = ortsys![unsafe GetCompileApi()];
				CompileApiPointer(api)
			})
			.0
			.cast_mut()
	)
	.ok_or_else(|| Error::new("The Compile API is not supported with this backend."))?;
	Ok(unsafe { ptr.as_ref() })
}

pub struct ModelCompiler<'i> {
	ptr: NonNull<ort_sys::OrtModelCompilationOptions>,
	session_options: SessionBuilder,
	_p: PhantomData<&'i ()>
}

impl<'i> ModelCompiler<'i> {
	pub fn new(options: SessionBuilder) -> Result<Self> {
		let mut ptr = ptr::null_mut();
		ortsys![@compile:
			unsafe CreateModelCompilationOptionsFromSessionOptions(
				options.environment.ptr(),
				options.ptr(),
				&mut ptr
			)?;
			nonNull(ptr)
		];
		crate::logging::create!(ModelCompiler, ptr);
		Ok(Self {
			ptr,
			session_options: options,
			_p: PhantomData
		})
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_model_from_file<P: AsRef<Path>>(self, path: P) -> Result<Self> {
		let model_path = crate::util::path_to_os_char(path);
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetInputModelPath(
				self.ptr.as_ptr(),
				model_path.as_ptr()
			)?
		];
		Ok(self)
	}

	pub fn with_model_from_memory<'i2>(self, data: &'i2 [u8]) -> Result<ModelCompiler<'i2>> {
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetInputModelFromBuffer(
				self.ptr.as_ptr(),
				data.as_ptr().cast(),
				data.len()
			)?
		];
		Ok(unsafe { mem::transmute::<ModelCompiler<'i>, ModelCompiler<'i2>>(self) })
	}

	/// Embed the execution provider context in the model.
	///
	/// This context typically includes binary data to be used by the execution provider, like weights. The default
	/// behavior (when this option is not enabled) is for the execution provider to place context data in a
	/// temporary path, and store that path in the compiled model. Enabling this option will instead embed that data
	/// directly in the compiled model, meaning it won't rely on other files on the system.
	pub fn with_embed_ep_context(self) -> Result<Self> {
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetEpContextEmbedMode(
				self.ptr.as_ptr(),
				true
			)?
		];
		Ok(self)
	}

	/// Store uncompiled initializers over a given `threshold` in a separate file at `path`.
	///
	/// Initializers present in the original model that 1) are not used in the compiled graph, and 2) are larger than
	/// `threshold` bytes, will be stored in the file.
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn with_external_initializers<P: AsRef<Path>>(self, threshold: usize, path: P) -> Result<Self> {
		let model_path = crate::util::path_to_os_char(path);
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetOutputModelExternalInitializersFile(
				self.ptr.as_ptr(),
				model_path.as_ptr(),
				threshold
			)?
		];
		Ok(self)
	}

	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn compile_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
		let model_path = crate::util::path_to_os_char(path);
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetOutputModelPath(
				self.ptr.as_ptr(),
				model_path.as_ptr()
			)?
		];
		ortsys![@compile: unsafe CompileModel(self.session_options.environment.ptr(), self.ptr.as_ptr())?];
		Ok(())
	}

	pub fn compile_to_buffer(&self) -> Result<CompiledModel> {
		let mut allocator = Allocator::default();
		let mut ptr = ptr::null_mut();
		let mut size = 0;
		ortsys![@compile:
			unsafe ModelCompilationOptions_SetOutputModelBuffer(
				self.ptr.as_ptr(),
				allocator.ptr_mut(),
				&mut ptr,
				&mut size
			)?
		];
		ortsys![@compile: unsafe CompileModel(self.session_options.environment.ptr(), self.ptr.as_ptr())?];
		crate::logging::create!(CompiledModel, ptr);
		Ok(CompiledModel { ptr, size, allocator })
	}
}

impl AsPointer for ModelCompiler<'_> {
	type Sys = ort_sys::OrtModelCompilationOptions;
	fn ptr(&self) -> *const Self::Sys {
		self.ptr.as_ptr()
	}
}

impl Drop for ModelCompiler<'_> {
	fn drop(&mut self) {
		ortsys![@compile: unsafe ReleaseModelCompilationOptions(self.ptr.as_ptr())];
		crate::logging::drop!(ModelCompiler, self.ptr);
	}
}

pub struct CompiledModel {
	ptr: *mut c_void,
	size: usize,
	allocator: Allocator
}

impl CompiledModel {
	pub fn as_slice(&self) -> &[u8] {
		unsafe { core::slice::from_raw_parts(self.ptr.cast(), self.size) }
	}
}

impl Deref for CompiledModel {
	type Target = [u8];
	fn deref(&self) -> &Self::Target {
		self.as_slice()
	}
}

impl Drop for CompiledModel {
	fn drop(&mut self) {
		unsafe { self.allocator.free(self.ptr) };
		crate::logging::drop!(CompiledModel, self.ptr);
	}
}
