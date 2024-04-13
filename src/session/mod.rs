//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

use std::{any::Any, ffi::CString, marker::PhantomData, ops::Deref, os::raw::c_char, ptr::NonNull, sync::Arc};

use super::{
	char_p_to_string,
	environment::Environment,
	error::{assert_non_null_pointer, assert_null_pointer, status_to_result, Error, ErrorInternal, Result},
	extern_system_fn,
	io_binding::IoBinding,
	memory::Allocator,
	metadata::ModelMetadata,
	ortsys,
	value::{Value, ValueType}
};

mod r#async;
pub(crate) mod builder;
pub(crate) mod input;
pub(crate) mod output;
use self::r#async::{AsyncInferenceContext, InferenceFutInner};
pub use self::{
	r#async::InferenceFut,
	builder::{GraphOptimizationLevel, SessionBuilder},
	input::{SessionInputValue, SessionInputs},
	output::SessionOutputs
};

/// Holds onto an [`ort_sys::OrtSession`] pointer and its associated allocator.
///
/// Internally, this is wrapped in an [`Arc`] and shared between a [`Session`] and any [`Value`]s created as a result
/// of [`Session::run`] to ensure that the [`Value`]s are kept alive until all references to the session are dropped.
#[derive(Debug)]
pub struct SharedSessionInner {
	pub(crate) session_ptr: NonNull<ort_sys::OrtSession>,
	allocator: Allocator,
	/// Additional things we may need to hold onto for the duration of this session, like [`crate::OperatorDomain`]s and
	/// DLL handles for operator libraries.
	_extras: Vec<Box<dyn Any>>,
	_environment: Arc<Environment>
}

impl SharedSessionInner {
	/// Returns the underlying [`ort_sys::OrtSession`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtSession {
		self.session_ptr.as_ptr()
	}
}

unsafe impl Send for SharedSessionInner {}
unsafe impl Sync for SharedSessionInner {}

impl Drop for SharedSessionInner {
	#[tracing::instrument]
	fn drop(&mut self) {
		tracing::debug!("dropping SharedSessionInner");
		ortsys![unsafe ReleaseSession(self.session_ptr.as_ptr())];
	}
}

/// An ONNX Runtime graph to be used for inference.
///
/// ```
/// # use ort::{GraphOptimizationLevel, Session};
/// # fn main() -> ort::Result<()> {
/// let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![input]?)?;
/// # 	Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Session {
	pub(crate) inner: Arc<SharedSessionInner>,
	/// Information about the graph's inputs.
	pub inputs: Vec<Input>,
	/// Information about the graph's outputs.
	pub outputs: Vec<Output>
}

/// A [`Session`] where the graph data is stored in memory.
///
/// This type is automatically `Deref`'d into a `Session`, so you can use it like you would a regular `Session`. See
/// [`Session`] for usage details.
pub struct InMemorySession<'s> {
	session: Session,
	phantom: PhantomData<&'s ()>
}

impl<'s> Deref for InMemorySession<'s> {
	type Target = Session;
	fn deref(&self) -> &Self::Target {
		&self.session
	}
}

/// Information about a [`Session`] input.
#[derive(Debug)]
pub struct Input {
	/// Name of the input.
	pub name: String,
	/// Type of the input's elements.
	pub input_type: ValueType
}

/// Information about a [`Session`] output.
#[derive(Debug)]
pub struct Output {
	/// Name of the output.
	pub name: String,
	/// Type of the output's elements.
	pub output_type: ValueType
}

/// A structure which can be passed to [`Session::run_with_options`] to allow terminating/unterminating a session
/// inference run from a different thread.
#[derive(Debug)]
pub struct RunOptions {
	pub(crate) run_options_ptr: NonNull<ort_sys::OrtRunOptions>
}

// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ac2a08cac0a657604bd5899e0d1a13675
unsafe impl Send for RunOptions {}
unsafe impl Sync for RunOptions {}

impl RunOptions {
	/// Creates a new [`RunOptions`] struct.
	pub fn new() -> Result<Self> {
		let mut run_options_ptr: *mut ort_sys::OrtRunOptions = std::ptr::null_mut();
		ortsys![unsafe CreateRunOptions(&mut run_options_ptr) -> Error::CreateRunOptions; nonNull(run_options_ptr)];
		Ok(Self {
			run_options_ptr: unsafe { NonNull::new_unchecked(run_options_ptr) }
		})
	}

	/// Sets a tag to identify this run in logs.
	pub fn set_tag(&mut self, tag: impl AsRef<str>) -> Result<()> {
		let tag = CString::new(tag.as_ref())?;
		ortsys![unsafe RunOptionsSetRunTag(self.run_options_ptr.as_ptr(), tag.as_ptr()) -> Error::RunOptionsSetTag];
		Ok(())
	}

	/// Sets the termination flag for the runs associated with this [`RunOptions`].
	///
	/// This function returns immediately (it does not wait for the session run to terminate). The run will terminate as
	/// soon as it is able to.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn terminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsSetTerminate(self.run_options_ptr.as_ptr()) -> Error::RunOptionsSetTerminate];
		Ok(())
	}

	/// Resets the termination flag for the runs associated with [`RunOptions`].
	///
	/// ```no_run
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// 	// ...oops, didn't mean to do that
	/// 	let _ = run_options_.unterminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// assert!(res.is_ok());
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn unterminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsUnsetTerminate(self.run_options_ptr.as_ptr()) -> Error::RunOptionsUnsetTerminate];
		Ok(())
	}
}

impl Drop for RunOptions {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseRunOptions(self.run_options_ptr.as_ptr())];
	}
}

impl Session {
	/// Creates a new [`SessionBuilder`].
	pub fn builder() -> Result<SessionBuilder> {
		SessionBuilder::new()
	}

	/// Returns this session's [`Allocator`].
	#[must_use]
	pub fn allocator(&self) -> &Allocator {
		&self.inner.allocator
	}

	/// Creates a new [`IoBinding`] for this session.
	pub fn create_binding(&self) -> Result<IoBinding> {
		IoBinding::new(self)
	}

	/// Returns the underlying [`ort_sys::OrtSession`] pointer.
	pub fn ptr(&self) -> *mut ort_sys::OrtSession {
		self.inner.ptr()
	}

	/// Get a shared ([`Arc`]'d) reference to the underlying [`SharedSessionInner`], which holds the
	/// [`ort_sys::OrtSession`] pointer and the session allocator.
	#[must_use]
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}

	/// Run input data through the ONNX graph, performing inference.
	///
	/// See [`crate::inputs!`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![input]?)?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run<'s, 'i, 'v: 'i, const N: usize>(&'s self, input_values: impl Into<SessionInputs<'i, 'v, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(&input_values.iter().map(|(k, _)| k.as_ref()).collect::<Vec<_>>(), input_values.iter().map(|(_, v)| v), None)
			}
		}
	}

	/// Run input data through the ONNX graph, performing inference, with a [`RunOptions`] struct. The most common usage
	/// of `RunOptions` is to allow the session run to be terminated from a different thread.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![input]?, run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run_with_options<'s, 'i, 'v: 'i, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: Arc<RunOptions>
	) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), Some(run_options))
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), Some(run_options))
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(&input_values.iter().map(|(k, _)| k.as_ref()).collect::<Vec<_>>(), input_values.iter().map(|(_, v)| v), Some(run_options))
			}
		}
	}

	fn run_inner<'i, 'v: 'i>(
		&self,
		input_names: &[&str],
		input_values: impl Iterator<Item = &'i SessionInputValue<'v>>,
		run_options: Option<Arc<RunOptions>>
	) -> Result<SessionOutputs<'_>> {
		let input_names_ptr: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();
		let output_names_ptr: Vec<*const c_char> = self
			.outputs
			.iter()
			.map(|output| CString::new(output.name.as_str()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();

		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		// The C API expects pointers for the arrays (pointers to C-arrays)
		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.map(|input_array_ort| input_array_ort.ptr().cast_const()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options {
			run_options.run_options_ptr.as_ptr()
		} else {
			std::ptr::null_mut()
		};

		ortsys![
			unsafe Run(
				self.inner.session_ptr.as_ptr(),
				run_options_ptr,
				input_names_ptr.as_ptr(),
				input_ort_values.as_ptr(),
				input_ort_values.len() as _,
				output_names_ptr.as_ptr(),
				output_names_ptr.len() as _,
				output_tensor_ptrs.as_mut_ptr()
			) -> Error::SessionRun
		];

		let outputs: Vec<Value> = output_tensor_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), Some(Arc::clone(&self.inner)))
			})
			.collect();

		// Reconvert name ptrs to CString so drop impl is called and memory is freed
		drop(
			input_names_ptr
				.into_iter()
				.chain(output_names_ptr.into_iter())
				.map(|p| {
					assert_non_null_pointer(p, "c_char for CString")?;
					unsafe { Ok(CString::from_raw(p.cast_mut().cast())) }
				})
				.collect::<Result<Vec<_>>>()?
		);

		Ok(SessionOutputs::new(self.outputs.iter().map(|o| o.name.as_str()), outputs))
	}

	/// Asynchronously run input data through the ONNX graph, performing inference.
	///
	/// Inference will be performed on a thread in the session's thread pool. **Thus, the session must have been
	/// configured to have multiple intra-op threads**; see [`SessionBuilder::with_intra_threads`].
	///
	/// See [`crate::inputs!`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{Session, RunOptions, Value, ValueType, TensorElementType};
	/// # fn main() -> ort::Result<()> { tokio_test::block_on(async {
	/// let session = Session::builder()?.with_intra_threads(2)?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run_async(ort::inputs![input]?)?.await?;
	/// # 	Ok(())
	/// # }) }
	/// ```
	pub fn run_async<'s, 'i, 'v: 'i + 's, const N: usize>(&'s self, input_values: impl Into<SessionInputs<'i, 'v, N>> + 'static) -> Result<InferenceFut<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(_) => unimplemented!("slices cannot be used in `run_async`"),
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(&self.inputs.iter().map(|input| input.name.to_string()).collect::<Vec<_>>(), input_values.into_iter())
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner_async(&input_values.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>(), input_values.into_iter().map(|(_, v)| v))
			}
		}
	}

	fn run_inner_async<'s, 'v: 's>(&'s self, input_names: &[String], input_values: impl Iterator<Item = SessionInputValue<'v>>) -> Result<InferenceFut<'s>> {
		// create a `RunOptions` to pass to the future so that when it drops, it terminates inference - crucial
		// (performance-wise) for routines involving `tokio::select!` or timeouts
		let run_options = Arc::new(RunOptions::new()?);

		let input_name_ptrs: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();
		let output_name_ptrs: Vec<*const c_char> = self
			.outputs
			.iter()
			.map(|output| CString::new(output.name.as_str()).unwrap())
			.map(|n| n.into_raw().cast_const())
			.collect();

		let output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = vec![std::ptr::null_mut(); self.outputs.len()];

		let input_values: Vec<_> = input_values.collect();
		let input_ort_values: Vec<*const ort_sys::OrtValue> = input_values.iter().map(|input_array_ort| input_array_ort.ptr().cast_const()).collect();

		let run_options_ptr = run_options.run_options_ptr.as_ptr();

		let async_inner = Arc::new(InferenceFutInner::new());

		let ctx = Box::leak(Box::new(AsyncInferenceContext {
			inner: Arc::clone(&async_inner),
			_input_values: input_values,
			// everything allocated within `run_inner_async` needs to be kept alive until we are certain inference has completed and ONNX Runtime no longer
			// needs the data - i.e. when `async_callback` is called. `async_callback` will free all of this data just like we do in `run_inner`
			input_ort_values,
			input_name_ptrs,
			output_name_ptrs,
			output_names: self.outputs.iter().map(|o| o.name.as_str()).collect::<Vec<_>>(),
			output_value_ptrs: output_tensor_ptrs,
			session_inner: &self.inner
		}));

		ortsys![
			unsafe RunAsync(
				self.inner.session_ptr.as_ptr(),
				run_options_ptr,
				ctx.input_name_ptrs.as_ptr(),
				ctx.input_ort_values.as_ptr(),
				ctx.input_ort_values.len() as _,
				ctx.output_name_ptrs.as_ptr(),
				ctx.output_name_ptrs.len() as _,
				ctx.output_value_ptrs.as_mut_ptr(),
				Some(self::r#async::async_callback),
				ctx as *mut _ as *mut ort_sys::c_void
			) -> Error::SessionRun
		];

		Ok(InferenceFut::new(async_inner, run_options))
	}

	/// Gets the session model metadata. See [`ModelMetadata`] for more info.
	pub fn metadata(&self) -> Result<ModelMetadata<'_>> {
		let mut metadata_ptr: *mut ort_sys::OrtModelMetadata = std::ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.inner.session_ptr.as_ptr(), &mut metadata_ptr) -> Error::GetModelMetadata; nonNull(metadata_ptr)];
		Ok(ModelMetadata::new(unsafe { NonNull::new_unchecked(metadata_ptr) }, &self.inner.allocator))
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiling file will be empty.
	pub fn end_profiling(&self) -> Result<String> {
		let mut profiling_name: *mut c_char = std::ptr::null_mut();

		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr.as_ptr(), self.inner.allocator.ptr.as_ptr(), &mut profiling_name)];
		assert_non_null_pointer(profiling_name, "ProfilingName")?;
		dangerous::raw_pointer_to_string(&self.inner.allocator, profiling_name)
	}
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
// Allowing `Sync` segfaults with CUDA, DirectML, and seemingly any EP other than the CPU EP. I'm not certain if it's a
// temporary bug in ONNX Runtime or a wontfix. Maybe this impl should be removed just to be safe?
unsafe impl Sync for Session {}

mod dangerous {
	use super::*;
	use crate::value::{extract_data_type_from_map_info, extract_data_type_from_sequence_info, extract_data_type_from_tensor_info};

	pub(super) fn extract_inputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![unsafe SessionGetInputCount];
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![unsafe SessionGetOutputCount];
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: extern_system_fn! { unsafe fn(*const ort_sys::OrtSession, *mut ort_sys::size_t) -> *mut ort_sys::OrtStatus },
		session_ptr: NonNull<ort_sys::OrtSession>
	) -> Result<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr.as_ptr(), &mut num_nodes) };
		status_to_result(status).map_err(Error::GetInOutCount)?;
		assert_null_pointer(status, "SessionStatus")?;
		(num_nodes != 0)
			.then_some(())
			.ok_or_else(|| Error::GetInOutCount(ErrorInternal::Msg("No nodes in model".to_owned())))?;
		Ok(num_nodes as _)
	}

	fn extract_input_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetInputName];
		extract_io_name(f, session_ptr, allocator, i)
	}

	fn extract_output_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: ort_sys::size_t) -> Result<String> {
		let f = ortsys![unsafe SessionGetOutputName];
		extract_io_name(f, session_ptr, allocator, i)
	}

	pub(crate) fn raw_pointer_to_string(allocator: &Allocator, c_str: *mut c_char) -> Result<String> {
		let name = match char_p_to_string(c_str) {
			Ok(name) => name,
			Err(e) => {
				unsafe { allocator.free(c_str) };
				return Err(e);
			}
		};
		unsafe { allocator.free(c_str) };
		Ok(name)
	}

	fn extract_io_name(
		f: extern_system_fn! { unsafe fn(
			*const ort_sys::OrtSession,
			ort_sys::size_t,
			*mut ort_sys::OrtAllocator,
			*mut *mut c_char,
		) -> *mut ort_sys::OrtStatus },
		session_ptr: NonNull<ort_sys::OrtSession>,
		allocator: &Allocator,
		i: ort_sys::size_t
	) -> Result<String> {
		let mut name_bytes: *mut c_char = std::ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, allocator.ptr.as_ptr(), &mut name_bytes) };
		status_to_result(status).map_err(Error::GetInputName)?;
		assert_non_null_pointer(name_bytes, "InputName")?;

		raw_pointer_to_string(allocator, name_bytes)
	}

	pub(super) fn extract_input(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Input> {
		let input_name = extract_input_name(session_ptr, allocator, i as _)?;
		let f = ortsys![unsafe SessionGetInputTypeInfo];
		let input_type = extract_io(f, session_ptr, i as _)?;
		Ok(Input { name: input_name, input_type })
	}

	pub(super) fn extract_output(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Output> {
		let output_name = extract_output_name(session_ptr, allocator, i as _)?;
		let f = ortsys![unsafe SessionGetOutputTypeInfo];
		let output_type = extract_io(f, session_ptr, i as _)?;
		Ok(Output { name: output_name, output_type })
	}

	fn extract_io(
		f: extern_system_fn! { unsafe fn(
			*const ort_sys::OrtSession,
			ort_sys::size_t,
			*mut *mut ort_sys::OrtTypeInfo,
		) -> *mut ort_sys::OrtStatus },
		session_ptr: NonNull<ort_sys::OrtSession>,
		i: ort_sys::size_t
	) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = std::ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, &mut typeinfo_ptr) };
		status_to_result(status).map_err(Error::GetTypeInfo)?;
		assert_non_null_pointer(typeinfo_ptr, "TypeInfo")?;

		let mut ty: ort_sys::ONNXType = ort_sys::ONNXType::ONNX_TYPE_UNKNOWN;
		let status = ortsys![unsafe GetOnnxTypeFromTypeInfo(typeinfo_ptr, &mut ty)];
		status_to_result(status).map_err(Error::GetOnnxTypeFromTypeInfo)?;
		let io_type = match ty {
			ort_sys::ONNXType::ONNX_TYPE_TENSOR | ort_sys::ONNXType::ONNX_TYPE_SPARSETENSOR => {
				let mut info_ptr: *const ort_sys::OrtTensorTypeAndShapeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToTensorInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToTensorInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_tensor_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_SEQUENCE => {
				let mut info_ptr: *const ort_sys::OrtSequenceTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToSequenceTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToSequenceTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_sequence_info(info_ptr)? }
			}
			ort_sys::ONNXType::ONNX_TYPE_MAP => {
				let mut info_ptr: *const ort_sys::OrtMapTypeInfo = std::ptr::null_mut();
				ortsys![unsafe CastTypeInfoToMapTypeInfo(typeinfo_ptr, &mut info_ptr) -> Error::CastTypeInfoToMapTypeInfo; nonNull(info_ptr)];
				unsafe { extract_data_type_from_map_info(info_ptr)? }
			}
			_ => unreachable!()
		};

		ortsys![unsafe ReleaseTypeInfo(typeinfo_ptr)];
		Ok(io_type)
	}
}
