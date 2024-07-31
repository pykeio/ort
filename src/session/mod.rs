//! Contains the [`Session`] and [`SessionBuilder`] types for managing ONNX Runtime sessions and performing inference.

use std::{any::Any, ffi::CString, marker::PhantomData, ops::Deref, os::raw::c_char, ptr::NonNull, sync::Arc};

use crate::{
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
mod run_options;
use self::r#async::{AsyncInferenceContext, InferenceFutInner, RunOptionsRef};
pub use self::{
	r#async::InferenceFut,
	builder::{GraphOptimizationLevel, SessionBuilder},
	input::{SessionInputValue, SessionInputs},
	output::SessionOutputs,
	run_options::{HasSelectedOutputs, NoSelectedOutputs, OutputSelector, RunOptions, SelectedOutputMarker}
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
	pub fn run<'s, 'i, 'v: 'i, const N: usize>(&'s self, input_values: impl Into<SessionInputs<'i, 'v, N>>) -> Result<SessionOutputs<'_, 's>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner::<NoSelectedOutputs>(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner::<NoSelectedOutputs>(&self.inputs.iter().map(|input| input.name.as_str()).collect::<Vec<_>>(), input_values.iter(), None)
			}
			SessionInputs::ValueMap(input_values) => self.run_inner::<NoSelectedOutputs>(
				&input_values.iter().map(|(k, _)| k.as_ref()).collect::<Vec<_>>(),
				input_values.iter().map(|(_, v)| v),
				None
			)
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
	/// let res = session.run_with_options(ort::inputs![input]?, &*run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run_with_options<'r, 's: 'r, 'i, 'v: 'i, O: SelectedOutputMarker, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<SessionOutputs<'r, 's>> {
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

	fn run_inner<'i, 'r, 's: 'r, 'v: 'i, O: SelectedOutputMarker>(
		&'s self,
		input_names: &[&str],
		input_values: impl Iterator<Item = &'i SessionInputValue<'v>>,
		run_options: Option<&'r RunOptions<O>>
	) -> Result<SessionOutputs<'r, 's>> {
		let input_names_ptr: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap_or_else(|_| unreachable!()))
			.map(|n| n.into_raw().cast_const())
			.collect();

		let (output_names, output_tensors) = match run_options {
			Some(r) => r.outputs.resolve_outputs(&self.outputs),
			None => (self.outputs.iter().map(|o| o.name.as_str()).collect(), std::iter::repeat_with(|| None).take(self.outputs.len()).collect())
		};
		let output_names_ptr: Vec<*const c_char> = output_names
			.iter()
			.map(|n| CString::new(*n).unwrap_or_else(|_| unreachable!()))
			.map(|n| n.into_raw().cast_const())
			.collect();
		let mut output_tensor_ptrs: Vec<*mut ort_sys::OrtValue> = output_tensors
			.iter()
			.map(|c| match c {
				Some(v) => v.ptr(),
				None => std::ptr::null_mut()
			})
			.collect();

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

		let outputs: Vec<Value> = output_tensors
			.into_iter()
			.enumerate()
			.map(|(i, v)| match v {
				Some(value) => value,
				None => unsafe {
					Value::from_ptr(
						NonNull::new(output_tensor_ptrs[i]).expect("OrtValue ptr returned from session Run should not be null"),
						Some(Arc::clone(&self.inner))
					)
				}
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

		Ok(SessionOutputs::new(output_names.into_iter(), outputs))
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
	pub fn run_async<'s, 'i, 'v: 'i + 's, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, 'v, N>> + 'static
	) -> Result<InferenceFut<'s, '_, NoSelectedOutputs>> {
		match input_values.into() {
			SessionInputs::ValueSlice(_) => unimplemented!("slices cannot be used in `run_async`"),
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(&self.inputs.iter().map(|input| input.name.to_string()).collect::<Vec<_>>(), input_values.into_iter(), None)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner_async(&input_values.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>(), input_values.into_iter().map(|(_, v)| v), None)
			}
		}
	}

	/// Asynchronously run input data through the ONNX graph, performing inference, with the given [`RunOptions`].
	/// See [`Session::run_with_options`] and [`Session::run_async`] for more details.
	pub fn run_async_with_options<'s, 'i, 'v: 'i + 's, 'r, O: SelectedOutputMarker, const N: usize>(
		&'s self,
		input_values: impl Into<SessionInputs<'i, 'v, N>> + 'static,
		run_options: &'r RunOptions<O>
	) -> Result<InferenceFut<'s, 'r, O>> {
		match input_values.into() {
			SessionInputs::ValueSlice(_) => unimplemented!("slices cannot be used in `run_async`"),
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(&self.inputs.iter().map(|input| input.name.to_string()).collect::<Vec<_>>(), input_values.into_iter(), Some(run_options))
			}
			SessionInputs::ValueMap(input_values) => self.run_inner_async(
				&input_values.iter().map(|(k, _)| k.to_string()).collect::<Vec<_>>(),
				input_values.into_iter().map(|(_, v)| v),
				Some(run_options)
			)
		}
	}

	fn run_inner_async<'s, 'v: 's, 'r, O: SelectedOutputMarker>(
		&'s self,
		input_names: &[String],
		input_values: impl Iterator<Item = SessionInputValue<'v>>,
		run_options: Option<&'r RunOptions<O>>
	) -> Result<InferenceFut<'s, 'r, O>> {
		let run_options = match run_options {
			Some(r) => RunOptionsRef::Ref(r),
			// create a `RunOptions` to pass to the future so that when it drops, it terminates inference - crucial
			// (performance-wise) for routines involving `tokio::select!` or timeouts
			None => RunOptionsRef::Arc(Arc::new(unsafe {
				// SAFETY: transmuting from `RunOptions<NoSelectedOutputs>` to `RunOptions<O>`; safe because its just a marker
				std::mem::transmute::<RunOptions<NoSelectedOutputs>, RunOptions<O>>(RunOptions::new()?)
			}))
		};

		let input_name_ptrs: Vec<*const c_char> = input_names
			.iter()
			.map(|n| CString::new(n.as_bytes()).unwrap_or_else(|_| unreachable!()))
			.map(|n| n.into_raw().cast_const())
			.collect();
		let output_name_ptrs: Vec<*const c_char> = self
			.outputs
			.iter()
			.map(|output| CString::new(output.name.as_str()).unwrap_or_else(|_| unreachable!()))
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

		ValueType::from_type_info(typeinfo_ptr)
	}
}
