//! Contains [`Session`], the main interface used to inference ONNX models.
//!
//! ```
//! # use ort::{session::Session, value::TensorRef};
//! # fn main() -> ort::Result<()> {
//! let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
//! let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
//! let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
//! # 	Ok(())
//! # }
//! ```

use alloc::{boxed::Box, format, string::String, sync::Arc, vec::Vec};
use core::{
	any::Any,
	ffi::{CStr, c_char},
	iter,
	marker::PhantomData,
	ops::{Deref, DerefMut},
	ptr::{self, NonNull},
	slice
};
#[cfg(feature = "std")]
use std::ffi::CString;

use smallvec::SmallVec;

use crate::{
	AsPointer, char_p_to_string,
	error::{Error, ErrorCode, Result, status_to_result},
	io_binding::IoBinding,
	memory::Allocator,
	metadata::ModelMetadata,
	ortsys,
	util::{STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS, with_cstr, with_cstr_ptr_array},
	value::{DynValue, Value, ValueType}
};

#[cfg(feature = "std")]
mod r#async;
pub mod builder;
pub mod input;
pub mod output;
pub mod run_options;
#[cfg(feature = "std")]
pub use self::r#async::InferenceFut;
#[cfg(feature = "std")]
use self::r#async::{AsyncInferenceContext, InferenceFutInner};
use self::{builder::SessionBuilder, run_options::UntypedRunOptions};
pub use self::{
	input::{SessionInputValue, SessionInputs},
	output::SessionOutputs,
	run_options::{HasSelectedOutputs, NoSelectedOutputs, RunOptions, SelectedOutputMarker}
};

/// Holds onto an [`ort_sys::OrtSession`] pointer and its associated allocator.
///
/// Internally, this is wrapped in an [`Arc`] and shared between a [`Session`] and any [`Value`]s created as a result
/// of [`Session::run`] to ensure that the [`Value`]s are kept alive until all references to the session are dropped.
#[derive(Debug)]
pub struct SharedSessionInner {
	session_ptr: NonNull<ort_sys::OrtSession>,
	pub(crate) allocator: Allocator,
	_initializers: SmallVec<Arc<DynValue>, 4>,
	/// Additional things we may need to hold onto for the duration of this session, like `OperatorDomain`s and
	/// DLL handles for operator libraries.
	_extras: SmallVec<Box<dyn Any>, 4>
}

unsafe impl Send for SharedSessionInner {}
unsafe impl Sync for SharedSessionInner {}

impl AsPointer for SharedSessionInner {
	type Sys = ort_sys::OrtSession;

	fn ptr(&self) -> *const Self::Sys {
		self.session_ptr.as_ptr()
	}
}

impl Drop for SharedSessionInner {
	fn drop(&mut self) {
		crate::debug!(ptr = ?self.session_ptr.as_ptr(), "dropping SharedSessionInner");
		ortsys![unsafe ReleaseSession(self.session_ptr.as_ptr())];
	}
}

/// An ONNX Runtime graph to be used for inference.
///
/// ```
/// # use ort::{session::Session, value::TensorRef};
/// # fn main() -> ort::Result<()> {
/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
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

impl Deref for InMemorySession<'_> {
	type Target = Session;
	fn deref(&self) -> &Self::Target {
		&self.session
	}
}
impl DerefMut for InMemorySession<'_> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.session
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

	/// Get a shared ([`Arc`]'d) reference to the underlying [`SharedSessionInner`], which holds the
	/// [`ort_sys::OrtSession`] pointer and the session allocator.
	#[must_use]
	pub fn inner(&self) -> Arc<SharedSessionInner> {
		Arc::clone(&self.inner)
	}

	/// Returns a list of initializers which are overridable (i.e. also graph inputs).
	#[must_use]
	pub fn overridable_initializers(&self) -> Vec<OverridableInitializer> {
		// can only fail if:
		// - index is out of bounds (impossible because of the loop)
		// - the model is not loaded (how could this even be possible?)
		let mut size = 0;
		ortsys![unsafe SessionGetOverridableInitializerCount(self.ptr(), &mut size).expect("infallible")];
		let allocator = Allocator::default();
		(0..size)
			.map(|i| {
				let mut name: *mut c_char = ptr::null_mut();
				ortsys![unsafe SessionGetOverridableInitializerName(self.ptr(), i, allocator.ptr().cast_mut(), &mut name).expect("infallible")];
				let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
				let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe SessionGetOverridableInitializerTypeInfo(self.ptr(), i, &mut typeinfo_ptr).expect("infallible"); nonNull(typeinfo_ptr)];
				let dtype = unsafe { ValueType::from_type_info(typeinfo_ptr) };
				OverridableInitializer { name, dtype }
			})
			.collect()
	}

	/// Run input data through the ONNX graph, performing inference.
	///
	/// See [`crate::inputs!`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{run_options::RunOptions, Session}, tensor::TensorElementType, value::{Value, ValueType, TensorRef}};
	/// # fn main() -> ort::Result<()> {
	/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run<'s, 'i, 'v: 'i, const N: usize>(&'s mut self, input_values: impl Into<SessionInputs<'i, 'v, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), None)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), None)
			}
		}
	}

	/// Run input data through the ONNX graph, performing inference, with a [`RunOptions`] struct. The most common usage
	/// of `RunOptions` is to allow the session run to be terminated from a different thread.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, run_options::RunOptions}, value::{Value, ValueType, TensorRef}, tensor::TensorElementType};
	/// # fn main() -> ort::Result<()> {
	/// # 	let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![&input], &*run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn run_with_options<'r, 's: 'r, 'i, 'v: 'i, O: SelectedOutputMarker, const N: usize>(
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<SessionOutputs<'r>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), Some(&run_options.inner))
			}
		}
	}

	fn run_inner<'i, 'r, 's: 'r, 'v: 'i>(
		&'s self,
		input_names: SmallVec<&str, { STACK_SESSION_INPUTS }>,
		input_values: SmallVec<&'i SessionInputValue<'v>, { STACK_SESSION_INPUTS }>,
		run_options: Option<&'r UntypedRunOptions>
	) -> Result<SessionOutputs<'r>> {
		if input_values.len() > input_names.len() {
			// If we provide more inputs than the model expects with `ort::inputs![a, b, c]`, then we get an `input_names` shorter
			// than `inputs`. ONNX Runtime will attempt to look up the name of all inputs before doing any checks, thus going out of
			// bounds of `input_names` and triggering a segfault, so we check that condition here. This will never trip for
			// `ValueMap` inputs since the number of names & values are always equal as its a vec of tuples.
			return Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("{} inputs were provided, but the model only accepts {}.", input_values.len(), input_names.len())
			));
		}

		let (output_names, mut output_tensors) = match run_options {
			Some(r) => r.outputs.resolve_outputs(&self.outputs),
			None => (self.outputs.iter().map(|o| o.name.as_str()).collect(), iter::repeat_with(|| None).take(self.outputs.len()).collect())
		};
		let output_value_ptrs: SmallVec<*mut ort_sys::OrtValue, { STACK_SESSION_OUTPUTS }> = output_tensors
			.iter_mut()
			.map(|c| match c {
				Some(v) => v.ptr_mut(),
				None => ptr::null_mut()
			})
			.collect();
		let input_value_ptrs: SmallVec<*const ort_sys::OrtValue, { STACK_SESSION_INPUTS }> = input_values.iter().map(|c| c.ptr()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr.as_ptr() } else { ptr::null() };

		with_cstr_ptr_array(&input_names, &|input_name_ptrs| {
			with_cstr_ptr_array(&output_names, &|output_name_ptrs| {
				ortsys![
					unsafe Run(
						self.inner.session_ptr.as_ptr(),
						run_options_ptr,
						input_name_ptrs.as_ptr(),
						input_value_ptrs.as_ptr(),
						input_value_ptrs.len(),
						output_name_ptrs.as_ptr(),
						output_name_ptrs.len(),
						output_value_ptrs.as_ptr().cast_mut()
					)?
				];
				Ok(())
			})
		})?;

		let outputs = output_tensors
			.into_iter()
			.enumerate()
			.map(|(i, v)| match v {
				Some(value) => value,
				None => unsafe {
					Value::from_ptr(
						NonNull::new(output_value_ptrs[i]).expect("OrtValue ptr returned from session Run should not be null"),
						Some(Arc::clone(&self.inner))
					)
				}
			})
			.collect();

		Ok(SessionOutputs::new(output_names, outputs))
	}

	pub fn run_binding<'b, 's: 'b>(&'s mut self, binding: &'b IoBinding) -> Result<SessionOutputs<'b>> {
		self.run_binding_inner(binding, None)
	}

	pub fn run_binding_with_options<'r, 'b, 's: 'b>(
		&'s mut self,
		binding: &'b IoBinding,
		run_options: &'r RunOptions<NoSelectedOutputs>
	) -> Result<SessionOutputs<'b>> {
		self.run_binding_inner(binding, Some(run_options))
	}

	fn run_binding_inner<'r, 'b, 's: 'b>(
		&'s self,
		binding: &'b IoBinding,
		run_options: Option<&'r RunOptions<NoSelectedOutputs>>
	) -> Result<SessionOutputs<'b>> {
		let run_options_ptr = if let Some(run_options) = run_options { run_options.ptr() } else { ptr::null() };
		ortsys![unsafe RunWithBinding(self.inner.ptr().cast_mut(), run_options_ptr, binding.ptr())?];

		let mut count = binding.output_values.len();
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			ortsys![unsafe GetBoundOutputValues(binding.ptr(), self.inner.allocator.ptr().cast_mut(), &mut output_values_ptr, &mut count)?; nonNull(output_values_ptr)];

			let output_values = unsafe { slice::from_raw_parts(output_values_ptr.as_ptr(), count) }
				.iter()
				.zip(binding.output_values.iter())
				.map(|(ptr, (_, value))| unsafe {
					if let Some(value) = value {
						DynValue::clone_of(value)
					} else {
						DynValue::from_ptr(NonNull::new(*ptr).expect("OrtValue ptrs returned by GetBoundOutputValues should not be null"), Some(self.inner()))
					}
				})
				.collect();
			unsafe {
				self.inner.allocator.free(output_values_ptr.as_ptr());
			}

			Ok(SessionOutputs::new(binding.output_values.iter().map(|(k, _)| k.as_str()).collect(), output_values))
		} else {
			Ok(SessionOutputs::new_empty())
		}
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
	/// # use ort::{session::{Session, run_options::RunOptions}, value::{Value, ValueType, TensorRef}, tensor::TensorElementType};
	/// # fn main() -> ort::Result<()> { tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
	/// let mut session = Session::builder()?.with_intra_threads(2)?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let options = RunOptions::new()?;
	/// let outputs = session.run_async(ort::inputs![TensorRef::from_array_view(&input)?], &options)?.await?;
	/// # 	Ok(())
	/// # }) }
	/// ```
	#[cfg(feature = "std")]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn run_async<'r, 's: 'r, 'i, 'v: 'i + 's, O: SelectedOutputMarker, const N: usize>(
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<InferenceFut<'r, 'v>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), &run_options.inner)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name.as_str()).collect(), input_values.iter().collect(), &run_options.inner)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner_async(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), &run_options.inner)
			}
		}
	}

	#[cfg(feature = "std")]
	fn run_inner_async<'i, 'r, 's: 'r, 'v: 'i + 's>(
		&'s self,
		input_names: SmallVec<&str, { STACK_SESSION_INPUTS }>,
		input_values: SmallVec<&SessionInputValue<'v>, { STACK_SESSION_INPUTS }>,
		run_options: &'r UntypedRunOptions
	) -> Result<InferenceFut<'r, 'v>> {
		let input_name_ptrs = input_names
			.into_iter()
			.map(|name| CString::new(name.as_bytes()).map(|s| s.into_raw().cast_const()))
			.collect::<Result<SmallVec<*const c_char, { STACK_SESSION_INPUTS }>, _>>()?;

		let mut input_inner_holders = SmallVec::with_capacity(input_values.len());
		let mut input_ort_values = SmallVec::with_capacity(input_values.len());
		for input in input_values {
			input_ort_values.push(input.ptr());
			input_inner_holders.push(Arc::clone(match input {
				SessionInputValue::ViewMut(v) => &(**v).inner,
				SessionInputValue::View(v) => &(**v).inner,
				SessionInputValue::Owned(v) => &v.inner
			}));
		}

		let (output_names, mut output_tensors) = run_options.outputs.resolve_outputs(&self.outputs);
		let output_name_ptrs = output_names
			.iter()
			.map(|n| CString::new(*n).unwrap_or_else(|_| unreachable!()))
			.map(|n| n.into_raw().cast_const())
			.collect();
		let output_tensor_ptrs = output_tensors
			.iter_mut()
			.map(|c| match c {
				Some(v) => v.ptr_mut(),
				None => ptr::null_mut()
			})
			.collect();

		let async_inner = Arc::new(InferenceFutInner::new());

		// AsyncInferenceContext can get pretty huge so we should see if we can bump MSRV to 1.82 and use `Box::new_uninit()`
		// if it causes problems
		let ctx = Box::leak(Box::new(AsyncInferenceContext {
			inner: Arc::clone(&async_inner),
			// everything allocated within `run_inner_async` needs to be kept alive until we are certain inference has completed and ONNX Runtime no longer
			// needs the data - i.e. when `async_callback` is called. `async_callback` will free all of this data just like we do in `run_inner`
			input_ort_values,
			_input_inner_holders: input_inner_holders,
			input_name_ptrs,
			output_name_ptrs,
			output_names,
			output_value_ptrs: output_tensor_ptrs,
			session_inner: &self.inner
		}));

		ortsys![
			unsafe RunAsync(
				self.inner.session_ptr.as_ptr(),
				run_options.ptr.as_ptr(),
				ctx.input_name_ptrs.as_ptr(),
				ctx.input_ort_values.as_ptr(),
				ctx.input_ort_values.len(),
				ctx.output_name_ptrs.as_ptr(),
				ctx.output_name_ptrs.len(),
				ctx.output_value_ptrs.as_mut_ptr(),
				Some(self::r#async::async_callback),
				ctx as *mut _ as *mut ort_sys::c_void
			)?
		];

		Ok(InferenceFut::new(async_inner, run_options))
	}

	/// Gets the session model metadata. See [`ModelMetadata`] for more info.
	pub fn metadata(&self) -> Result<ModelMetadata<'_>> {
		let mut metadata_ptr: *mut ort_sys::OrtModelMetadata = ptr::null_mut();
		ortsys![unsafe SessionGetModelMetadata(self.inner.session_ptr.as_ptr(), &mut metadata_ptr)?; nonNull(metadata_ptr)];
		Ok(unsafe { ModelMetadata::new(metadata_ptr) })
	}

	/// Returns the time that profiling was started, in nanoseconds.
	pub fn profiling_start_ns(&self) -> Result<u64> {
		let mut out = 0;
		ortsys![unsafe SessionGetProfilingStartTimeNs(self.inner.session_ptr.as_ptr(), &mut out)?];
		Ok(out)
	}

	/// Ends profiling for this session.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiling file will be empty.
	pub fn end_profiling(&mut self) -> Result<String> {
		let mut profiling_name: *mut c_char = ptr::null_mut();
		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr.as_ptr(), self.inner.allocator.ptr().cast_mut(), &mut profiling_name)?; nonNull(profiling_name)];
		dangerous::raw_pointer_to_string(&self.inner.allocator, profiling_name.as_ptr())
	}

	/// Sets this session's [workload type][`WorkloadType`] to instruct execution providers to prioritize performance or
	/// efficiency.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{run_options::RunOptions, Session, WorkloadType}, tensor::TensorElementType, value::{Value, ValueType, TensorRef}};
	/// # fn main() -> ort::Result<()> {
	/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// session.set_workload_type(WorkloadType::Efficient)?;
	///
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn set_workload_type(&mut self, workload_type: WorkloadType) -> Result<()> {
		static KEY: &[u8] = b"ep.dynamic.workload_type\0";
		match workload_type {
			WorkloadType::Default => self.set_dynamic_option(KEY.as_ptr().cast(), c"Default".as_ptr().cast()),
			WorkloadType::Efficient => self.set_dynamic_option(KEY.as_ptr().cast(), c"Efficient".as_ptr().cast())
		}
	}

	pub(crate) fn set_dynamic_option(&mut self, key: *const c_char, value: *const c_char) -> Result<()> {
		ortsys![unsafe SetEpDynamicOptions(self.inner.session_ptr.as_ptr(), &key, &value, 1)?];
		Ok(())
	}

	pub fn opset_for_domain(&self, domain: impl AsRef<str>) -> Result<u32> {
		with_cstr(domain.as_ref().as_bytes(), &|domain| {
			let mut opset = 0;
			ortsys![@editor: unsafe SessionGetOpsetForDomain(self.inner.session_ptr.as_ptr(), domain.as_ptr(), &mut opset)?];
			Ok(opset as u32)
		})
	}
}

/// Workload type, used to signal to execution providers whether to prioritize performance or efficiency.
///
/// See [`Session::set_workload_type`].
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadType {
	/// Prioritize performance. This is the default behavior when the workload type is not overridden.
	#[default]
	Default,
	/// Prioritize efficiency, by i.e. reducing scheduling priority and/or offloading to efficiency cores.
	Efficient
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
// Allowing `Sync` segfaults with CUDA, DirectML, and seemingly any EP other than the CPU EP. I'm not certain if it's a
// temporary bug in ONNX Runtime or a wontfix. Maybe this impl should be removed just to be safe?
unsafe impl Sync for Session {}

impl AsPointer for Session {
	type Sys = ort_sys::OrtSession;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.ptr()
	}
}

#[derive(Debug, Clone)]
pub struct OverridableInitializer {
	name: String,
	dtype: ValueType
}

impl OverridableInitializer {
	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn dtype(&self) -> &ValueType {
		&self.dtype
	}
}

mod dangerous {
	use super::*;

	pub(super) fn extract_inputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![SessionGetInputCount];
		extract_io_count(f, session_ptr)
	}

	pub(super) fn extract_outputs_count(session_ptr: NonNull<ort_sys::OrtSession>) -> Result<usize> {
		let f = ortsys![SessionGetOutputCount];
		extract_io_count(f, session_ptr)
	}

	fn extract_io_count(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, *mut usize) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>
	) -> Result<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr.as_ptr(), &mut num_nodes) };
		unsafe { status_to_result(status) }?;
		Ok(num_nodes)
	}

	fn extract_input_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<String> {
		let f = ortsys![SessionGetInputName];
		extract_io_name(f, session_ptr, allocator, i)
	}

	fn extract_output_name(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<String> {
		let f = ortsys![SessionGetOutputName];
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
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, usize, *mut ort_sys::OrtAllocator, *mut *mut c_char) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>,
		allocator: &Allocator,
		i: usize
	) -> Result<String> {
		let mut name_ptr: *mut c_char = ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_ptr) };
		unsafe { status_to_result(status) }?;
		if name_ptr.is_null() {
			crate::util::cold();
			return Err(crate::Error::new(concat!("expected `name_ptr` to not be null")));
		}

		raw_pointer_to_string(allocator, name_ptr)
	}

	pub(super) fn extract_input(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Input> {
		let input_name = extract_input_name(session_ptr, allocator, i)?;
		let f = ortsys![SessionGetInputTypeInfo];
		let input_type = extract_io(f, session_ptr, i)?;
		Ok(Input { name: input_name, input_type })
	}

	pub(super) fn extract_output(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Output> {
		let output_name = extract_output_name(session_ptr, allocator, i)?;
		let f = ortsys![SessionGetOutputTypeInfo];
		let output_type = extract_io(f, session_ptr, i)?;
		Ok(Output { name: output_name, output_type })
	}

	fn extract_io(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, usize, *mut *mut ort_sys::OrtTypeInfo) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>,
		i: usize
	) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, &mut typeinfo_ptr) };
		unsafe { status_to_result(status) }?;
		let Some(typeinfo_ptr) = NonNull::new(typeinfo_ptr) else {
			crate::util::cold();
			return Err(crate::Error::new(concat!("expected `typeinfo_ptr` to not be null")));
		};
		Ok(unsafe { ValueType::from_type_info(typeinfo_ptr) })
	}
}
