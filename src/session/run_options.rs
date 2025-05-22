use alloc::{string::String, sync::Arc, vec::Vec};
use core::{
	ffi::{CStr, c_char, c_int},
	marker::PhantomData,
	mem,
	ptr::{self, NonNull}
};

use smallvec::SmallVec;

use crate::{
	AsPointer,
	adapter::{Adapter, AdapterInner},
	error::Result,
	logging::LogLevel,
	ortsys,
	session::Output,
	util::{MiniMap, STACK_SESSION_OUTPUTS, with_cstr},
	value::{DynValue, Value, ValueTypeMarker}
};

/// Allows selecting/deselecting/preallocating the outputs of a [`Session`] inference call.
///
/// ```
/// # use std::sync::Arc;
/// # use ort::{session::{Session, run_options::{RunOptions, OutputSelector}}, memory::Allocator, value::Tensor};
/// # fn main() -> ort::Result<()> {
/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = Tensor::<f32>::new(&Allocator::default(), [1_usize, 64, 64, 3])?;
///
/// let output0 = session.outputs[0].name.as_str();
/// let options = RunOptions::new()?.with_outputs(
/// 	// Disable all outputs...
/// 	OutputSelector::no_default()
/// 		// except for the first one...
/// 		.with(output0)
/// 		// and since this is a 2x upsampler model, pre-allocate the output to be twice as large.
/// 		.preallocate(output0, Tensor::<f32>::new(&Allocator::default(), [1_usize, 128, 128, 3])?)
/// );
///
/// // `outputs[0]` will be the tensor we just pre-allocated.
/// let outputs = session.run_with_options(ort::inputs![input], &options)?;
/// # 	Ok(())
/// # }
/// ```
///
/// [`Session`]: crate::session::Session
#[derive(Debug)]
pub struct OutputSelector {
	use_defaults: bool,
	default_blocklist: Vec<String>,
	allowlist: Vec<String>,
	preallocated_outputs: MiniMap<String, Value>
}

impl Default for OutputSelector {
	/// Creates an [`OutputSelector`] that enables all outputs by default. Use [`OutputSelector::without`] to disable a
	/// specific output.
	fn default() -> Self {
		Self {
			use_defaults: true,
			allowlist: Vec::new(),
			default_blocklist: Vec::new(),
			preallocated_outputs: MiniMap::new()
		}
	}
}

impl OutputSelector {
	/// Creates an [`OutputSelector`] that does not enable any outputs. Use [`OutputSelector::with`] to enable a
	/// specific output.
	pub fn no_default() -> Self {
		Self {
			use_defaults: false,
			..Default::default()
		}
	}

	/// Mark the output specified by the `name` for inclusion.
	pub fn with(mut self, name: impl Into<String>) -> Self {
		self.allowlist.push(name.into());
		self
	}

	/// Mark the output specified by `name` to be **excluded**. ONNX Runtime may prune some of the output node's
	/// ancestor nodes.
	pub fn without(mut self, name: impl Into<String>) -> Self {
		self.default_blocklist.push(name.into());
		self
	}

	/// Pre-allocates an output. Assuming the type & shape of the value matches what is expected by the model, the
	/// output value corresponding to `name` returned by the inference call will be the exact same value as the
	/// pre-allocated value.
	///
	/// **The same value will be reused as long as this [`OutputSelector`] and its parent [`RunOptions`] is used**, so
	/// if you use the same `RunOptions` across multiple runs with a preallocated value, the preallocated value will be
	/// overwritten upon each run.
	///
	/// This can improve performance if the size and type of the output is known, and does not change between runs, i.e.
	/// for an ODE or embeddings model.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, run_options::{RunOptions, OutputSelector}}, memory::Allocator, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = Tensor::<f32>::new(&Allocator::default(), [1_usize, 64, 64, 3])?;
	///
	/// let output0 = session.outputs[0].name.as_str();
	/// let options = RunOptions::new()?.with_outputs(
	/// 	OutputSelector::default()
	/// 		.preallocate(output0, Tensor::<f32>::new(&Allocator::default(), [1_usize, 128, 128, 3])?)
	/// );
	///
	/// let outputs = session.run_with_options(ort::inputs![input], &options)?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn preallocate<T: ValueTypeMarker>(mut self, name: impl Into<String>, value: Value<T>) -> Self {
		self.preallocated_outputs.insert(name.into(), value.into_dyn());
		self
	}

	pub(crate) fn resolve_outputs<'a, 's: 'a>(
		&'a self,
		outputs: &'s [Output]
	) -> (SmallVec<&'a str, { STACK_SESSION_OUTPUTS }>, SmallVec<Option<DynValue>, { STACK_SESSION_OUTPUTS }>) {
		if self.use_defaults { outputs.iter() } else { [].iter() }
			.map(|o| &o.name)
			.filter(|n| !self.default_blocklist.contains(n))
			.chain(self.allowlist.iter())
			.map(|n| (n.as_str(), self.preallocated_outputs.get(n).map(DynValue::clone_of)))
			.unzip()
	}
}

/// Types that specify whether a [`RunOptions`] was configured with an [`OutputSelector`].
pub trait SelectedOutputMarker {}
/// Marks that a [`RunOptions`] was not configured with an [`OutputSelector`].
pub struct NoSelectedOutputs;
impl SelectedOutputMarker for NoSelectedOutputs {}
/// Marks that a [`RunOptions`] was configured with an [`OutputSelector`].
pub struct HasSelectedOutputs;
impl SelectedOutputMarker for HasSelectedOutputs {}

#[derive(Debug)]
pub(crate) struct UntypedRunOptions {
	pub(crate) ptr: NonNull<ort_sys::OrtRunOptions>,
	pub(crate) outputs: OutputSelector,
	adapters: Vec<Arc<AdapterInner>>
}

impl UntypedRunOptions {
	pub fn terminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsSetTerminate(self.ptr.as_ptr())?];
		Ok(())
	}
}

// https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ac2a08cac0a657604bd5899e0d1a13675
unsafe impl Send for UntypedRunOptions {}

/// Allows for finer control over session inference.
///
/// [`RunOptions`] provides three main features:
/// - **Run tagging**: Each individual session run can have a uniquely identifiable tag attached with
///   [`RunOptions::set_tag`], which will show up in logs. This can be especially useful for debugging
///   performance/errors in inference servers.
/// - **Termination**: Allows for terminating an inference call from another thread; when [`RunOptions::terminate`] is
///   called, any sessions currently running under that [`RunOptions`] instance will halt graph execution as soon as the
///   termination signal is received. This allows for [`Session::run_async`]'s cancel-safety.
/// - **Output specification**: Certain session outputs can be [disabled](`OutputSelector::without`) or
///   [pre-allocated](`OutputSelector::preallocate`). Disabling an output might mean ONNX Runtime will not execute parts
///   of the graph that are only used by that output. Pre-allocation can reduce expensive re-allocations by allowing you
///   to use the same memory across runs.
///
/// [`RunOptions`] can be passed to most places where a session can be inferred, e.g.
/// [`Session::run_with_options`], [`Session::run_async`],
/// [`Session::run_binding_with_options`]. Some of these patterns (notably `IoBinding`) do not accept
/// [`OutputSelector`], hence [`RunOptions`] contains an additional type parameter that marks whether or not outputs
/// have been selected.
///
/// [`Session::run_async`]: crate::session::Session::run_async
/// [`Session::run_with_options`]: crate::session::Session::run_with_options
/// [`Session::run_binding_with_options`]: crate::session::Session::run_binding_with_options
#[derive(Debug)]
pub struct RunOptions<O: SelectedOutputMarker = NoSelectedOutputs> {
	pub(crate) inner: UntypedRunOptions,
	_marker: PhantomData<O>
}

// Only allow `Sync` if we don't have (potentially pre-allocated) outputs selected.
// Allowing `Sync` here would mean a single pre-allocated `Value` could be mutated simultaneously in different threads -
// a brazen crime against crabkind.
unsafe impl Sync for RunOptions<NoSelectedOutputs> {}

impl RunOptions {
	/// Creates a new [`RunOptions`] struct.
	pub fn new() -> Result<RunOptions<NoSelectedOutputs>> {
		let mut run_options_ptr: *mut ort_sys::OrtRunOptions = ptr::null_mut();
		ortsys![unsafe CreateRunOptions(&mut run_options_ptr)?; nonNull(run_options_ptr)];
		Ok(RunOptions {
			inner: UntypedRunOptions {
				ptr: run_options_ptr,
				outputs: OutputSelector::default(),
				adapters: Vec::new()
			},
			_marker: PhantomData
		})
	}
}

impl<O: SelectedOutputMarker> RunOptions<O> {
	/// Select/deselect/preallocate outputs for this run.
	///
	/// See [`OutputSelector`] for more details.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, run_options::{RunOptions, OutputSelector}}, memory::Allocator, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = Tensor::<f32>::new(&Allocator::default(), [1_usize, 64, 64, 3])?;
	///
	/// let output0 = session.outputs[0].name.as_str();
	/// let options = RunOptions::new()?.with_outputs(
	/// 	// Disable all outputs...
	/// 	OutputSelector::no_default()
	/// 		// except for the first one...
	/// 		.with(output0)
	/// 		// and since this is a 2x upsampler model, pre-allocate the output to be twice as large.
	/// 		.preallocate(output0, Tensor::<f32>::new(&Allocator::default(), [1_usize, 128, 128, 3])?)
	/// );
	///
	/// // `outputs[0]` will be the tensor we just pre-allocated.
	/// let outputs = session.run_with_options(ort::inputs![input], &options)?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn with_outputs(mut self, outputs: OutputSelector) -> RunOptions<HasSelectedOutputs> {
		self.inner.outputs = outputs;
		unsafe { mem::transmute(self) }
	}

	/// Sets a tag to identify this run in logs.
	pub fn with_tag(mut self, tag: impl AsRef<str>) -> Result<Self> {
		self.set_tag(tag).map(|_| self)
	}

	/// Sets a tag to identify this run in logs.
	pub fn set_tag(&mut self, tag: impl AsRef<str>) -> Result<()> {
		with_cstr(tag.as_ref().as_bytes(), &|tag| {
			ortsys![unsafe RunOptionsSetRunTag(self.inner.ptr.as_ptr(), tag.as_ptr())?];
			Ok(())
		})
	}

	pub fn tag(&self) -> Result<&str> {
		let mut tag_ptr: *const c_char = ptr::null();
		ortsys![unsafe RunOptionsGetRunTag(self.inner.ptr.as_ptr(), &mut tag_ptr)?];
		Ok(unsafe { CStr::from_ptr(tag_ptr) }.to_str()?)
	}

	/// Sets the termination flag for the runs associated with this [`RunOptions`].
	///
	/// This function returns immediately (it does not wait for the session run to terminate). The run will terminate as
	/// soon as it is able to.
	///
	/// ```no_run
	/// # // no_run because upsample.onnx is too simple of a model for the termination signal to be reliable enough
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, run_options::{RunOptions, OutputSelector}}, value::Value};
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
	/// let res = session.run_with_options(ort::inputs![input], &*run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.`
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn terminate(&self) -> Result<()> {
		self.inner.terminate()
	}

	/// Resets the termination flag for the runs associated with [`RunOptions`].
	///
	/// ```no_run
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, run_options::{RunOptions, OutputSelector}}, value::Value};
	/// # fn main() -> ort::Result<()> {
	/// # 	let mut session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
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
	/// let res = session.run_with_options(ort::inputs![input], &*run_options);
	/// assert!(res.is_ok());
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn unterminate(&self) -> Result<()> {
		ortsys![unsafe RunOptionsUnsetTerminate(self.inner.ptr.as_ptr())?];
		Ok(())
	}

	/// Adds a custom configuration option to the `RunOptions`.
	///
	/// This can be used to, for example, configure the graph ID when using compute graphs with an execution provider
	/// like CUDA:
	/// ```no_run
	/// # use std::sync::Arc;
	/// # use ort::session::run_options::RunOptions;
	/// # fn main() -> ort::Result<()> {
	/// let mut run_options = RunOptions::new()?;
	/// run_options.add_config_entry("gpu_graph_id", "1")?;
	/// # 	Ok(())
	/// # }
	/// ```
	pub fn add_config_entry(&mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Result<()> {
		with_cstr(key.as_ref().as_bytes(), &|key| {
			with_cstr(value.as_ref().as_bytes(), &|value| {
				ortsys![unsafe AddRunConfigEntry(self.inner.ptr.as_ptr(), key.as_ptr(), value.as_ptr())?];
				Ok(())
			})
		})
	}

	pub fn add_adapter(&mut self, adapter: &Adapter) -> Result<()> {
		ortsys![unsafe RunOptionsAddActiveLoraAdapter(self.inner.ptr.as_ptr(), adapter.ptr())?];
		self.inner.adapters.push(Arc::clone(&adapter.inner));
		Ok(())
	}

	pub fn set_log_level(&mut self, level: LogLevel) -> Result<()> {
		ortsys![unsafe RunOptionsSetRunLogSeverityLevel(self.ptr_mut(), ort_sys::OrtLoggingLevel::from(level) as _)?];
		Ok(())
	}

	pub fn log_level(&self) -> Result<LogLevel> {
		let mut log_level = ort_sys::OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
		ortsys![unsafe RunOptionsGetRunLogSeverityLevel(self.ptr(), &mut log_level as *mut ort_sys::OrtLoggingLevel as *mut _)?];
		Ok(LogLevel::from(log_level))
	}

	pub fn set_log_verbosity(&mut self, verbosity: c_int) -> Result<()> {
		ortsys![unsafe RunOptionsSetRunLogVerbosityLevel(self.ptr_mut(), verbosity)?];
		Ok(())
	}

	pub fn log_verbosity(&self) -> Result<i32> {
		let mut verbosity = 0;
		ortsys![unsafe RunOptionsGetRunLogVerbosityLevel(self.ptr(), &mut verbosity)?];
		Ok(verbosity)
	}

	pub fn disable_device_sync(&mut self) -> Result<()> {
		self.add_config_entry("disable_synchronize_execution_providers", "1")
	}
}

impl<O: SelectedOutputMarker> AsPointer for RunOptions<O> {
	type Sys = ort_sys::OrtRunOptions;

	fn ptr(&self) -> *const Self::Sys {
		self.inner.ptr.as_ptr()
	}
}

impl<O: SelectedOutputMarker> Drop for RunOptions<O> {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseRunOptions(self.inner.ptr.as_ptr())];
	}
}
