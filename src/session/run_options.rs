use std::{collections::HashMap, ffi::CString, marker::PhantomData, ptr::NonNull, sync::Arc};

use crate::{ortsys, DynValue, Error, Output, Result, Value, ValueTypeMarker};

#[derive(Debug)]
pub struct OutputSelector {
	use_defaults: bool,
	default_blocklist: Vec<String>,
	allowlist: Vec<String>,
	preallocated_outputs: HashMap<String, Value>
}

impl Default for OutputSelector {
	fn default() -> Self {
		Self {
			use_defaults: true,
			allowlist: Vec::new(),
			default_blocklist: Vec::new(),
			preallocated_outputs: HashMap::new()
		}
	}
}

impl OutputSelector {
	pub fn no_default() -> Self {
		Self {
			use_defaults: false,
			..Default::default()
		}
	}

	pub fn with(mut self, name: impl Into<String>) -> Self {
		self.allowlist.push(name.into());
		self
	}

	pub fn without(mut self, name: impl Into<String>) -> Self {
		self.default_blocklist.push(name.into());
		self
	}

	pub fn preallocate<T: ValueTypeMarker>(mut self, name: impl Into<String>, value: Value<T>) -> Self {
		self.preallocated_outputs.insert(name.into(), value.into_dyn());
		self
	}

	pub(crate) fn resolve_outputs<'a, 's: 'a>(&'a self, outputs: &'s [Output]) -> (Vec<&'a str>, Vec<Option<DynValue>>) {
		if self.use_defaults { outputs.iter() } else { [].iter() }
			.map(|o| &o.name)
			.filter(|n| !self.default_blocklist.contains(n))
			.chain(self.allowlist.iter())
			.map(|n| {
				(
					n.as_str(),
					self.preallocated_outputs.get(n).map(|v| DynValue {
						inner: Arc::clone(&v.inner),
						_markers: PhantomData
					})
				)
			})
			.unzip()
	}
}

/// A structure which can be passed to [`crate::Session::run_with_options`] to allow terminating/unterminating a session
/// inference run from a different thread.
#[derive(Debug)]
pub struct RunOptions {
	pub(crate) run_options_ptr: NonNull<ort_sys::OrtRunOptions>,
	pub(crate) outputs: OutputSelector
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
			run_options_ptr: unsafe { NonNull::new_unchecked(run_options_ptr) },
			outputs: OutputSelector::default()
		})
	}

	pub fn with_outputs(mut self, outputs: OutputSelector) -> Self {
		self.outputs = outputs;
		self
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
	/// let res = session.run_with_options(ort::inputs![input]?, &*run_options);
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
	/// let res = session.run_with_options(ort::inputs![input]?, &*run_options);
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
