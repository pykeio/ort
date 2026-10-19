//! Contains [`Session`], the main interface used to inference ONNX models.
//!
//! ```
//! # use ort::{session::Session, value::TensorRef};
//! # fn main() -> ort::Result<()> {
//! # let env = ort::test_util::test_env().clone();
//! let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
//! let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
//! let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
//! # 	Ok(())
//! # }
//! ```

use alloc::{
	string::{String, ToString},
	sync::Arc,
	vec::Vec
};
use core::{
	any::Any,
	ffi::{CStr, c_char},
	marker::PhantomData,
	ops::Deref,
	ptr::{self, NonNull}
};

use smallvec::SmallVec;

use crate::{
	AsPointer,
	environment::Environment,
	error::{Error, Result},
	memory::Allocator,
	ortsys,
	util::{AllocatedString, with_cstr},
	value::{Outlet, ValueType}
};
#[cfg(feature = "api-23")]
use crate::{memory::MemoryInfo, util::MiniMap};

#[cfg(feature = "api-20")]
mod adapter;
#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
mod r#async;
pub mod builder;
mod input;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
mod io_binding;
mod metadata;
mod output;
mod run;
mod run_options;
#[cfg(feature = "api-20")]
#[cfg_attr(docsrs, doc(cfg(feature = "api-20")))]
pub use self::adapter::Adapter;
#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
pub use self::r#async::InferenceFut;
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
pub use self::io_binding::IoBinding;
use self::{builder::SessionBuilder, run_options::UntypedRunOptions};
pub use self::{
	input::{SessionInputValue, SessionInputs},
	metadata::ModelMetadata,
	output::SessionOutputs,
	run_options::{HasSelectedOutputs, NoSelectedOutputs, OutputSelector, RunOptions, SelectedOutputMarker}
};

/// Holds onto an [`ort_sys::OrtSession`] pointer and its associated allocator.
///
/// Internally, this is wrapped in an [`Arc`] and shared between a [`Session`] and any [`Value`]s created as a result
/// of [`Session::run`] to ensure that the [`Value`]s are kept alive until all references to the session are dropped.
#[derive(Debug)]
pub struct SharedSessionInner {
	session_ptr: NonNull<ort_sys::OrtSession>,
	pub(crate) allocator: Allocator,
	#[cfg(feature = "api-23")]
	pub(crate) inputs_memory_info: MiniMap<String, MemoryInfo<'static>>,
	#[cfg(feature = "api-23")]
	pub(crate) outputs_memory_info: MiniMap<String, MemoryInfo<'static>>,
	/// Additional things we may need to hold onto for the duration of this session, like `OperatorDomain`s and
	/// DLL handles for operator libraries.
	_extras: SmallVec<[Arc<dyn Any>; 4]>,
	_environment: Environment
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
		ortsys![unsafe ReleaseSession(self.session_ptr.as_ptr())];
		crate::logging::drop!(Session, self.session_ptr);
	}
}

/// An ONNX Runtime graph to be used for inference.
///
/// Use [`SessionBuilder`]/[`Session::builder`] to create a session, and [`Session::run`] to perform inference.
///
/// ```
/// # use ort::{session::Session, value::TensorRef};
/// # fn main() -> ort::Result<()> {
/// # let env = ort::test_util::test_env().clone();
/// let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
/// # 	Ok(())
/// # }
/// ```
///
/// ### Why does `run` take `&mut self`?
/// `run_*` methods take `&mut self` not because they mutate internal state, but because various ONNX Runtime
/// internals -- like certain EP allocators or statistics trackers -- are not thread safe. Previous versions of
/// `ort` that allowed concurrent inference often saw crashes and memory corruption, hence the change.
///
/// If you need concurrent inference, we recommend creating one session per thread or batching multiple requests
/// at once until ONNX Runtime's thread safety issues are fixed.
#[derive(Debug)]
pub struct Session {
	pub(crate) inner: Arc<SharedSessionInner>,
	inputs: Vec<Outlet>,
	outputs: Vec<Outlet>
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

impl Session {
	pub fn inputs(&self) -> &[Outlet] {
		&self.inputs
	}

	pub fn outputs(&self) -> &[Outlet] {
		&self.outputs
	}

	/// Creates a new [`SessionBuilder`].
	pub fn builder(environment: &Environment) -> Result<SessionBuilder> {
		SessionBuilder::new(environment)
	}

	/// Returns this session's [`Allocator`].
	pub fn allocator(&self) -> &Allocator {
		&self.inner.allocator
	}

	/// Creates a new [`IoBinding`] for this session.
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
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
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{RunOptions, Session}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let session = Session::builder(&env)?.commit_from_file("tests/data/overridable_initializer.onnx")?;
	///
	/// let mut overridable_initializers = session.overridable_initializers();
	/// assert_eq!(overridable_initializers.len(), 1);
	/// let f1 = overridable_initializers.pop().unwrap();
	/// assert_eq!(f1.name(), "F1");
	/// assert!(f1.dtype().is_tensor());
	/// # 	Ok(())
	/// # }
	/// ```
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
				let mut name_ptr: *mut c_char = ptr::null_mut();
				ortsys![unsafe SessionGetOverridableInitializerName(self.ptr(), i, allocator.ptr().cast_mut(), &mut name_ptr).expect("infallible")];
				let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned();
				unsafe { allocator.free(name_ptr) };
				let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = ptr::null_mut();
				ortsys![unsafe SessionGetOverridableInitializerTypeInfo(self.ptr(), i, &mut typeinfo_ptr).expect("infallible"); nonNull(typeinfo_ptr)];
				let dtype = unsafe { ValueType::from_type_info(typeinfo_ptr) };
				OverridableInitializer { name, dtype }
			})
			.collect()
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

	#[cfg(feature = "api-20")]
	pub(crate) fn set_dynamic_option(&mut self, key: *const c_char, value: *const c_char) -> Result<()> {
		ortsys![unsafe SetEpDynamicOptions(self.inner.session_ptr.as_ptr(), &key, &value, 1)?];
		Ok(())
	}

	/// Returns the version of the opset domain used by the model.
	///
	/// Requires the Model Editor API to be supported by the backend.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{RunOptions, Session}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let session = Session::builder(&env)?.commit_from_file("tests/data/lora_model.onnx")?;
	/// assert_eq!(session.opset_for_domain(ort::editor::ONNX_DOMAIN), Some(21));
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "api-22")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-22")))]
	pub fn opset_for_domain(&self, domain: impl AsRef<str>) -> Option<u32> {
		with_cstr(domain.as_ref().as_bytes(), &|domain| {
			let mut opset = 0;
			ortsys![@editor: unsafe SessionGetOpsetForDomain(self.inner.session_ptr.as_ptr(), domain.as_ptr(), &mut opset)?];
			Ok(opset as u32)
		})
		.ok()
	}
}

macro_rules! mut_methods {
	($(
		$(#[$m:meta])*
		$vis:vis fn $name:ident $([$($fn_bounds:tt)*])? (&$($lt:lifetime)? mut $self:ident $(, $arg:ident: $arg_ty:ty)*) $(-> $ret:ty)?
		$body:block
	)*) => {
		impl $crate::session::Session {
			$(
				$(#[$m])*
				$vis fn $name $(<$($fn_bounds)*>)? (&$($lt)? mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				$body
			)*
		}

		impl<'ims> $crate::session::InMemorySession<'ims> {
			$(
				$(#[$m])*
				$vis fn $name $(<$($fn_bounds)*>)? (&$($lt)? mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				{
					$self.session.$name($($arg),*)
				}
			)*
		}
	};

	($(
		$(#[$m:meta])*
		$vis:vis async fn $name:ident $([$($fn_bounds:tt)*])? (&$($lt:lifetime)? mut $self:ident $(, $arg:ident: $arg_ty:ty)*) $(-> $ret:ty)?
		$body:block
	)*) => {
		impl $crate::session::Session {
			$(
				$(#[$m])*
				$vis async fn $name $(<$($fn_bounds)*>)? (&$($lt)? mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				$body
			)*
		}

		impl<'ims> $crate::session::InMemorySession<'ims> {
			$(
				$(#[$m])*
				$vis async fn $name $(<$($fn_bounds)*>)? (&$($lt)? mut $self $(, $arg: $arg_ty)*) $(-> $ret)?
				{
					$self.session.$name($($arg),*).await
				}
			)*
		}
	};
}
pub(crate) use mut_methods;

mut_methods! {
	/// Ends profiling for this session. Returns the file name of the finalized profile.
	///
	/// Note that this must be explicitly called at the end of profiling, otherwise the profiling file will be empty.
	pub fn end_profiling(&mut self) -> Result<String> {
		let mut profiling_name: *mut c_char = ptr::null_mut();
		ortsys![unsafe SessionEndProfiling(self.inner.session_ptr.as_ptr(), self.inner.allocator.ptr().cast_mut(), &mut profiling_name)?; nonNull(profiling_name)];
		unsafe { AllocatedString::from_ptr(profiling_name.as_ptr(), &self.inner.allocator) }.map(|x| x.to_string())
	}

	/// Sets this session's [workload type][`WorkloadType`] to instruct execution providers to prioritize performance or
	/// efficiency.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{RunOptions, Session, WorkloadType}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
	/// session.set_workload_type(WorkloadType::Efficient)?;
	///
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(feature = "api-20")]
	#[cfg_attr(docsrs, doc(cfg(feature = "api-20")))]
	pub fn set_workload_type(&mut self, workload_type: WorkloadType) -> Result<()> {
		static KEY: &[u8] = b"ep.dynamic.workload_type\0";
		match workload_type {
			WorkloadType::Default => self.set_dynamic_option(KEY.as_ptr().cast(), c"Default".as_ptr().cast()),
			WorkloadType::Efficient => self.set_dynamic_option(KEY.as_ptr().cast(), c"Efficient".as_ptr().cast())
		}
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
	/// Prioritize efficiency, e.g. by reducing scheduling priority and/or offloading to efficiency cores.
	Efficient
}

// https://github.com/microsoft/onnxruntime/issues/114
unsafe impl Send for Session {}
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

pub(crate) mod io {
	use alloc::vec;

	use super::*;

	pub(super) fn extract_io_count(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, *mut usize) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>
	) -> Result<usize> {
		let mut num_nodes = 0;
		let status = unsafe { f(session_ptr.as_ptr(), &mut num_nodes) };
		unsafe { Error::result_from_status(status) }?;
		Ok(num_nodes)
	}

	fn extract_io_name(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, usize, *mut ort_sys::OrtAllocator, *mut *mut c_char) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>,
		allocator: &Allocator,
		i: usize
	) -> Result<String> {
		let mut name_ptr: *mut c_char = ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, allocator.ptr().cast_mut(), &mut name_ptr) };
		unsafe { Error::result_from_status(status) }?;
		if name_ptr.is_null() {
			crate::util::cold();
			return Err(crate::Error::new("expected `name_ptr` to not be null"));
		}

		unsafe { AllocatedString::from_ptr(name_ptr, allocator) }.map(|x| x.to_string())
	}

	fn extract_io(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, usize, *mut *mut ort_sys::OrtTypeInfo) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>,
		i: usize
	) -> Result<ValueType> {
		let mut typeinfo_ptr: *mut ort_sys::OrtTypeInfo = ptr::null_mut();

		let status = unsafe { f(session_ptr.as_ptr(), i, &mut typeinfo_ptr) };
		unsafe { Error::result_from_status(status) }?;
		let Some(typeinfo_ptr) = NonNull::new(typeinfo_ptr) else {
			crate::util::cold();
			return Err(crate::Error::new("expected `typeinfo_ptr` to not be null"));
		};
		Ok(unsafe { ValueType::from_type_info(typeinfo_ptr) })
	}

	pub(super) fn extract_input(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Outlet> {
		let name = extract_io_name(ortsys![SessionGetInputName], session_ptr, allocator, i)?;
		let dtype = extract_io(ortsys![SessionGetInputTypeInfo], session_ptr, i)?;
		Ok(Outlet::new(name, dtype))
	}

	pub(super) fn extract_output(session_ptr: NonNull<ort_sys::OrtSession>, allocator: &Allocator, i: usize) -> Result<Outlet> {
		let name = extract_io_name(ortsys![SessionGetOutputName], session_ptr, allocator, i)?;
		let dtype = extract_io(ortsys![SessionGetOutputTypeInfo], session_ptr, i)?;
		Ok(Outlet::new(name, dtype))
	}

	#[cfg(feature = "api-23")]
	pub(super) fn extract_io_memory_info(
		f: unsafe extern "system" fn(*const ort_sys::OrtSession, *mut *const ort_sys::OrtMemoryInfo, usize) -> ort_sys::OrtStatusPtr,
		session_ptr: NonNull<ort_sys::OrtSession>,
		outlets: &[Outlet]
	) -> Result<MiniMap<String, MemoryInfo<'static>>> {
		let mut memory_info_ptrs = vec![ptr::null(); outlets.len()];
		let status = unsafe { f(session_ptr.as_ptr(), memory_info_ptrs.as_mut_ptr(), memory_info_ptrs.len()) };
		unsafe { Error::result_from_status(status) }?;
		let mut memory_infos = MiniMap::new();
		for (ptr, outlet) in memory_info_ptrs.iter().zip(outlets.iter()) {
			let Some(ptr) = NonNull::new(*ptr as *mut ort_sys::OrtMemoryInfo) else {
				crate::util::cold();
				return Err(crate::Error::new("expected in/output memory info to not be null"));
			};
			memory_infos.insert(outlet.name().to_string(), MemoryInfo::from_raw(ptr, false));
		}
		Ok(memory_infos)
	}
}
