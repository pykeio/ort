#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
use alloc::sync::Arc;
use alloc::{ffi::CString, format};
use core::{
	ffi::c_char,
	iter,
	ptr::{self, NonNull},
	slice
};

use smallvec::SmallVec;

#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
use super::r#async::{AsyncInferenceContext, InferenceFut, InferenceFutInner, async_callback};
use super::{
	Session, SessionInputValue, SessionInputs, SessionOutputs,
	run_options::{RunOptions, SelectedOutputMarker, UntypedRunOptions}
};
#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
use super::{io_binding::IoBinding, run_options::NoSelectedOutputs};
use crate::{
	AsPointer,
	error::{Error, ErrorCode, Result},
	ortsys,
	util::{STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS, with_cstr_ptr_array},
	value::DynValue
};

super::mut_methods! {
	/// Run input data through the ONNX graph, performing inference.
	///
	/// See [`crate::inputs!`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{RunOptions, Session}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn run['s, 'i, 'v: 'i, const N: usize](&'s mut self, input_values: impl Into<SessionInputs<'i, 'v, N>>) -> Result<SessionOutputs<'s>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), None)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), None)
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
	/// # use ort::{session::{Session, RunOptions}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> {
	/// # 	let env = ort::test_util::test_env().clone();
	/// # 	let mut session = Session::builder(&env)?.commit_from_file("tests/data/upsample.onnx")?;
	/// # 	let input = Value::from_array(ndarray::Array4::<f32>::zeros((1, 64, 64, 3)))?;
	/// let run_options = Arc::new(RunOptions::new()?);
	///
	/// let run_options_ = Arc::clone(&run_options);
	/// std::thread::spawn(move || {
	/// 	let _ = run_options_.terminate();
	/// });
	///
	/// let res = session.run_with_options(ort::inputs![&input], &*run_options);
	/// // upon termination, the session will return an `Error::SessionRun` error.
	/// assert_eq!(
	/// 	&res.unwrap_err().to_string(),
	/// 	"Failed to run inference on model: Exiting due to terminate flag being set to true."
	/// );
	/// # 	Ok(())
	/// # }
	/// ```
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn run_with_options['r, 's: 'r, 'i, 'v: 'i, O: SelectedOutputMarker, const N: usize](
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'v, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<SessionOutputs<'r>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), Some(&run_options.inner))
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), Some(&run_options.inner))
			}
		}
	}

	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn run_binding['b, 's: 'b](&'s mut self, binding: &'b IoBinding) -> Result<SessionOutputs<'b>> {
		self.run_binding_inner(binding, None)
	}

	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	pub fn run_binding_with_options['r, 'b, 's: 'b](
		&'s mut self,
		binding: &'b IoBinding,
		run_options: &'r RunOptions<NoSelectedOutputs>
	) -> Result<SessionOutputs<'b>> {
		self.run_binding_inner(binding, Some(run_options))
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
	/// # use ort::{session::{Session, RunOptions}, value::{Value, ValueType, Tensor, TensorElementType}};
	/// # fn main() -> ort::Result<()> { tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
	/// # let env = ort::test_util::test_env().clone();
	/// let mut session = Session::builder(&env)?.with_intra_threads(2)?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let options = RunOptions::new()?;
	/// let outputs = session.run_async(ort::inputs![Tensor::from_array(input)?], &options)?.await?;
	/// # 	Ok(())
	/// # }) }
	/// ```
	///
	/// ## Cancellation safety
	/// This function is *cancel-safe*, though note that ONNX Runtime will not terminate execution immediately (only
	/// between kernel launches). It is for this reason that **only owned tensors** are accepted.
	#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
	#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
	pub fn run_async['r, 's: 'r, 'i, O: SelectedOutputMarker, const N: usize](
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'static, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<InferenceFut<'r>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), &run_options.inner)
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), &run_options.inner)
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner_async(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), &run_options.inner)
			}
		}
	}
}

super::mut_methods! {
	/// Run input data through the ONNX graph, performing inference.
	///
	/// See [`crate::inputs!`] for a convenient macro which will help you create your session inputs from `ndarray`s or
	/// other data. You can also provide a `Vec`, array, or `HashMap` of [`Value`]s if you create your inputs
	/// dynamically.
	///
	/// ```
	/// # use std::sync::Arc;
	/// # use ort::{session::{Session, RunOptions}, value::{Value, ValueType, TensorRef, TensorElementType}};
	/// # fn main() -> ort::Result<()> { tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
	/// # let env = ort::test_util::test_env().clone();
	/// let mut session = Session::builder(&env)?.with_intra_threads(2)?.commit_from_file("tests/data/upsample.onnx")?;
	/// let input = ndarray::Array4::<f32>::zeros((1, 64, 64, 3));
	/// let options = RunOptions::new()?;
	/// let outputs = session.run_async(ort::inputs![TensorRef::from_array_view(&input)?], &options)?.await?;
	/// # 	Ok(())
	/// # }) }
	/// ```
	#[cfg(all(target_family = "wasm", target_os = "unknown"))]
	pub async fn run_async['r, 's: 'r, 'i, O: SelectedOutputMarker, const N: usize](
		&'s mut self,
		input_values: impl Into<SessionInputs<'i, 'static, N>>,
		run_options: &'r RunOptions<O>
	) -> Result<SessionOutputs<'r>> {
		match input_values.into() {
			SessionInputs::ValueSlice(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), &run_options.inner).await
			}
			SessionInputs::ValueArray(input_values) => {
				self.run_inner_async(self.inputs.iter().map(|input| input.name()).collect(), input_values.iter().collect(), &run_options.inner).await
			}
			SessionInputs::ValueMap(input_values) => {
				self.run_inner_async(input_values.iter().map(|(k, _)| k.as_ref()).collect(), input_values.iter().map(|(_, v)| v).collect(), &run_options.inner).await
			}
		}
	}
}

impl Session {
	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	fn run_inner<'i, 'r, 's: 'r, 'v: 'i>(
		// Yes, the internal run functions take `&self`. In a perfect world, the public functions would take `&self` too. But we don't live in a perfect world,
		// and ONNX Runtime's own `Run` is **not thread safe** (despite Microsoft saying it should be years ago!). You may modify those functions to take
		// `&self` and run your models from a bunch of threads without a care in the world, and everything may *seem* fine, but that's the beauty of undefined
		// behavior - it's undefined! You have no idea what could be going wrong behind the scenes. At least EPs like CUDA have the decency to outright crash
		// the process most of the time. The internal run functions will stay `&self` on the off chance that MS stumbles their way into asking Copilot to fix
		// the thread safety of `Run`; if, by some miracle, it actually manages to do that, we can then make the other functions `&self` with minimal changes.
		//
		// If you decide to change the run functions to take `&self` before then, you forfeit your rights to complain to me when your app crashes or data gets
		// corrupted or your machine blows up. If you submit that change in a PR, you give me the right to repeatedly hit you over the head with a cartoon
		// caveman's club. This comment is legally binding.
		&'s self,
		input_names: SmallVec<[&str; STACK_SESSION_INPUTS]>,
		input_values: SmallVec<[&'i SessionInputValue<'v>; STACK_SESSION_INPUTS]>,
		run_options: Option<&'r UntypedRunOptions>
	) -> Result<SessionOutputs<'r>> {
		if input_values.len() > input_names.len() {
			// If we provide more inputs than the model expects with `ort::inputs![a, b, c]`, then we get an
			// `input_names` shorter than `inputs`. ONNX Runtime will attempt to look up the name of all
			// inputs before doing any checks, thus going out of bounds of `input_names` and triggering a
			// segfault, so we check that condition here. This will never trip for `ValueMap` inputs since
			// the number of names & values is always equal as it's a vec of tuples.
			return Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("{} inputs were provided, but the model only accepts {}.", input_values.len(), input_names.len())
			));
		}

		let (output_names, mut output_tensors) = match run_options {
			Some(r) => r.outputs.resolve_outputs(&self.outputs),
			None => (self.outputs.iter().map(|o| o.name()).collect(), iter::repeat_with(|| None).take(self.outputs.len()).collect())
		};
		let mut output_value_ptrs: SmallVec<[*mut ort_sys::OrtValue; STACK_SESSION_OUTPUTS]> = output_tensors
			.iter_mut()
			.map(|c| match c {
				Some(v) => v.ptr_mut(),
				None => ptr::null_mut()
			})
			.collect();
		let input_value_ptrs: SmallVec<[*const ort_sys::OrtValue; STACK_SESSION_INPUTS]> = input_values.iter().map(|c| c.ptr()).collect();

		let run_options_ptr = if let Some(run_options) = &run_options { run_options.ptr.as_ptr() } else { ptr::null() };

		with_cstr_ptr_array(&input_names, &mut |input_name_ptrs| {
			with_cstr_ptr_array(&output_names, &mut |output_name_ptrs| {
				ortsys![
					unsafe Run(
						self.inner.session_ptr.as_ptr(),
						run_options_ptr,
						input_name_ptrs.as_ptr(),
						input_value_ptrs.as_ptr(),
						input_value_ptrs.len(),
						output_name_ptrs.as_ptr(),
						output_name_ptrs.len(),
						output_value_ptrs.as_mut_ptr()
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
					DynValue::from_session_output_ptr(
						NonNull::new(output_value_ptrs[i]).expect("OrtValue ptr returned from session Run should not be null"),
						&self.inner
					)
				}
			})
			.collect();

		Ok(SessionOutputs::new(output_names, outputs))
	}

	#[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
	fn run_binding_inner<'r, 'b, 's: 'b>(
		&'s self, // `RunWithBinding` is also not thread safe.
		binding: &'b IoBinding,
		run_options: Option<&'r RunOptions<NoSelectedOutputs>>
	) -> Result<SessionOutputs<'b>> {
		use crate::util::run_on_drop;

		let run_options_ptr = if let Some(run_options) = run_options { run_options.ptr() } else { ptr::null() };
		ortsys![unsafe RunWithBinding(self.inner.ptr().cast_mut(), run_options_ptr, binding.ptr())?];

		let mut count = binding.output_values.len();
		if count > 0 {
			let mut output_values_ptr: *mut *mut ort_sys::OrtValue = ptr::null_mut();
			ortsys![unsafe GetBoundOutputValues(binding.ptr(), self.inner.allocator.ptr().cast_mut(), &mut output_values_ptr, &mut count)?; nonNull(output_values_ptr)];

			let _guard = run_on_drop(|| unsafe {
				self.inner.allocator.free(output_values_ptr.as_ptr());
			});
			let output_values = unsafe { slice::from_raw_parts(output_values_ptr.as_ptr(), count) }
				.iter()
				.map(|ptr| unsafe {
					DynValue::from_session_output_ptr(
						NonNull::new(*ptr).expect("OrtValue ptrs returned by GetBoundOutputValues should not be null"),
						&self.inner
					)
				})
				.collect();

			Ok(SessionOutputs::new(binding.output_values.iter().map(|(k, _)| k.as_str()).collect(), output_values))
		} else {
			Ok(SessionOutputs::new_empty())
		}
	}

	#[cfg(all(feature = "std", not(all(target_family = "wasm", target_os = "unknown"))))]
	fn run_inner_async<'i, 'r, 's: 'r>(
		&'s self, // Yes, `RunAsync` is also not thread safe. That doesn't make a whole lot of sense.
		input_names: SmallVec<[&str; STACK_SESSION_INPUTS]>,
		input_values: SmallVec<[&SessionInputValue<'static>; STACK_SESSION_INPUTS]>,
		run_options: &'r Arc<UntypedRunOptions>
	) -> Result<InferenceFut<'r>> {
		// See the comment in `run_inner`; ONNX Runtime would read past the end of `input_name_ptrs`.

		if input_values.len() > input_names.len() {
			return Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("{} inputs were provided, but the model only accepts {}.", input_values.len(), input_names.len())
			));
		}

		// Convert every name before leaking any of them, so an invalid name doesn't leak the ones before it.
		let input_name_ptrs = input_names
			.into_iter()
			.map(|name| CString::new(name.as_bytes()))
			.collect::<Result<SmallVec<[CString; STACK_SESSION_INPUTS]>, _>>()?
			.into_iter()
			.map(|s| s.into_raw().cast_const())
			.collect::<SmallVec<[*const c_char; STACK_SESSION_INPUTS]>>();

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

		let async_inner = Arc::new(InferenceFutInner::new(Arc::clone(run_options)));

		// Avoid creating AsyncInferenceContext on the stack since it is a very large struct. Instead, create it on the
		// heap and then fill it with values.
		let mut ctx = Box::<AsyncInferenceContext>::new_uninit();
		unsafe {
			use core::ptr::write;

			let ctx = ctx.as_mut_ptr();
			write(&raw mut (*ctx).inner, Arc::clone(&async_inner));
			// everything allocated within `run_inner_async` needs to be kept alive until we are certain inference has
			// completed and ONNX Runtime no longer needs the data - i.e. when `async_callback` is called.
			// `async_callback` will free all of this data just like we do in `run_inner`
			write(&raw mut (*ctx).input_ort_values, input_ort_values);
			write(&raw mut (*ctx)._input_inner_holders, input_inner_holders);
			write(&raw mut (*ctx).input_name_ptrs, input_name_ptrs);
			write(&raw mut (*ctx).output_name_ptrs, output_name_ptrs);
			write(&raw mut (*ctx).output_names, output_names);
			write(&raw mut (*ctx).output_value_ptrs, output_tensor_ptrs);
			write(&raw mut (*ctx).output_values, output_tensors);
			write(&raw mut (*ctx).session_inner, Arc::clone(&self.inner));
		};
		let ctx = Box::leak(unsafe { ctx.assume_init() });
		crate::logging::create!(AsyncInferenceContext, ctx);

		let res = ortsys![@ort:
			unsafe RunAsync(
				self.inner.session_ptr.as_ptr(),
				run_options.ptr.as_ptr(),
				ctx.input_name_ptrs.as_ptr(),
				ctx.input_ort_values.as_ptr(),
				ctx.input_ort_values.len(),
				ctx.output_name_ptrs.as_ptr(),
				ctx.output_name_ptrs.len(),
				ctx.output_value_ptrs.as_mut_ptr(),
				Some(async_callback),
				ctx as *mut _ as *mut ort_sys::c_void
			) as Result
		];
		if let Err(e) = res {
			// The callback is never called if `RunAsync` fails, so we have to free the context ourselves.
			let ctx = unsafe { Box::from_raw(ctx as *mut AsyncInferenceContext<'_>) };
			crate::logging::drop!(AsyncInferenceContext, &*ctx);
			ctx.free_name_ptrs();
			return Err(e);
		}

		Ok(InferenceFut::new(async_inner))
	}

	#[cfg(all(target_family = "wasm", target_os = "unknown"))]
	async fn run_inner_async<'i, 'r, 's: 'r>(
		&'s self,
		input_names: SmallVec<[&str; STACK_SESSION_INPUTS]>,
		input_values: SmallVec<[&SessionInputValue<'static>; STACK_SESSION_INPUTS]>,
		run_options: &'r UntypedRunOptions
	) -> Result<SessionOutputs<'r>> {
		if input_values.len() > input_names.len() {
			return Err(Error::new_with_code(
				ErrorCode::InvalidArgument,
				format!("{} inputs were provided, but the model only accepts {}.", input_values.len(), input_names.len())
			));
		}

		let (output_names, mut output_tensors) = run_options.outputs.resolve_outputs(&self.outputs);
		let mut output_value_ptrs: SmallVec<[*mut ort_sys::OrtValue; STACK_SESSION_OUTPUTS]> = output_tensors
			.iter_mut()
			.map(|c| match c {
				Some(v) => v.ptr_mut(),
				None => ptr::null_mut()
			})
			.collect();
		let input_value_ptrs: SmallVec<[*const ort_sys::OrtValue; STACK_SESSION_INPUTS]> = input_values.iter().map(|c| c.ptr()).collect();

		let status = ortsys![
			unsafe RunAsync(
				self.inner.session_ptr.as_ptr(),
				run_options.ptr.as_ptr(),
				&input_names,
				&input_value_ptrs,
				&output_names,
				&mut output_value_ptrs
			)
		]
		.await;
		unsafe { Error::result_from_status(status) }?;

		let outputs = output_tensors
			.into_iter()
			.enumerate()
			.map(|(i, v)| match v {
				Some(value) => value,
				None => unsafe {
					DynValue::from_session_output_ptr(
						NonNull::new(output_value_ptrs[i]).expect("OrtValue ptr returned from session Run should not be null"),
						&self.inner
					)
				}
			})
			.collect();

		Ok(SessionOutputs::new(output_names, outputs))
	}
}
