use alloc::{ffi::CString, sync::Arc};
use core::{
	ffi::{c_char, c_void},
	future::Future,
	marker::PhantomData,
	pin::Pin,
	ptr::NonNull,
	task::{Context, Poll, Waker}
};

use smallvec::SmallVec;

use crate::{
	Error,
	error::Result,
	session::{SessionOutputs, SharedSessionInner, UntypedRunOptions},
	util::{Mutex, STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS},
	value::{DynValue, Value, ValueInner}
};

struct InferenceFutState<'r> {
	value: Option<Result<SessionOutputs<'r>>>,
	waker: Option<Waker>
}

pub(crate) struct InferenceFutInner<'r> {
	// The value and waker share one lock so the callback can't complete between `poll` checking the value and storing
	// its waker.
	state: Mutex<InferenceFutState<'r>>,
	run_options: Arc<UntypedRunOptions>
}

impl<'r> InferenceFutInner<'r> {
	pub(crate) fn new(run_options: Arc<UntypedRunOptions>) -> Self {
		InferenceFutInner {
			state: Mutex::new(InferenceFutState { value: None, waker: None }),
			run_options
		}
	}

	pub(crate) fn complete(&self, value: Result<SessionOutputs<'r>>) {
		let waker = {
			let mut state = self.state.lock();
			state.value = Some(value);
			state.waker.take()
		};
		if let Some(waker) = waker {
			waker.wake();
		}
	}
}

unsafe impl Send for InferenceFutInner<'_> {}
unsafe impl Sync for InferenceFutInner<'_> {}

pub struct InferenceFut<'r, 'v> {
	inner: Arc<InferenceFutInner<'r>>,
	did_receive: bool,
	_inputs: PhantomData<&'v ()>
}

unsafe impl Send for InferenceFut<'_, '_> {}

impl<'r> InferenceFut<'r, '_> {
	pub(crate) fn new(inner: Arc<InferenceFutInner<'r>>) -> Self {
		Self {
			inner,
			did_receive: false,
			_inputs: PhantomData
		}
	}
}

impl<'r> Future for InferenceFut<'r, '_> {
	type Output = Result<SessionOutputs<'r>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let this = Pin::into_inner(self);

		let mut state = this.inner.state.lock();
		if let Some(v) = state.value.take() {
			this.did_receive = true;
			return Poll::Ready(v);
		}

		state.waker = Some(cx.waker().clone());
		Poll::Pending
	}
}

impl Drop for InferenceFut<'_, '_> {
	fn drop(&mut self) {
		if !self.did_receive {
			let _ = self.inner.run_options.terminate();
			self.inner.state.lock().waker = None;
		}
	}
}

pub(crate) struct AsyncInferenceContext<'r, 's> {
	pub(crate) inner: Arc<InferenceFutInner<'r>>,
	pub(crate) input_ort_values: SmallVec<[*const ort_sys::OrtValue; STACK_SESSION_INPUTS]>,
	pub(crate) _input_inner_holders: SmallVec<[Arc<ValueInner>; STACK_SESSION_INPUTS]>,
	pub(crate) input_name_ptrs: SmallVec<[*const c_char; STACK_SESSION_INPUTS]>,
	pub(crate) output_name_ptrs: SmallVec<[*const c_char; STACK_SESSION_OUTPUTS]>,
	pub(crate) session_inner: &'s Arc<SharedSessionInner>,
	pub(crate) output_names: SmallVec<[&'r str; STACK_SESSION_OUTPUTS]>,
	pub(crate) output_value_ptrs: SmallVec<[*mut ort_sys::OrtValue; STACK_SESSION_OUTPUTS]>,
	/// Preallocated outputs, which already own their `OrtValue`s.
	pub(crate) output_values: SmallVec<[Option<DynValue>; STACK_SESSION_OUTPUTS]>
}

impl AsyncInferenceContext<'_, '_> {
	/// Frees the input & output names, which were leaked from `CString`s by `run_inner_async`.
	pub(crate) fn free_name_ptrs(&self) {
		for &p in self.input_name_ptrs.iter().chain(self.output_name_ptrs.iter()) {
			drop(unsafe { CString::from_raw(p.cast_mut()) });
		}
	}
}

pub(crate) extern "system" fn async_callback(user_data: *mut c_void, _: *mut *mut ort_sys::OrtValue, _: usize, status: ort_sys::OrtStatusPtr) {
	let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_, '_>>()) };

	ctx.free_name_ptrs();

	crate::logging::drop!(AsyncInferenceContext, user_data);

	if let Err(e) = unsafe { Error::result_from_status(status) } {
		ctx.inner.complete(Err(e));
		return;
	}

	let outputs = ctx
		.output_value_ptrs
		.into_iter()
		.zip(ctx.output_values)
		.map(|(tensor_ptr, value)| match value {
			Some(value) => value,
			None => unsafe {
				Value::from_ptr(
					NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"),
					Some(&ctx.session_inner.allocator)
				)
			}
		})
		.collect();

	ctx.inner.complete(Ok(SessionOutputs::new(ctx.output_names, outputs)));
}
