use alloc::{ffi::CString, sync::Arc};
use core::{
	cell::UnsafeCell,
	ffi::{c_char, c_void},
	future::Future,
	marker::PhantomData,
	pin::Pin,
	ptr::NonNull,
	task::{Context, Poll, Waker}
};
use std::sync::Mutex;

use smallvec::SmallVec;

use crate::{
	error::Result,
	session::{SessionOutputs, SharedSessionInner, run_options::UntypedRunOptions},
	util::{STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS},
	value::{Value, ValueInner}
};

#[derive(Debug)]
pub(crate) struct InferenceFutInner<'r> {
	value: UnsafeCell<Option<Result<SessionOutputs<'r>>>>,
	waker: Mutex<Option<Waker>>
}

impl<'r> InferenceFutInner<'r> {
	pub(crate) fn new() -> Self {
		InferenceFutInner {
			waker: Mutex::new(None),
			value: UnsafeCell::new(None)
		}
	}

	pub(crate) fn try_take(&self) -> Option<Result<SessionOutputs<'r>>> {
		unsafe { &mut *self.value.get() }.take()
	}

	pub(crate) fn emplace_value(&self, value: Result<SessionOutputs<'r>>) {
		unsafe { &mut *self.value.get() }.replace(value);
	}

	pub(crate) fn set_waker(&self, waker: Option<&Waker>) {
		*self.waker.lock().expect("Poisoned waker mutex") = waker.map(|c| c.to_owned());
	}

	pub(crate) fn wake(&self) {
		if let Some(waker) = self.waker.lock().expect("Poisoned waker mutex").take() {
			waker.wake();
		}
	}
}

unsafe impl Send for InferenceFutInner<'_> {}
unsafe impl Sync for InferenceFutInner<'_> {}

pub struct InferenceFut<'r, 'v> {
	inner: Arc<InferenceFutInner<'r>>,
	run_options: &'r UntypedRunOptions,
	did_receive: bool,
	_inputs: PhantomData<&'v ()>
}

unsafe impl Send for InferenceFut<'_, '_> {}

impl<'r> InferenceFut<'r, '_> {
	pub(crate) fn new(inner: Arc<InferenceFutInner<'r>>, run_options: &'r UntypedRunOptions) -> Self {
		Self {
			inner,
			run_options,
			did_receive: false,
			_inputs: PhantomData
		}
	}
}

impl<'r> Future for InferenceFut<'r, '_> {
	type Output = Result<SessionOutputs<'r>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let this = Pin::into_inner(self);

		if let Some(v) = this.inner.try_take() {
			this.did_receive = true;
			return Poll::Ready(v);
		}

		this.inner.set_waker(Some(cx.waker()));
		Poll::Pending
	}
}

impl Drop for InferenceFut<'_, '_> {
	fn drop(&mut self) {
		if !self.did_receive {
			let _ = self.run_options.terminate();
			self.inner.set_waker(None);
		}
	}
}

pub(crate) struct AsyncInferenceContext<'r, 's> {
	pub(crate) inner: Arc<InferenceFutInner<'r>>,
	pub(crate) input_ort_values: SmallVec<*const ort_sys::OrtValue, { STACK_SESSION_INPUTS }>,
	pub(crate) _input_inner_holders: SmallVec<Arc<ValueInner>, { STACK_SESSION_INPUTS }>,
	pub(crate) input_name_ptrs: SmallVec<*const c_char, { STACK_SESSION_INPUTS }>,
	pub(crate) output_name_ptrs: SmallVec<*const c_char, { STACK_SESSION_OUTPUTS }>,
	pub(crate) session_inner: &'s Arc<SharedSessionInner>,
	pub(crate) output_names: SmallVec<&'r str, { STACK_SESSION_OUTPUTS }>,
	pub(crate) output_value_ptrs: SmallVec<*mut ort_sys::OrtValue, { STACK_SESSION_OUTPUTS }>
}

pub(crate) extern "system" fn async_callback(user_data: *mut c_void, _: *mut *mut ort_sys::OrtValue, _: usize, status: ort_sys::OrtStatusPtr) {
	let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_, '_>>()) };

	// Reconvert name ptrs to CString so drop impl is called and memory is freed
	for p in ctx.input_name_ptrs {
		drop(unsafe { CString::from_raw(p.cast_mut().cast()) });
	}

	if let Err(e) = unsafe { crate::error::status_to_result(status) } {
		ctx.inner.emplace_value(Err(e));
		ctx.inner.wake();
		return;
	}

	let outputs = ctx
		.output_value_ptrs
		.into_iter()
		.map(|tensor_ptr| unsafe {
			Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), Some(Arc::clone(ctx.session_inner)))
		})
		.collect();

	ctx.inner.emplace_value(Ok(SessionOutputs::new(ctx.output_names, outputs)));
	ctx.inner.wake();
}
