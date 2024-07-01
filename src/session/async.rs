use std::{
	cell::UnsafeCell,
	ffi::{c_char, CString},
	future::Future,
	ops::Deref,
	pin::Pin,
	ptr::NonNull,
	sync::{Arc, Mutex},
	task::{Context, Poll, Waker}
};

use ort_sys::{c_void, OrtStatus};

use crate::{error::assert_non_null_pointer, Error, Result, RunOptions, SessionInputValue, SessionOutputs, SharedSessionInner, Value};

#[derive(Debug)]
pub(crate) struct InferenceFutInner<'r, 's> {
	value: UnsafeCell<Option<Result<SessionOutputs<'r, 's>>>>,
	waker: Mutex<Option<Waker>>
}

impl<'r, 's> InferenceFutInner<'r, 's> {
	pub(crate) fn new() -> Self {
		InferenceFutInner {
			waker: Mutex::new(None),
			value: UnsafeCell::new(None)
		}
	}

	pub(crate) fn try_take(&self) -> Option<Result<SessionOutputs<'r, 's>>> {
		unsafe { &mut *self.value.get() }.take()
	}

	pub(crate) fn emplace_value(&self, value: Result<SessionOutputs<'r, 's>>) {
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

unsafe impl<'r, 's> Send for InferenceFutInner<'r, 's> {}
unsafe impl<'r, 's> Sync for InferenceFutInner<'r, 's> {}

pub enum RunOptionsRef<'r> {
	Arc(Arc<RunOptions>),
	Ref(&'r RunOptions)
}

impl<'r> From<&Arc<RunOptions>> for RunOptionsRef<'r> {
	fn from(value: &Arc<RunOptions>) -> Self {
		Self::Arc(Arc::clone(value))
	}
}

impl<'r> From<&'r RunOptions> for RunOptionsRef<'r> {
	fn from(value: &'r RunOptions) -> Self {
		Self::Ref(value)
	}
}

impl<'r> Deref for RunOptionsRef<'r> {
	type Target = RunOptions;

	fn deref(&self) -> &Self::Target {
		match self {
			Self::Arc(r) => r,
			Self::Ref(r) => r
		}
	}
}

pub struct InferenceFut<'s, 'r> {
	inner: Arc<InferenceFutInner<'r, 's>>,
	run_options: RunOptionsRef<'r>,
	did_receive: bool
}

impl<'s, 'r> InferenceFut<'s, 'r> {
	pub(crate) fn new(inner: Arc<InferenceFutInner<'r, 's>>, run_options: RunOptionsRef<'r>) -> Self {
		Self {
			inner,
			run_options,
			did_receive: false
		}
	}
}

impl<'s, 'r> Future for InferenceFut<'s, 'r> {
	type Output = Result<SessionOutputs<'r, 's>>;

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

impl<'s, 'r> Drop for InferenceFut<'s, 'r> {
	fn drop(&mut self) {
		if !self.did_receive {
			let _ = self.run_options.terminate();
			self.inner.set_waker(None);
		}
	}
}

pub(crate) struct AsyncInferenceContext<'r, 's> {
	pub(crate) inner: Arc<InferenceFutInner<'r, 's>>,
	pub(crate) _input_values: Vec<SessionInputValue<'s>>,
	pub(crate) input_ort_values: Vec<*const ort_sys::OrtValue>,
	pub(crate) input_name_ptrs: Vec<*const c_char>,
	pub(crate) output_name_ptrs: Vec<*const c_char>,
	pub(crate) session_inner: &'s Arc<SharedSessionInner>,
	pub(crate) output_names: Vec<&'s str>,
	pub(crate) output_value_ptrs: Vec<*mut ort_sys::OrtValue>
}

crate::extern_system_fn! {
	pub(crate) fn async_callback(user_data: *mut c_void, _: *mut *mut ort_sys::OrtValue, _: ort_sys::size_t, status: *mut OrtStatus) {
		let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_, '_>>()) };

		// Reconvert name ptrs to CString so drop impl is called and memory is freed
		drop(
			ctx.input_name_ptrs
				.into_iter()
				.chain(ctx.output_name_ptrs)
				.map(|p| {
					assert_non_null_pointer(p, "c_char for CString")?;
					unsafe { Ok(CString::from_raw(p.cast_mut().cast())) }
				})
				.collect::<Result<Vec<_>>>()
				.expect("Input name should not be null")
		);

		if let Err(e) = crate::error::status_to_result(status) {
			ctx.inner.emplace_value(Err(Error::SessionRun(e)));
			ctx.inner.wake();
		}

		let outputs: Vec<Value> = ctx
			.output_value_ptrs
			.into_iter()
			.map(|tensor_ptr| unsafe {
				Value::from_ptr(NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"), Some(Arc::clone(ctx.session_inner)))
			})
			.collect();

		ctx.inner.emplace_value(Ok(SessionOutputs::new(ctx.output_names.into_iter(), outputs)));
		ctx.inner.wake();
	}
}
