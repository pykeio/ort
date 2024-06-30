use std::{
	cell::UnsafeCell,
	ffi::{c_char, CString},
	future::Future,
	mem::MaybeUninit,
	ops::Deref,
	pin::Pin,
	ptr::NonNull,
	sync::{
		atomic::{AtomicUsize, Ordering},
		Arc, Mutex
	},
	task::{Context, Poll, Waker}
};

use ort_sys::{c_void, OrtStatus};

use crate::{error::assert_non_null_pointer, Error, Result, RunOptions, SessionInputValue, SessionOutputs, SharedSessionInner, Value};

pub(crate) enum InnerValue<T> {
	Present(T),
	Pending,
	Closed
}

const VALUE_PRESENT: usize = 1 << 0;
const CHANNEL_CLOSED: usize = 1 << 1;

#[derive(Debug)]
pub(crate) struct InferenceFutInner<'s> {
	presence: AtomicUsize,
	value: UnsafeCell<MaybeUninit<Result<SessionOutputs<'s>>>>,
	waker: Mutex<Option<Waker>>
}

impl<'s> InferenceFutInner<'s> {
	pub(crate) fn new() -> Self {
		InferenceFutInner {
			presence: AtomicUsize::new(0),
			waker: Mutex::new(None),
			value: UnsafeCell::new(MaybeUninit::uninit())
		}
	}

	pub(crate) fn try_take(&self) -> InnerValue<Result<SessionOutputs<'s>>> {
		let state_snapshot = self.presence.fetch_and(!VALUE_PRESENT, Ordering::Acquire);
		if state_snapshot & VALUE_PRESENT == 0 {
			if self.presence.load(Ordering::Acquire) & CHANNEL_CLOSED != 0 {
				InnerValue::Closed
			} else {
				InnerValue::Pending
			}
		} else {
			InnerValue::Present(unsafe { (*self.value.get()).assume_init_read() })
		}
	}

	pub(crate) fn emplace_value(&self, value: Result<SessionOutputs<'s>>) {
		unsafe { (*self.value.get()).write(value) };
		self.presence.fetch_or(VALUE_PRESENT, Ordering::Release);
	}

	pub(crate) fn set_waker(&self, waker: Option<&Waker>) {
		*self.waker.lock().expect("Poisoned waker mutex") = waker.map(|c| c.to_owned());
	}

	pub(crate) fn wake(&self) {
		if let Some(waker) = self.waker.lock().expect("Poisoned waker mutex").take() {
			waker.wake();
		}
	}

	pub(crate) fn close(&self) -> bool {
		self.presence.fetch_or(CHANNEL_CLOSED, Ordering::Acquire) & CHANNEL_CLOSED == 0
	}
}

impl<'s> Drop for InferenceFutInner<'s> {
	fn drop(&mut self) {
		if self.presence.load(Ordering::Acquire) & VALUE_PRESENT != 0 {
			unsafe { (*self.value.get()).assume_init_drop() };
		}
	}
}

unsafe impl<'s> Send for InferenceFutInner<'s> {}
unsafe impl<'s> Sync for InferenceFutInner<'s> {}

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
	inner: Arc<InferenceFutInner<'s>>,
	run_options: RunOptionsRef<'r>,
	did_receive: bool
}

impl<'s, 'r> InferenceFut<'s, 'r> {
	pub(crate) fn new(inner: Arc<InferenceFutInner<'s>>, run_options: RunOptionsRef<'r>) -> Self {
		Self {
			inner,
			run_options,
			did_receive: false
		}
	}
}

impl<'s, 'r> Future for InferenceFut<'s, 'r> {
	type Output = Result<SessionOutputs<'s>>;

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let this = Pin::into_inner(self);

		match this.inner.try_take() {
			InnerValue::Present(v) => {
				this.did_receive = true;
				return Poll::Ready(v);
			}
			InnerValue::Pending => {}
			InnerValue::Closed => panic!()
		};

		this.inner.set_waker(Some(cx.waker()));

		Poll::Pending
	}
}

impl<'s, 'r> Drop for InferenceFut<'s, 'r> {
	fn drop(&mut self) {
		if !self.did_receive && self.inner.close() {
			let _ = self.run_options.terminate();
			self.inner.set_waker(None);
		}
	}
}

pub(crate) struct AsyncInferenceContext<'s> {
	pub(crate) inner: Arc<InferenceFutInner<'s>>,
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
		let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_>>()) };

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
