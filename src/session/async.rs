use alloc::{ffi::CString, sync::Arc};
use core::{
	ffi::{c_char, c_void},
	future::Future,
	pin::Pin,
	ptr::NonNull,
	task::{Context, Poll, Waker}
};
use std::sync::Condvar;

use smallvec::SmallVec;

use crate::{
	Error,
	error::Result,
	session::{SessionOutputs, SharedSessionInner, UntypedRunOptions, run::OutputAliasGuard},
	util::{Mutex, STACK_SESSION_INPUTS, STACK_SESSION_OUTPUTS, run_on_drop},
	value::{DynValue, ValueInner}
};

#[derive(Debug, Default)]
pub(crate) struct AsyncRunGate {
	running: std::sync::Mutex<bool>,
	finished: Condvar
}

impl AsyncRunGate {
	fn wait_idle(&self) -> std::sync::MutexGuard<'_, bool> {
		let mut running = self.running.lock().unwrap_or_else(|e| e.into_inner());
		while *running {
			running = self.finished.wait(running).unwrap_or_else(|e| e.into_inner());
		}
		running
	}

	pub(crate) fn wait(&self) {
		drop(self.wait_idle());
	}

	pub(crate) fn begin(&self) {
		*self.wait_idle() = true;
	}

	pub(crate) fn end(&self) {
		*self.running.lock().unwrap_or_else(|e| e.into_inner()) = false;
		self.finished.notify_all();
	}
}

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

pub struct InferenceFut<'r> {
	inner: Arc<InferenceFutInner<'r>>,
	did_receive: bool
}

unsafe impl Send for InferenceFut<'_> {}

impl<'r> InferenceFut<'r> {
	pub(crate) fn new(inner: Arc<InferenceFutInner<'r>>) -> Self {
		Self { inner, did_receive: false }
	}
}

impl<'r> Future for InferenceFut<'r> {
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

impl Drop for InferenceFut<'_> {
	fn drop(&mut self) {
		if !self.did_receive {
			let _ = self.inner.run_options.terminate();
			self.inner.state.lock().waker = None;
		}
	}
}

pub(crate) struct AsyncInferenceContext<'r> {
	pub(crate) inner: Arc<InferenceFutInner<'r>>,
	pub(crate) input_ort_values: SmallVec<[*const ort_sys::OrtValue; STACK_SESSION_INPUTS]>,
	pub(crate) _input_inner_holders: SmallVec<[Arc<ValueInner>; STACK_SESSION_INPUTS]>,
	pub(crate) input_name_ptrs: SmallVec<[*const c_char; STACK_SESSION_INPUTS]>,
	pub(crate) output_name_ptrs: SmallVec<[*const c_char; STACK_SESSION_OUTPUTS]>,
	pub(crate) session_inner: Arc<SharedSessionInner>,
	pub(crate) output_names: SmallVec<[&'r str; STACK_SESSION_OUTPUTS]>,
	pub(crate) output_value_ptrs: SmallVec<[*mut ort_sys::OrtValue; STACK_SESSION_OUTPUTS]>,
	/// Preallocated outputs, which already own their `OrtValue`s.
	pub(crate) output_values: SmallVec<[Option<DynValue>; STACK_SESSION_OUTPUTS]>
}

impl AsyncInferenceContext<'_> {
	/// Frees the input & output names, which were leaked from `CString`s by `run_inner_async`.
	pub(crate) fn free_name_ptrs(&self) {
		for &p in self.input_name_ptrs.iter().chain(self.output_name_ptrs.iter()) {
			drop(unsafe { CString::from_raw(p.cast_mut()) });
		}
	}
}

pub(crate) extern "C" fn async_callback(user_data: *mut c_void, _: *mut *mut ort_sys::OrtValue, _: usize, status: ort_sys::OrtStatusPtr) {
	let ctx = unsafe { Box::from_raw(user_data.cast::<AsyncInferenceContext<'_>>()) };

	let session_inner = Arc::clone(&ctx.session_inner);
	let _gate = run_on_drop(move || session_inner.async_gate.end());

	ctx.free_name_ptrs();

	crate::logging::drop!(AsyncInferenceContext, user_data);

	if let Err(e) = unsafe { Error::result_from_status(status) } {
		ctx.inner.complete(Err(e));
		return;
	}

	let mut alias_guard = OutputAliasGuard::new();
	for value in ctx
		._input_inner_holders
		.iter()
		.map(|input| &**input)
		.chain(ctx.output_values.iter().flatten().map(|output| &*output.inner))
	{
		alias_guard.add(value);
	}

	let outputs: SmallVec<[(DynValue, bool); STACK_SESSION_OUTPUTS]> = ctx
		.output_value_ptrs
		.into_iter()
		.zip(ctx.output_values)
		.map(|(tensor_ptr, v)| match v {
			Some(value) => (value, true),
			None => (
				unsafe {
					DynValue::from_session_output_ptr(
						NonNull::new(tensor_ptr).expect("OrtValue ptr returned from session Run should not be null"),
						&ctx.session_inner
					)
				},
				false
			)
		})
		.collect();
	// split so we take ownership before doing anything fallible
	let outputs = outputs
		.into_iter()
		.map(|(value, preallocated)| if preallocated { Ok(value) } else { alias_guard.unalias(value) })
		.collect::<Result<_>>();

	let output_names = ctx.output_names;
	ctx.inner.complete(outputs.map(|outputs| SessionOutputs::new(output_names, outputs)));
}

#[cfg(test)]
mod tests {
	use alloc::sync::Arc;
	use core::{
		mem,
		sync::atomic::{AtomicUsize, Ordering},
		time::Duration
	};
	use std::thread;

	use crate::{
		Error, Result,
		memory::Allocator,
		operator::{BoxedKernel, ComputeContext, KernelContext, Operator, OperatorDomain, OperatorInput, OperatorOutput},
		session::{RunOptions, Session},
		test_util::test_env,
		value::{Tensor, TensorElementType}
	};

	#[derive(Default)]
	struct Activity {
		active: AtomicUsize,
		max_active: AtomicUsize
	}

	struct SlowOp {
		name: &'static str,
		inputs: usize,
		output_type: TensorElementType,
		activity: Arc<Activity>
	}

	impl Operator for SlowOp {
		type Kernel<'attr> = BoxedKernel<'attr>;

		fn name(&self) -> &str {
			self.name
		}

		fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
			(0..self.inputs).map(|_| OperatorInput::required(TensorElementType::Float32))
		}

		fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
			[OperatorOutput::required(self.output_type)]
		}

		fn create_kernel<'attr>(&self, _: &KernelContext<'attr>) -> Result<Self::Kernel<'attr>> {
			let activity = Arc::clone(&self.activity);
			Ok(Box::new(move |ctx: &ComputeContext| {
				let active = activity.active.fetch_add(1, Ordering::SeqCst) + 1;
				activity.max_active.fetch_max(active, Ordering::SeqCst);
				thread::sleep(Duration::from_millis(100));

				let shape = ctx.input(0)?.ok_or_else(|| Error::new("missing input"))?.shape().to_vec();
				ctx.output(0, shape)?.ok_or_else(|| Error::new("missing output"))?;

				activity.active.fetch_sub(1, Ordering::SeqCst);
				Ok(())
			}))
		}
	}

	#[test]
	fn test_abandoned_async_run_does_not_overlap() -> Result<()> {
		let activity = Arc::new(Activity::default());
		let slow_op = |name, inputs, output_type| SlowOp {
			name,
			inputs,
			output_type,
			activity: Arc::clone(&activity)
		};
		let mut session = Session::builder(test_env())?
			.with_intra_threads(2)?
			.with_operators(
				OperatorDomain::new("test.customop")?
					.add(slow_op("CustomOpOne", 2, TensorElementType::Float32))?
					.add(slow_op("CustomOpTwo", 1, TensorElementType::Int32))?
			)?
			.commit_from_file("tests/data/custom_op_test.onnx")?;
		let input = || Tensor::<f32>::new(&Allocator::default(), [3_usize, 5]);

		for forget in [false, true] {
			let options = RunOptions::new()?;
			let fut = session.run_async(crate::inputs![input()?, input()?], &options)?;
			thread::sleep(Duration::from_millis(20));
			if forget {
				mem::forget(fut);
			} else {
				drop(fut);
			}

			session.run(crate::inputs![input()?, input()?])?;
			assert_eq!(activity.max_active.load(Ordering::SeqCst), 1);
		}

		Ok(())
	}
}
