use std::{
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering}
	},
	thread
};

use ort::{environment::ThreadManager, session::Session, value::Tensor};

struct DetachedThreadManager {
	created: Arc<AtomicUsize>
}

impl ThreadManager for DetachedThreadManager {
	type Thread = ();

	fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
		self.created.fetch_add(1, Ordering::AcqRel);
		thread::spawn(work);
		Ok(())
	}

	fn join(_: Self::Thread) -> ort::Result<()> {
		Ok(())
	}
}

#[test]
fn detached_thread_manager() -> ort::Result<()> {
	let created = Arc::new(AtomicUsize::new(0));
	for _ in 0..8 {
		let mut session = Session::builder(ort_test_util::test_env())?
			.with_intra_threads(4)?
			.with_thread_manager(DetachedThreadManager { created: Arc::clone(&created) })?
			.commit_from_file("tests/data/upsample.onnx")?;
		session.run(ort::inputs![Tensor::<f32>::new(session.allocator(), [1_usize, 32, 32, 3])?])?;
		drop(session);
	}
	assert!(created.load(Ordering::Acquire) > 0);
	Ok(())
}
