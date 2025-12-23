use std::{
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering}
	},
	thread::{self, JoinHandle}
};

use ort::{
	environment::{GlobalThreadPoolOptions, ThreadManager},
	session::Session
};

struct ThreadStats {
	active_threads: AtomicUsize
}

struct StdThread {
	stats: Arc<ThreadStats>,
	join_handle: JoinHandle<()>
}

impl StdThread {
	pub fn spawn(work: impl FnOnce() + Send + 'static, stats: &Arc<ThreadStats>) -> Self {
		let join_handle = thread::spawn(work);
		stats.active_threads.fetch_add(1, Ordering::AcqRel);
		Self {
			stats: Arc::clone(stats),
			join_handle
		}
	}

	pub fn join(self) {
		let _ = self.join_handle.join();
		self.stats.active_threads.fetch_sub(1, Ordering::AcqRel);
	}
}

struct StdThreadManager {
	stats: Arc<ThreadStats>
}

impl ThreadManager for StdThreadManager {
	type Thread = StdThread;

	fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
		Ok(StdThread::spawn(work, &self.stats))
	}

	fn join(thread: Self::Thread) -> ort::Result<()> {
		thread.join();
		Ok(())
	}
}

#[test]
fn global_thread_manager() -> ort::Result<()> {
	let stats = Arc::new(ThreadStats { active_threads: AtomicUsize::new(0) });

	ort::init()
		.with_name("integration_test")
		.with_global_thread_pool(
			GlobalThreadPoolOptions::default()
				.with_inter_threads(4)?
				.with_intra_threads(2)?
				.with_thread_manager(StdThreadManager { stats: Arc::clone(&stats) })?
		)
		.commit();

	let _session = Session::builder()?.commit_from_file("tests/data/upsample.ort")?;

	assert_eq!(stats.active_threads.load(Ordering::Acquire), 4);

	Ok(())
}
