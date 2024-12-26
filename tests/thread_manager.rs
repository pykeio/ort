use std::{
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering}
	},
	thread::{self, JoinHandle}
};

use ort::environment::{GlobalThreadPoolOptions, ThreadManager, ThreadWorker};
use test_log::test;

struct ThreadStats {
	active_threads: AtomicUsize
}

struct StdThread {
	stats: Arc<ThreadStats>,
	join_handle: JoinHandle<()>
}

impl StdThread {
	pub fn spawn(worker: ThreadWorker, stats: &Arc<ThreadStats>) -> Self {
		let join_handle = thread::spawn(move || worker.work());
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

	fn create(&mut self, worker: ThreadWorker) -> ort::Result<Self::Thread> {
		Ok(StdThread::spawn(worker, &self.stats))
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
		.commit()?;

	assert_eq!(stats.active_threads.load(Ordering::Acquire), 4);

	Ok(())
}
