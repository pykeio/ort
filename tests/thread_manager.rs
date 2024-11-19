use std::{
	path::Path,
	sync::{
		Arc,
		atomic::{AtomicUsize, Ordering}
	},
	thread::{self, JoinHandle}
};

use image::{ImageBuffer, Luma, Pixel, imageops::FilterType};
use ort::{
	environment::{GlobalThreadPoolOptions, ThreadManager, ThreadWorker},
	inputs,
	session::{Session, builder::GraphOptimizationLevel}
};
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

#[test]
fn session_thread_manager() -> ort::Result<()> {
	const IMAGE_TO_LOAD: &str = "mnist_5.jpg";

	let stats = Arc::new(ThreadStats { active_threads: AtomicUsize::new(0) });

	let session = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_inter_threads(2)?
		.with_intra_threads(2)?
		.with_thread_manager(StdThreadManager { stats: Arc::clone(&stats) })?
		.commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/mnist.onnx")
		.expect("Could not download model from file");

	assert_eq!(stats.active_threads.load(Ordering::Acquire), 1);

	let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(IMAGE_TO_LOAD))
		.unwrap()
		.resize(28, 28, FilterType::Nearest)
		.to_luma8();
	let array = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
		let pixel = image_buffer.get_pixel(i as u32, j as u32);
		let channels = pixel.channels();
		(channels[c] as f32) / 255.0
	});

	let _ = session.run(inputs![array]?)?;

	assert_eq!(stats.active_threads.load(Ordering::Acquire), 1);

	Ok(())
}
