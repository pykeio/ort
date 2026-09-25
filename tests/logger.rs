use std::sync::{Arc, Mutex};

use ort::{environment::Environment, logging::LogLevel, session::Session};

type Seen = Arc<Mutex<Vec<(String, String)>>>;

// Build the environment in its own frame so the builder's stack slot is gone once this returns.
#[inline(never)]
fn build_env(seen: &Seen) -> ort::Result<Environment> {
	ort::init()
		.with_name("logger-test")
		.with_logger(Arc::new({
			let seen = Arc::clone(seen);
			move |_level: LogLevel, category: &str, _id: &str, code_location: &str, _message: &str| {
				seen.lock().unwrap().push((category.to_string(), code_location.to_string()));
			}
		}))
		.build()
}

#[inline(never)]
fn clobber_stack() {
	let mut buf = [0xAAu8; 64 * 1024];
	std::hint::black_box(&mut buf);
}

#[test]
fn environment_custom_logger() -> ort::Result<()> {
	let seen = Seen::default();
	let env = build_env(&seen)?;
	clobber_stack();
	let logged_during_build = seen.lock().unwrap().len();

	let _session = Session::builder(&env)?
		.with_log_level(LogLevel::Verbose)?
		.commit_from_file("tests/data/upsample.onnx")?;

	let seen = seen.lock().unwrap();
	assert!(seen.len() > logged_during_build, "custom logger was not called after the environment was built");
	for (category, code_location) in seen.iter() {
		assert_ne!(category, code_location, "logger received the code location as the category");
	}
	Ok(())
}
