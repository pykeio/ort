use alloc::string::ToString;
use core::fmt;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
	#[default]
	None,
	Level1,
	Level2,
	Level3,
	MaximumEffort
}

impl fmt::Display for OptimizationLevel {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::None => "0",
			Self::Level1 => "1",
			Self::Level2 => "2",
			Self::Level3 => "3",
			Self::MaximumEffort => "65536"
		})
	}
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Target {
	/// Newer, more flexible & performant backend for INT8 models; supports STX, KRK, and newer devices.
	#[default]
	X2,
	/// Legacy backend for INT8 models; supports PHX, HPT, STX, and KRK devices.
	X1
}

impl fmt::Display for Target {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			Self::X2 => "X2",
			Self::X1 => "X1"
		})
	}
}

#[derive(Debug, Default, Clone)]
pub struct Vitis(ExecutionProviderOptions);

super::impl_ep!(arbitrary; Vitis);

impl Vitis {
	super::define_options! {
		pub fn with_config_file(mut self, config_file: impl ToString) -> Self = "config_file";

		pub fn with_cache_dir(mut self, cache_dir: impl ToString) -> Self = "cache_dir";

		/// Name of the subfolder under [`cache_dir`](Self::with_cache_dir) the compiled model will live in when
		/// [`with_cache_in_memory(false)`](Self::with_cache_in_memory). The default is the MD5 hash of the model.
		pub fn with_cache_key(mut self, cache_key: impl ToString) -> Self = "cache_key";

		/// Whether to store the compiled model in memory or save it to disk. Default is `true`, so the compiled model
		/// is not saved.
		///
		/// When disabled, the cache directory can be controlled with [`with_cache_dir`](Self::with_cache_dir).
		pub fn with_cache_in_memory(mut self, enable: bool) -> Self = "enable_cache_file_io_in_mem";

		pub fn with_xclbin(mut self, path: impl ToString) -> Self = "xclbin";

		/// Set the optimization level (for INT8 models only). Default is [`OptimizationLevel::None`].
		pub fn with_opt_level(mut self, level: OptimizationLevel) -> Self = "opt_level";

		/// Sets the backend used for INT8 models. Default is [`Target::X2`].
		pub fn with_target(mut self, target: Target) -> Self = "target";
	}
}

impl SimpleExecutionProvider for Vitis {
	const CANONICAL_NAME: &'static str = "VitisAIExecutionProvider";
	const SHORT_NAME: &'static str = "VitisAI";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
