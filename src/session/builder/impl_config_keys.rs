use alloc::string::ToString;

use super::{BuilderResult, SessionBuilder};

// https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h

impl SessionBuilder {
	/// Enable/disable the usage of prepacking.
	///
	/// This option is **enabled** by default.
	pub fn with_prepacking(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.disable_prepacking", if enable { "0" } else { "1" })
	}

	/// Use allocators from the registered environment.
	///
	/// This option is **disabled** by default.
	pub fn with_env_allocators(self) -> BuilderResult {
		self.with_config_entry("session.use_env_allocators", "1")
	}

	/// Disables subnormal floats by enabling the denormals-are-zero and flush-to-zero flags for all threads in the
	/// session's internal thread pool.
	///
	/// [Subnormal floats](https://en.wikipedia.org/wiki/Subnormal_number) are extremely small numbers very close to zero.
	/// Operations involving subnormal numbers can be very slow; enabling this flag will instead treat them as `0.0`,
	/// giving faster & more consistent performance, but lower accuracy (in cases where subnormals are involved).
	///
	/// This option is **disabled** by default, as it may hurt model accuracy.
	pub fn with_flush_to_zero(self) -> BuilderResult {
		self.with_config_entry("session.set_denormal_as_zero", "1")
	}

	/// Enable/disable fusion for quantized models in QDQ (`QuantizeLinear`/`DequantizeLinear`) format.
	///
	/// This option is **enabled** by default for all EPs except DirectML.
	pub fn with_quant_qdq(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.disable_quant_qdq", if enable { "0" } else { "1" })
	}

	/// Enable/disable the optimization step removing double QDQ nodes.
	///
	/// This option is **enabled** by default.
	pub fn with_double_qdq_remover(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.disable_double_qdq_remover", if enable { "0" } else { "1" })
	}

	/// Enable the removal of Q/DQ node pairs once all QDQ handling has been completed.
	///
	/// This option is **disabled** by default.
	pub fn with_qdq_cleanup(self) -> BuilderResult {
		self.with_config_entry("session.enable_quant_qdq_cleanup", "1")
	}

	/// Enable fast tanh-based GELU approximation (like PyTorch's `nn.GELU(approximate='tanh')`).
	///
	/// This option is **disabled** by default, as it may impact results.
	pub fn with_approximate_gelu(self) -> BuilderResult {
		self.with_config_entry("optimization.enable_gelu_approximation", "1")
	}

	/// Enable the `Cast` chain elimination optimization.
	///
	/// This option is **disabled** by default, as it may impact results.
	pub fn with_cast_chain_elimination(self) -> BuilderResult {
		self.with_config_entry("optimization.enable_cast_chain_elimination", "1")
	}

	/// Enable/disable ahead-of-time function inlining.
	///
	/// This option is **enabled** by default.
	pub fn with_aot_inlining(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.disable_aot_function_inlining", if enable { "0" } else { "1" })
	}

	/// Accepts a comma-separated list of optimizers to disable.
	pub fn with_disabled_optimizers(self, optimizers: impl AsRef<str>) -> BuilderResult {
		self.with_config_entry("optimization.disable_specified_optimizers", optimizers)
	}

	/// Enable using the device allocator for allocating initialized tensor memory, potentially bypassing arena
	/// allocators.
	///
	/// This option is **disabled** by default.
	pub fn with_device_allocated_initializers(self) -> BuilderResult {
		self.with_config_entry("session.use_device_allocator_for_initializers", "1")
	}

	/// Enable/disable allowing the inter-op threads to spin for a short period before blocking.
	///
	/// This option is **enabled** by defualt.
	pub fn with_inter_op_spinning(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.inter_op.allow_spinning", if enable { "1" } else { "0" })
	}

	/// Enable/disable allowing the intra-op threads to spin for a short period before blocking.
	///
	/// This option is **enabled** by defualt.
	pub fn with_intra_op_spinning(self, enable: bool) -> BuilderResult {
		self.with_config_entry("session.intra_op.allow_spinning", if enable { "1" } else { "0" })
	}

	/// Disables falling back to the CPU for operations not supported by any other EP.
	/// Models with graphs that cannot be placed entirely on the EP(s) will fail to commit.
	pub fn with_disable_cpu_fallback(self) -> BuilderResult {
		self.with_config_entry("session.disable_cpu_ep_fallback", "1")
	}

	/// Uses slower U8U8 matrix multiplication in place of U8S8 matrix multiplication that could potentially overflow on
	/// x86-64 platforms without the VNNI extension.
	///
	/// This should only be enabled if you encounter overflow issues with quantized models.
	pub fn with_precise_qmm(self) -> BuilderResult {
		self.with_config_entry("session.x64quantprecision", "1")
	}

	/// Enables dynamic thread block sizing with the given base block size.
	pub fn with_dynamic_block_base(self, size: u32) -> BuilderResult {
		self.with_config_entry("session.dynamic_block_base", size.to_string())
	}
}
