use super::SessionBuilder;
use crate::Result;

// https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h

impl SessionBuilder {
	/// Enable/disable the usage of prepacking.
	///
	/// This option is **enabled** by default.
	pub fn with_prepacking(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.disable_prepacking", if enable { "0" } else { "1" })?;
		Ok(self)
	}

	/// Use allocators from the registered environment.
	///
	/// This option is **disabled** by default.
	pub fn with_env_allocators(mut self) -> Result<Self> {
		self.add_config_entry("session.use_env_allocators", "1")?;
		Ok(self)
	}

	/// Enable flush-to-zero and denormal-as-zero.
	///
	/// This option is **disabled** by default, as it may hurt model accuracy.
	pub fn with_denormal_as_zero(mut self) -> Result<Self> {
		self.add_config_entry("session.set_denormal_as_zero", "1")?;
		Ok(self)
	}

	/// Enable/disable fusion for quantized models in QDQ (QuantizeLinear/DequantizeLinear) format.
	///
	/// This option is **enabled** by default for all EPs except DirectML.
	pub fn with_quant_qdq(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.disable_quant_qdq", if enable { "0" } else { "1" })?;
		Ok(self)
	}

	/// Enable/disable the optimization step removing double QDQ nodes.
	///
	/// This option is **enabled** by default.
	pub fn with_double_qdq_remover(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.disable_double_qdq_remover", if enable { "0" } else { "1" })?;
		Ok(self)
	}

	/// Enable the removal of Q/DQ node pairs once all QDQ handling has been completed.
	///
	/// This option is **disabled** by default.
	pub fn with_qdq_cleanup(mut self) -> Result<Self> {
		self.add_config_entry("session.enable_quant_qdq_cleanup", "1")?;
		Ok(self)
	}

	/// Enable fast GELU approximation.
	///
	/// This option is **disabled** by default, as it may hurt accuracy.
	pub fn with_approximate_gelu(mut self) -> Result<Self> {
		self.add_config_entry("optimization.enable_gelu_approximation", "1")?;
		Ok(self)
	}

	/// Enable/disable ahead-of-time function inlining.
	///
	/// This option is **enabled** by default.
	pub fn with_aot_inlining(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.disable_aot_function_inlining", if enable { "0" } else { "1" })?;
		Ok(self)
	}

	/// Accepts a comma-separated list of optimizers to disable.
	pub fn with_disabled_optimizers(mut self, optimizers: &str) -> Result<Self> {
		self.add_config_entry("optimization.disable_specified_optimizers", optimizers)?;
		Ok(self)
	}

	/// Enable using device allocator for allocating initialized tensor memory.
	///
	/// This option is **disabled** by default.
	pub fn with_device_allocator_for_initializers(mut self) -> Result<Self> {
		self.add_config_entry("session.use_device_allocator_for_initializers", "1")?;
		Ok(self)
	}

	/// Enable/disable allowing the inter-op threads to spin for a short period before blocking.
	///
	/// This option is **enabled** by defualt.
	pub fn with_inter_op_spinning(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.inter_op.allow_spinning", if enable { "1" } else { "0" })?;
		Ok(self)
	}

	/// Enable/disable allowing the intra-op threads to spin for a short period before blocking.
	///
	/// This option is **enabled** by defualt.
	pub fn with_intra_op_spinning(mut self, enable: bool) -> Result<Self> {
		self.add_config_entry("session.intra_op.allow_spinning", if enable { "1" } else { "0" })?;
		Ok(self)
	}
}
