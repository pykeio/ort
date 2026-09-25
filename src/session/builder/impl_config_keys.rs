use alloc::{
	format,
	string::{String, ToString},
	vec::Vec
};
use core::fmt::Write;

use super::{BuilderResult, SessionBuilder};
use crate::error::{Error, ErrorCode};

/// Layout of the Value KV cache used by `GroupQueryAttention`, set with [`SessionBuilder::with_gqa_value_layout`].
///
/// Only the Value cache bound by the application (a `past_value` graph input and `present_value` graph output) is
/// affected; the Key cache always uses BNSH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GqaValueLayout {
	/// `(batch_size, num_heads, sequence_length, head_size)`, as in the operator schema. This is the default.
	///
	/// Committing fails if the model was already converted to BNHS.
	BNSH,
	/// `(batch_size, num_heads, head_size, sequence_length)`.
	///
	/// ONNX Runtime inserts transposes around each `GroupQueryAttention` node for execution providers that do not fuse
	/// them, which costs a copy of the Value cache in each direction per step. Committing fails if a cache can't be
	/// converted, such as with models in ORT format.
	BNHS
}

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

	/// Accepts a semicolon-separated list of optimizers to disable.
	///
	/// ```
	/// # use ort::session::Session;
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let session = Session::builder(&env)?
	/// 	.with_disabled_optimizers("ConstantFolding;GeluFusionL2")?
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
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

	/// Enable weightless mode, requesting that execution providers operate without embedding or copying constant
	/// initializers.
	///
	/// Committing will fail if an execution provider does not support weightless mode.
	///
	/// This option is **disabled** by default. Requires ONNX Runtime v1.29 or later.
	pub fn with_weightless(self) -> BuilderResult {
		self.with_config_entry("ep.enable_weightless", "1")
	}

	/// Sets the largest shapes expected for the given model inputs, which ONNX Runtime uses to estimate workspace
	/// requirements for models with dynamic shapes.
	///
	/// Symbolic dimensions are replaced by these values for estimation only; the shapes of inputs at runtime are not
	/// limited by this option. Each name must match a model input, and dimensions must be positive. Input names
	/// containing `:` or `;` are not supported.
	///
	/// ```
	/// # use ort::session::Session;
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let session = Session::builder(&env)?
	/// 	.with_max_shape_override([("string_input", [16])])?
	/// 	.commit_from_file("tests/data/vectorizer.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	///
	/// Requires ONNX Runtime v1.29 or later.
	pub fn with_max_shape_override<N: AsRef<str>, S: AsRef<[i64]>>(self, overrides: impl IntoIterator<Item = (N, S)>) -> BuilderResult {
		let mut value = String::new();
		for (name, shape) in overrides {
			let name = name.as_ref();
			if name.contains([':', ';']) {
				return Err(
					Error::new_with_code(ErrorCode::InvalidArgument, format!("input name `{name}` cannot be used in a max shape override")).with_recover(self)
				);
			}
			if !value.is_empty() {
				value.push(';');
			}
			let dims = shape.as_ref().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
			let _ = write!(value, "{name}:[{dims}]");
		}
		self.with_config_entry("session.max_shape_override", value)
	}

	/// Sets the layout of the Value KV cache that the application binds to `GroupQueryAttention`'s `past_value` input
	/// and `present_value` output. See [`GqaValueLayout`] for details.
	///
	/// ```
	/// # use ort::session::{Session, builder::GqaValueLayout};
	/// # fn main() -> ort::Result<()> {
	/// # let env = ort::test_util::test_env().clone();
	/// let session = Session::builder(&env)?
	/// 	.with_gqa_value_layout(GqaValueLayout::BNHS)?
	/// 	.commit_from_file("tests/data/upsample.onnx")?;
	/// # Ok(())
	/// # }
	/// ```
	///
	/// Requires ONNX Runtime v1.29.1 or later.
	pub fn with_gqa_value_layout(self, layout: GqaValueLayout) -> BuilderResult {
		self.with_config_entry(
			"session.gqa_value_layout",
			match layout {
				GqaValueLayout::BNSH => "BNSH",
				GqaValueLayout::BNHS => "BNHS"
			}
		)
	}

	/// Sets the path to the original (source) model when creating a session from a weightless EPContext model, so the
	/// execution provider can load initializer data from it.
	///
	/// If not set, the path stored in the EPContext node's `onnx_model_filename` attribute is used.
	///
	/// Requires ONNX Runtime v1.29 or later.
	pub fn with_ep_context_source_model_path(self, path: impl AsRef<str>) -> BuilderResult {
		self.with_config_entry("ep.context_source_model_path", path)
	}
}
