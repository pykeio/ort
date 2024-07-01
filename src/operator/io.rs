use crate::{memory::MemoryType, tensor::TensorElementType};

#[repr(i32)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum InputOutputCharacteristic {
	#[default]
	Required = 0,
	Optional = 1,
	Variadic = 2
}

impl From<InputOutputCharacteristic> for ort_sys::OrtCustomOpInputOutputCharacteristic {
	fn from(val: InputOutputCharacteristic) -> Self {
		match val {
			InputOutputCharacteristic::Required => ort_sys::OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
			InputOutputCharacteristic::Optional => ort_sys::OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL,
			InputOutputCharacteristic::Variadic => ort_sys::OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC
		}
	}
}

pub struct OperatorInput {
	pub(crate) characteristic: InputOutputCharacteristic,
	pub(crate) r#type: Option<TensorElementType>,
	pub(crate) variadic_min_arity: Option<usize>,
	pub(crate) variadic_homogeneity: Option<bool>,
	pub(crate) memory_type: MemoryType
}

impl OperatorInput {
	#[inline]
	pub const fn required(r#type: TensorElementType) -> Self {
		Self {
			r#type: Some(r#type),
			characteristic: InputOutputCharacteristic::Required,
			variadic_homogeneity: None,
			variadic_min_arity: None,
			memory_type: MemoryType::Default
		}
	}

	#[inline]
	pub const fn optional(r#type: TensorElementType) -> Self {
		Self {
			r#type: Some(r#type),
			characteristic: InputOutputCharacteristic::Optional,
			variadic_homogeneity: None,
			variadic_min_arity: None,
			memory_type: MemoryType::Default
		}
	}

	#[inline]
	pub const fn variadic(min_arity: usize) -> Self {
		Self {
			r#type: None,
			characteristic: InputOutputCharacteristic::Variadic,
			variadic_homogeneity: None,
			variadic_min_arity: Some(min_arity),
			memory_type: MemoryType::Default
		}
	}

	#[inline]
	pub const fn homogenous(mut self, r#type: TensorElementType) -> Self {
		self.r#type = Some(r#type);
		self.variadic_homogeneity = Some(true);
		self
	}

	#[inline]
	pub const fn memory_type(mut self, memory_type: MemoryType) -> Self {
		self.memory_type = memory_type;
		self
	}
}

pub struct OperatorOutput {
	pub(crate) characteristic: InputOutputCharacteristic,
	pub(crate) r#type: Option<TensorElementType>,
	pub(crate) variadic_min_arity: Option<usize>,
	pub(crate) variadic_homogeneity: Option<bool>
}

impl OperatorOutput {
	#[inline]
	pub const fn required(r#type: TensorElementType) -> Self {
		Self {
			r#type: Some(r#type),
			characteristic: InputOutputCharacteristic::Required,
			variadic_homogeneity: None,
			variadic_min_arity: None
		}
	}

	#[inline]
	pub const fn optional(r#type: TensorElementType) -> Self {
		Self {
			r#type: Some(r#type),
			characteristic: InputOutputCharacteristic::Optional,
			variadic_homogeneity: None,
			variadic_min_arity: None
		}
	}

	#[inline]
	pub const fn variadic(min_arity: usize) -> Self {
		Self {
			r#type: None,
			characteristic: InputOutputCharacteristic::Variadic,
			variadic_homogeneity: None,
			variadic_min_arity: Some(min_arity)
		}
	}

	#[inline]
	pub const fn homogenous(mut self, r#type: TensorElementType) -> Self {
		self.r#type = Some(r#type);
		self.variadic_homogeneity = Some(true);
		self
	}
}
