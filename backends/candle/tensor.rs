use candle_core::{DType, Shape};

pub struct TypeInfo {
	pub dtype: DType,
	pub shape: Shape
}

impl TypeInfo {
	pub fn new_sys(dtype: DType, shape: Shape) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(Self { dtype, shape })) as *mut TypeInfo).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		Box::from_raw(ptr.cast::<TypeInfo>())
	}
}
