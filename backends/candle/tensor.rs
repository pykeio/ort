use candle_core::DType;

pub struct TypeInfo {
	pub dtype: DType,
	pub shape: Vec<i64>
}

impl TypeInfo {
	pub fn new_sys(dtype: DType, shape: Vec<i64>) -> *mut ort_sys::OrtTypeInfo {
		(Box::leak(Box::new(Self { dtype, shape })) as *mut TypeInfo).cast()
	}

	pub unsafe fn consume_sys(ptr: *mut ort_sys::OrtTypeInfo) -> Box<TypeInfo> {
		Box::from_raw(ptr.cast::<TypeInfo>())
	}
}
