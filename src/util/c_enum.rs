use core::{marker::PhantomData, mem::transmute};

/// # Safety
/// The enum must be `#[repr(i32)]`.
pub(crate) unsafe trait CEnum: Copy + 'static {
	const VARIANTS: &'static [Self];
}

#[inline]
pub(crate) fn from_raw<E: CEnum>(raw: i32) -> Option<E> {
	const { assert!(size_of::<E>() == size_of::<i32>()) };
	// rustc 1.100 is able to optimize this very well even at opt-level 1
	E::VARIANTS
		.iter()
		.find(|variant| unsafe { *(*variant as *const E).cast::<i32>() } == raw)
		.copied()
}

#[inline]
pub(crate) unsafe fn call<A, E: CEnum>(f: unsafe extern "system" fn(A) -> E, arg: A) -> Option<E> {
	let f = unsafe { transmute::<unsafe extern "system" fn(A) -> E, unsafe extern "system" fn(A) -> i32>(f) };
	from_raw(unsafe { f(arg) })
}

pub(crate) struct Out<E: CEnum> {
	raw: i32,
	_enum: PhantomData<E>
}

impl<E: CEnum> Out<E> {
	#[inline]
	pub const fn new(initial: E) -> Self {
		const { assert!(size_of::<E>() == size_of::<i32>()) };
		Self {
			raw: unsafe { *(&raw const initial).cast::<i32>() },
			_enum: PhantomData
		}
	}

	#[inline]
	pub const fn as_mut_ptr(&mut self) -> *mut E {
		(&raw mut self.raw).cast()
	}

	#[inline]
	pub fn get(&self) -> Option<E> {
		from_raw(self.raw)
	}
}

macro_rules! c_enum {
	($($(#[$attr:meta])* $ty:ident { $($variant:ident),+ $(,)? })+) => {$(
		$(#[$attr])*
		unsafe impl CEnum for ort_sys::$ty {
			const VARIANTS: &'static [Self] = &[$(ort_sys::$ty::$variant),+];
		}

		$(#[$attr])*
		const _: () = {
			#[allow(dead_code)]
			fn assert_exhaustive(value: ort_sys::$ty) {
				match value {
					$(ort_sys::$ty::$variant => {}),+
				}
			}
		};
	)+};
}

c_enum! {
	ONNXTensorElementDataType {
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0
	}
	ONNXType {
		ONNX_TYPE_UNKNOWN,
		ONNX_TYPE_TENSOR,
		ONNX_TYPE_SEQUENCE,
		ONNX_TYPE_MAP,
		ONNX_TYPE_OPAQUE,
		ONNX_TYPE_SPARSETENSOR,
		ONNX_TYPE_OPTIONAL
	}
	OrtErrorCode {
		ORT_OK,
		ORT_FAIL,
		ORT_INVALID_ARGUMENT,
		ORT_NO_SUCHFILE,
		ORT_NO_MODEL,
		ORT_ENGINE_ERROR,
		ORT_RUNTIME_EXCEPTION,
		ORT_INVALID_PROTOBUF,
		ORT_MODEL_LOADED,
		ORT_NOT_IMPLEMENTED,
		ORT_INVALID_GRAPH,
		ORT_EP_FAIL,
		ORT_MODEL_LOAD_CANCELED,
		ORT_MODEL_REQUIRES_COMPILATION,
		ORT_NOT_FOUND,
		ORT_DEVICE_RESET
	}
	OrtLoggingLevel {
		ORT_LOGGING_LEVEL_VERBOSE,
		ORT_LOGGING_LEVEL_INFO,
		ORT_LOGGING_LEVEL_WARNING,
		ORT_LOGGING_LEVEL_ERROR,
		ORT_LOGGING_LEVEL_FATAL
	}
	OrtAllocatorType {
		OrtInvalidAllocator,
		OrtDeviceAllocator,
		OrtArenaAllocator,
		OrtReadOnlyAllocator
	}
	OrtMemType {
		OrtMemTypeCPUInput,
		OrtMemTypeCPUOutput,
		OrtMemTypeDefault
	}
	OrtMemoryInfoDeviceType {
		OrtMemoryInfoDeviceType_CPU,
		OrtMemoryInfoDeviceType_GPU,
		OrtMemoryInfoDeviceType_FPGA,
		OrtMemoryInfoDeviceType_NPU
	}
	OrtDeviceMemoryType {
		OrtDeviceMemoryType_DEFAULT,
		OrtDeviceMemoryType_HOST_ACCESSIBLE
	}
	OrtHardwareDeviceType {
		OrtHardwareDeviceType_CPU,
		OrtHardwareDeviceType_GPU,
		OrtHardwareDeviceType_NPU
	}
	OrtCompiledModelCompatibility {
		EP_NOT_APPLICABLE,
		EP_SUPPORTED_OPTIMAL,
		EP_SUPPORTED_PREFER_RECOMPILATION,
		EP_UNSUPPORTED
	}
	#[cfg(feature = "training")]
	OrtPropertyType {
		OrtIntProperty,
		OrtFloatProperty,
		OrtStringProperty
	}
}

#[cfg(test)]
mod tests {
	use core::mem::transmute;

	use super::{Out, call, from_raw};

	#[test]
	fn test_from_raw() {
		assert_eq!(from_raw(0), Some(ort_sys::OrtErrorCode::ORT_OK));
		assert_eq!(from_raw(15), Some(ort_sys::OrtErrorCode::ORT_DEVICE_RESET));
		assert_eq!(from_raw::<ort_sys::OrtErrorCode>(16), None);
		assert_eq!(from_raw::<ort_sys::OrtErrorCode>(-1), None);
		assert_eq!(from_raw(-1), Some(ort_sys::OrtAllocatorType::OrtInvalidAllocator));
		assert_eq!(from_raw(5), Some(ort_sys::OrtDeviceMemoryType::OrtDeviceMemoryType_HOST_ACCESSIBLE));
		assert_eq!(from_raw::<ort_sys::OrtDeviceMemoryType>(1), None);
	}

	#[test]
	fn test_out() {
		let mut out = Out::new(ort_sys::ONNXType::ONNX_TYPE_MAP);
		assert_eq!(out.get(), Some(ort_sys::ONNXType::ONNX_TYPE_MAP));

		unsafe { out.as_mut_ptr().cast::<i32>().write(1234) };
		assert_eq!(out.get(), None);
		unsafe { out.as_mut_ptr().cast::<i32>().write(6) };
		assert_eq!(out.get(), Some(ort_sys::ONNXType::ONNX_TYPE_OPTIONAL));
	}

	#[test]
	fn test_call() {
		unsafe extern "system" fn identity(code: i32) -> i32 {
			code
		}

		let f = unsafe { transmute::<unsafe extern "system" fn(i32) -> i32, unsafe extern "system" fn(i32) -> ort_sys::OrtErrorCode>(identity) };
		assert_eq!(unsafe { call(f, 2) }, Some(ort_sys::OrtErrorCode::ORT_INVALID_ARGUMENT));
		assert_eq!(unsafe { call(f, 21) }, None);
	}
}
