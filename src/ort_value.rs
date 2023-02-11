use std::ffi::{c_char, CStr, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::ptr::NonNull;
use half::{bf16, f16};
use crate::{AllocatorType, MemType};
use crate::sys::{ONNXTensorElementDataType, ONNXType, OrtMemoryInfo, OrtTensorTypeAndShapeInfo, OrtTypeInfo, OrtValue};
use crate::sys::{
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
    ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
};

use super::{
    error::{OrtApiError, OrtError, OrtResult},
    memory::MemoryInfo,
    ort, ortsys, sys,
};

#[allow(clippy::upper_case_acronyms)]
pub trait AsONNXTensorElementDataType {
    fn as_onnx_tensor_element_data_type() -> ONNXTensorElementDataType;
}

#[macro_export]
macro_rules! impl_AsONNXTensorElementDataType {
    ($typ:ty, $onnx_tensor_element_data_type:expr$(,)?) => {
        impl AsONNXTensorElementDataType for $typ {
            fn as_onnx_tensor_element_data_type() -> ONNXTensorElementDataType {
                $onnx_tensor_element_data_type
            }
        }
    };
}

impl_AsONNXTensorElementDataType!(f32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
impl_AsONNXTensorElementDataType!(f16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
impl_AsONNXTensorElementDataType!(bf16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
impl_AsONNXTensorElementDataType!(u8, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
impl_AsONNXTensorElementDataType!(i8, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8);
impl_AsONNXTensorElementDataType!(u16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
impl_AsONNXTensorElementDataType!(i16, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16);
impl_AsONNXTensorElementDataType!(i32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
impl_AsONNXTensorElementDataType!(i64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
impl_AsONNXTensorElementDataType!(bool, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
impl_AsONNXTensorElementDataType!(f64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
impl_AsONNXTensorElementDataType!(u32, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32);
impl_AsONNXTensorElementDataType!(u64, ONNXTensorElementDataType_ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64);


#[derive(Debug)]
#[repr(transparent)]
pub struct Value<'d> {
    raw: *mut OrtValue,
    phantom: PhantomData<&'d ()>,
}

impl<'a> Value<'a>
{
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.13.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L1185-L1202)
    pub fn new_tensor_with_data<'d, T: AsONNXTensorElementDataType>(
        memory_info: &MemoryInfo,
        data: &'d [T],
        shape: &'d [i64],
    ) -> OrtResult<Self> {
        let mut value = ptr::null_mut::<OrtValue>();
        unsafe {
            ortsys![CreateTensorWithDataAsOrtValue(
            memory_info.ptr,
            data.as_ptr() as *const c_void as *mut c_void,
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            T::as_onnx_tensor_element_data_type(),
            &mut value,
        )];
        }
        Ok(Value { raw: value, phantom: PhantomData })
    }

    pub fn as_ptr(&self) -> *const OrtValue {
        self.raw
    }

    pub fn as_mut_ptr(&mut self) -> *mut OrtValue {
        self.raw
    }
}

impl<'d> Drop for Value<'d> {
    /// [`onnxruntime_c_api.h`](https://github.com/microsoft/onnxruntime/blob/v1.13.1/include/onnxruntime/core/session/onnxruntime_c_api.h#L1735)
    fn drop(&mut self) {
        unsafe {
            ortsys![ ReleaseValue(self.raw) ];
        }
    }
}
