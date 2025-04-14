mod create;
mod extract;

use alloc::{format, string::ToString, sync::Arc};
use core::{
	fmt::{self, Debug},
	marker::PhantomData,
	ops::{Index, IndexMut},
	ptr::{self, NonNull}
};

pub use self::create::{OwnedTensorArrayData, TensorArrayData, TensorArrayDataMut, TensorArrayDataParts, ToShape};
use super::{DowncastableTarget, DynValue, Value, ValueInner, ValueRef, ValueRefMut, ValueType, ValueTypeMarker};
use crate::{
	AsPointer,
	error::Result,
	memory::{AllocationDevice, Allocator, MemoryInfo},
	ortsys,
	tensor::{IntoTensorElementType, Shape, SymbolicDimensions, TensorElementType}
};

pub trait TensorValueTypeMarker: ValueTypeMarker {
	private_trait!();
}

#[derive(Debug)]
pub struct DynTensorValueType;
impl ValueTypeMarker for DynTensorValueType {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("DynTensor")
	}

	private_impl!();
}
impl TensorValueTypeMarker for DynTensorValueType {
	private_impl!();
}

#[derive(Debug)]
pub struct TensorValueType<T: IntoTensorElementType + Debug>(PhantomData<T>);
impl<T: IntoTensorElementType + Debug> ValueTypeMarker for TensorValueType<T> {
	fn fmt(f: &mut fmt::Formatter) -> fmt::Result {
		f.write_str("Tensor<")?;
		<TensorElementType as fmt::Display>::fmt(&T::into_tensor_element_type(), f)?;
		f.write_str(">")
	}

	private_impl!();
}
impl<T: IntoTensorElementType + Debug> TensorValueTypeMarker for TensorValueType<T> {
	private_impl!();
}

/// A tensor [`Value`] whose data type is unknown.
pub type DynTensor = Value<DynTensorValueType>;
/// A strongly-typed tensor [`Value`].
pub type Tensor<T> = Value<TensorValueType<T>>;

/// A reference to a tensor [`Value`] whose data type is unknown.
pub type DynTensorRef<'v> = ValueRef<'v, DynTensorValueType>;
/// A mutable reference to a tensor [`Value`] whose data type is unknown.
pub type DynTensorRefMut<'v> = ValueRefMut<'v, DynTensorValueType>;
/// A reference to a strongly-typed tensor [`Value`].
pub type TensorRef<'v, T> = ValueRef<'v, TensorValueType<T>>;
/// A mutable reference to a strongly-typed tensor [`Value`].
pub type TensorRefMut<'v, T> = ValueRefMut<'v, TensorValueType<T>>;

impl DowncastableTarget for DynTensorValueType {
	fn can_downcast(dtype: &ValueType) -> bool {
		matches!(dtype, ValueType::Tensor { .. })
	}

	private_impl!();
}

impl DynTensor {
	/// Construct a tensor via a given allocator with a given shape and datatype. The data in the tensor will be
	/// **uninitialized**.
	///
	/// This can be used to create a tensor with data on a certain device. For example, to create a tensor with pinned
	/// (CPU) memory for use with CUDA:
	/// ```no_run
	/// # use ort::{memory::{Allocator, MemoryInfo, MemoryType, AllocationDevice, AllocatorType}, session::Session, tensor::TensorElementType, value::DynTensor};
	/// # fn main() -> ort::Result<()> {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA_PINNED, 0, AllocatorType::Device, MemoryType::CPUInput)?
	/// )?;
	///
	/// let mut img_input = DynTensor::new(&allocator, TensorElementType::Float32, [1_usize, 128, 128, 3])?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn new(allocator: &Allocator, data_type: TensorElementType, shape: impl Into<Shape>) -> Result<DynTensor> {
		Self::new_inner(allocator, data_type, shape.into())
	}

	fn new_inner(allocator: &Allocator, data_type: TensorElementType, shape: Shape) -> Result<DynTensor> {
		let mut value_ptr: *mut ort_sys::OrtValue = ptr::null_mut();

		let shape_ptr: *const i64 = shape.as_ptr();
		let shape_len = shape.len();

		ortsys![
			unsafe CreateTensorAsOrtValue(
				allocator.ptr().cast_mut(),
				shape_ptr,
				shape_len,
				data_type.into(),
				&mut value_ptr
			)?;
			nonNull(value_ptr)
		];

		// `CreateTensorAsOrtValue` actually does not guarantee that the data allocated is zero'd out, so if we can, we should
		// do it manually.
		let memory_info = MemoryInfo::from_value(value_ptr).expect("CreateTensorAsOrtValue returned non-tensor");
		if memory_info.is_cpu_accessible() && data_type != TensorElementType::String {
			let mut buffer_ptr: *mut ort_sys::c_void = ptr::null_mut();
			ortsys![unsafe GetTensorMutableData(value_ptr, &mut buffer_ptr)?];
			if !buffer_ptr.is_null() {
				unsafe { buffer_ptr.write_bytes(0, data_type.byte_size(shape.num_elements())) };
			}
		}

		Ok(Value {
			inner: Arc::new(ValueInner {
				ptr: unsafe { NonNull::new_unchecked(value_ptr) },
				dtype: ValueType::Tensor {
					ty: data_type,
					shape,
					dimension_symbols: SymbolicDimensions::empty(shape_len)
				},
				drop: true,
				memory_info: MemoryInfo::from_value(value_ptr),
				_backing: None
			}),
			_markers: PhantomData
		})
	}
}

impl<Type: TensorValueTypeMarker + ?Sized> Value<Type> {
	/// Returns a mutable pointer to the tensor's data. The pointer may be null in the case of zero-sized tensors.
	///
	/// It's important to note that the resulting pointer may not point to CPU-accessible memory. In the case of a
	/// tensor created on a different EP device, e.g. via [`Tensor::new`], the pointer returned by this function may be
	/// a CUDA pointer, which would require a separate crate (like [`cudarc`](https://crates.io/crates/cudarc)) to access.
	/// Use [`Tensor::memory_info`] & [`MemoryInfo::allocation_device`] to check which device the data resides on before
	/// accessing it.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let mut tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	/// let ptr = tensor.data_ptr_mut().cast::<i64>();
	/// unsafe {
	/// 	*ptr.add(3) = 42;
	/// };
	///
	/// let (_, extracted) = tensor.extract_tensor();
	/// assert_eq!(&extracted, &[0, 1, 2, 42, 4]);
	/// # Ok(())
	/// # }
	/// ```
	pub fn data_ptr_mut(&mut self) -> *mut ort_sys::c_void {
		let mut buffer_ptr: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr_mut(), &mut buffer_ptr).expect("failed to get tensor data")]; // infallible
		buffer_ptr
	}

	/// Returns an immutable pointer to the tensor's underlying data. The pointer may be null in the case of zero-sized
	/// tensors.
	///
	/// It's important to note that the resulting pointer may not point to CPU-accessible memory. In the case of a
	/// tensor created on a different EP device, e.g. via [`Tensor::new`], the pointer returned by this function may be
	/// a CUDA pointer, which would require a separate crate (like [`cudarc`](https://crates.io/crates/cudarc)) to access.
	/// Use [`Tensor::memory_info`] & [`MemoryInfo::allocation_device`] to check which device the data resides on before
	/// accessing it.
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<i64>::from_array((vec![5], vec![0, 1, 2, 3, 4]))?;
	/// let ptr = tensor.data_ptr().cast::<i64>();
	/// assert_eq!(unsafe { *ptr.add(3) }, 3);
	/// # Ok(())
	/// # }
	/// ```
	pub fn data_ptr(&self) -> *const ort_sys::c_void {
		let mut buffer_ptr: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe GetTensorMutableData(self.ptr().cast_mut(), &mut buffer_ptr).expect("failed to get tensor data")]; // infallible
		buffer_ptr
	}

	/// Returns information about the device this tensor is allocated on.
	///
	/// ```
	/// # use ort::{memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType}, session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	/// // Tensors are allocated on CPU by default.
	/// assert_eq!(tensor.memory_info().allocation_device(), AllocationDevice::CPU);
	///
	/// # if false {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let cuda_allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
	/// )?;
	/// let tensor = Tensor::<f32>::new(&cuda_allocator, [1_usize, 3, 224, 224])?;
	/// assert_eq!(tensor.memory_info().allocation_device(), AllocationDevice::CUDA);
	/// # }
	/// # Ok(())
	/// # }
	/// ```
	pub fn memory_info(&self) -> &MemoryInfo {
		unsafe { self.inner.memory_info.as_ref().unwrap_unchecked() }
	}

	/// Copies the contents of this tensor to another device, returning the newly created tensor value.
	///
	/// ```
	/// # use ort::{memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType}, session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # if false {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let cuda_allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
	/// )?;
	/// let cuda_tensor = Tensor::<f32>::new(&cuda_allocator, [1_usize, 3, 224, 224])?;
	/// # }
	/// # let cuda_tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	///
	/// let cpu_tensor = cuda_tensor.to(AllocationDevice::CPU, 0)?;
	/// assert_eq!(cpu_tensor.memory_info().allocation_device(), AllocationDevice::CPU);
	/// assert_eq!(**cpu_tensor.shape(), [1, 3, 224, 224]);
	/// # Ok(())
	/// # }
	/// ```
	pub fn to(&self, device: AllocationDevice, device_id: i32) -> Result<Value<Type>> {
		use crate::{
			OnceLock, execution_providers as ep,
			io_binding::IoBinding,
			memory::{AllocatorType, MemoryType},
			session::{Session, builder::GraphOptimizationLevel},
			util::{MiniMap, Mutex}
		};

		type IdentitySessionKey = (AllocationDevice, i32, ort_sys::ONNXTensorElementDataType);
		type IdentitySession = (Session, IoBinding);

		static SESSIONS: OnceLock<Mutex<MiniMap<IdentitySessionKey, IdentitySession>>> = OnceLock::new();
		static IDENTITY_MODEL: &[u8] = include_bytes!("./identity.ort");

		let target_memory_info = MemoryInfo::new(device, device_id, AllocatorType::Device, MemoryType::Default)?;
		let tensor_type = ort_sys::ONNXTensorElementDataType::from(*self.data_type());

		let mut sessions = SESSIONS.get_or_init(|| Mutex::new(MiniMap::new())).lock();
		let (session, binding) = match sessions.get_mut(&(device, device_id, tensor_type)) {
			Some(entry) => entry,
			None => {
				let mut model_bytes = IDENTITY_MODEL.to_vec();
				// Override the expected element type of the input & output nodes, respectively.
				model_bytes[544] = tensor_type as u8;
				model_bytes[604] = tensor_type as u8;

				let session = Session::builder()?
					.with_optimization_level(GraphOptimizationLevel::Disable)?
					.with_intra_threads(1)?
					.with_inter_threads(1)?
					.with_inter_op_spinning(false)?
					.with_intra_op_spinning(false)?
					.with_memory_pattern(false)?
					.with_no_environment_execution_providers()?
					.with_execution_providers([match device {
						AllocationDevice::CPU => ep::CPUExecutionProvider::default().with_arena_allocator(false).build(),
						AllocationDevice::CUDA | AllocationDevice::CUDA_PINNED => ep::CUDAExecutionProvider::default()
							.with_device_id(device_id)
							.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
							.with_conv_max_workspace(false)
							.with_conv_algorithm_search(ep::cuda::CuDNNConvAlgorithmSearch::Default)
							.build(),
						AllocationDevice::DIRECTML | AllocationDevice::DIRECTML_CPU => {
							ep::DirectMLExecutionProvider::default().with_device_id(device_id).build()
						}
						AllocationDevice::CANN | AllocationDevice::CANN_PINNED => ep::CANNExecutionProvider::default()
							.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
							.with_cann_graph(false)
							.with_device_id(device_id)
							.build(),
						AllocationDevice::OPENVINO_CPU | AllocationDevice::OPENVINO_GPU => ep::OpenVINOExecutionProvider::default()
							.with_num_threads(1)
							.with_device_type(if device == AllocationDevice::OPENVINO_CPU {
								"CPU".to_string()
							} else {
								format!("GPU.{device_id}")
							})
							.build(),
						AllocationDevice::HIP | AllocationDevice::HIP_PINNED => ep::ROCmExecutionProvider::default()
							.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
							.with_hip_graph(false)
							.with_exhaustive_conv_search(false)
							.with_device_id(device_id)
							.build(),
						AllocationDevice::TVM => ep::TVMExecutionProvider::default().build(),
						AllocationDevice::XNNPACK => ep::XNNPACKExecutionProvider::default().build(),
						_ => return Err(crate::Error::new("Unsupported allocation device {device} for tensor copy target"))
					}])?
					.with_allocator(MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?)?
					.commit_from_memory(&model_bytes)?;
				let binding = session.create_binding()?;
				sessions.insert((device, device_id, tensor_type), (session, binding));
				sessions.get_mut(&(device, device_id, tensor_type)).expect("insert should have worked")
			}
		};

		binding.bind_input("input", self)?;
		binding.bind_output_to_device("output", &target_memory_info)?;

		let output = session
			.run_binding(binding)?
			.remove("output")
			.expect("identity model should have single output");
		Ok(unsafe { output.transmute_type() })
	}
}

impl<Type: TensorValueTypeMarker + ?Sized> Clone for Value<Type> {
	/// Creates a copy of this tensor and its data on the same device it resides on.
	///
	/// ```
	/// # use ort::{value::Tensor, AsPointer};
	/// # fn main() -> ort::Result<()> {
	/// let array = vec![1_i64, 2, 3, 4, 5];
	/// let tensor = Tensor::from_array(([array.len()], array.into_boxed_slice()))?;
	///
	/// let new_tensor = tensor.clone();
	///
	/// // same data
	/// assert_eq!(tensor.extract_tensor(), new_tensor.extract_tensor());
	///
	/// // different allocations
	/// assert_ne!(tensor.ptr(), new_tensor.ptr());
	/// assert_ne!(tensor.data_ptr(), new_tensor.data_ptr());
	/// # 	Ok(())
	/// # }
	/// ```
	fn clone(&self) -> Self {
		let memory_info = self.memory_info();
		self.to(memory_info.allocation_device(), memory_info.device_id())
			.expect("Failed to clone tensor")
	}
}

impl<T: IntoTensorElementType + Debug> Tensor<T> {
	/// Converts from a strongly-typed [`Tensor<T>`] to a type-erased [`DynTensor`].
	///
	/// ```
	/// # use ort::{memory::Allocator, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	/// let tensor_dyn = tensor.upcast();
	/// assert!(tensor_dyn.try_extract_tensor::<f32>().is_ok());
	/// assert!(tensor_dyn.try_extract_tensor::<i64>().is_err());
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast(self) -> DynTensor {
		unsafe { self.transmute_type() }
	}

	/// Creates a type-erased [`DynTensorRef`] from a strongly-typed [`Tensor<T>`].
	///
	/// ```
	/// # use ort::{memory::Allocator, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	/// let tensor_dyn = tensor.upcast_ref();
	///
	/// let (_, original_extract) = tensor.extract_tensor();
	/// let (_, ref_extract) = tensor_dyn.try_extract_tensor::<f32>()?;
	/// assert_eq!(original_extract, ref_extract);
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast_ref(&self) -> DynTensorRef {
		DynTensorRef::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}

	/// Converts from a strongly-typed [`Tensor<T>`] to a mutable reference to a type-erased [`DynTensor`].
	///
	/// ```
	/// # use ort::value::Tensor;
	/// # fn main() -> ort::Result<()> {
	/// let mut tensor = Tensor::<i64>::from_array((vec![5], vec![1, 2, 3, 4, 5]))?;
	/// let mut tensor_dyn = tensor.upcast_mut();
	///
	/// let (_, mut_view) = tensor_dyn.try_extract_tensor_mut::<i64>()?;
	/// mut_view[3] = 0;
	///
	/// let (_, original_view) = tensor.extract_tensor();
	/// assert_eq!(original_view, &[1, 2, 3, 0, 5]);
	/// # Ok(())
	/// # }
	/// ```
	#[inline]
	pub fn upcast_mut(&mut self) -> DynTensorRefMut {
		DynTensorRefMut::new(Value {
			inner: Arc::clone(&self.inner),
			_markers: PhantomData
		})
	}
}

impl<T: IntoTensorElementType + Debug> DowncastableTarget for TensorValueType<T> {
	fn can_downcast(dtype: &ValueType) -> bool {
		match dtype {
			ValueType::Tensor { ty, .. } => *ty == T::into_tensor_element_type(),
			_ => false
		}
	}

	private_impl!();
}

impl<T: IntoTensorElementType + Debug> From<Value<TensorValueType<T>>> for DynValue {
	fn from(value: Value<TensorValueType<T>>) -> Self {
		value.into_dyn()
	}
}
impl From<Value<DynTensorValueType>> for DynValue {
	fn from(value: Value<DynTensorValueType>) -> Self {
		value.into_dyn()
	}
}

impl<T: IntoTensorElementType + Clone + Debug, const N: usize> Index<[i64; N]> for Tensor<T> {
	type Output = T;
	fn index(&self, index: [i64; N]) -> &Self::Output {
		// Interestingly, the `TensorAt` API doesn't check if the tensor is on CPU, so we have to perform the check ourselves.
		if !self.memory_info().is_cpu_accessible() {
			panic!("Cannot directly index a tensor which is not allocated on the CPU.");
		}

		let mut out: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr().cast_mut(), index.as_ptr(), N, &mut out).expect("Failed to index tensor")];
		unsafe { &*out.cast::<T>() }
	}
}
impl<T: IntoTensorElementType + Clone + Debug, const N: usize> IndexMut<[i64; N]> for Tensor<T> {
	fn index_mut(&mut self, index: [i64; N]) -> &mut Self::Output {
		if !self.memory_info().is_cpu_accessible() {
			panic!("Cannot directly index a tensor which is not allocated on the CPU.");
		}

		let mut out: *mut ort_sys::c_void = ptr::null_mut();
		ortsys![unsafe TensorAt(self.ptr_mut(), index.as_ptr(), N, &mut out).expect("Failed to index tensor")];
		unsafe { &mut *out.cast::<T>() }
	}
}

#[cfg(test)]
mod tests {
	use alloc::sync::Arc;

	#[cfg(feature = "ndarray")]
	use ndarray::{ArcArray1, Array1, CowArray};

	use super::Tensor;
	use crate::{
		memory::Allocator,
		tensor::{Shape, SymbolicDimensions, TensorElementType},
		value::{TensorRef, ValueType}
	};

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_value() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];
		let value = Tensor::from_array(Array1::from_vec(v.clone()))?;
		assert_eq!(value.dtype().tensor_type(), Some(TensorElementType::Float32));
		assert_eq!(
			value.dtype(),
			&ValueType::Tensor {
				ty: TensorElementType::Float32,
				shape: Shape::new([v.len() as i64]),
				dimension_symbols: SymbolicDimensions::empty(1)
			}
		);

		let (shape, data) = value.extract_tensor();
		assert_eq!(&**shape, [v.len() as i64]);
		assert_eq!(data, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_tensor_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc1 = ArcArray1::from_vec(v.clone());
		let arc2 = ArcArray1::clone(&arc1);
		let value = TensorRef::from_array_view(arc2.clone())?;
		drop((arc1, arc2));

		assert_eq!(value.extract_tensor().1, &v);

		let cow = CowArray::from(Array1::from_vec(v.clone()));
		let value = TensorRef::from_array_view(&cow)?;
		assert_eq!(value.extract_tensor().1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_lifetimes() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let arc = Arc::new(v.clone().into_boxed_slice());
		let shape = vec![v.len() as i64];
		let value = TensorRef::from_array_view((shape, Arc::clone(&arc)))?;
		drop(arc);
		assert_eq!(value.try_extract_tensor::<f32>()?.1, &v);

		Ok(())
	}

	#[test]
	#[cfg(feature = "ndarray")]
	fn test_string_tensor_ndarray() -> crate::Result<()> {
		let v = Array1::from_vec(vec!["hello world".to_string(), "こんにちは世界".to_string()]);

		let value = Tensor::from_string_array(v.view())?;
		let extracted = value.try_extract_string_array()?;
		assert_eq!(extracted, v.into_dyn());

		Ok(())
	}

	#[test]
	fn test_string_tensor_raw() -> crate::Result<()> {
		let v = vec!["hello world".to_string(), "こんにちは世界".to_string()];

		let value = Tensor::from_string_array((vec![v.len() as i64], &*v))?;
		let (extracted_shape, extracted_view) = value.try_extract_strings()?;
		assert_eq!(&**extracted_shape, [v.len() as i64]);
		assert_eq!(extracted_view, v);

		Ok(())
	}

	#[test]
	fn test_tensor_raw_inputs() -> crate::Result<()> {
		let v: Vec<f32> = vec![1., 2., 3., 4., 5.];

		let shape = [v.len()];
		let value_arc_box = TensorRef::from_array_view((shape, Arc::new(v.clone().into_boxed_slice())))?;
		let value_box = Tensor::from_array((shape, v.clone().into_boxed_slice()))?;
		let value_vec = Tensor::from_array((shape, v.clone()))?;
		let value_slice = TensorRef::from_array_view((shape, &v[..]))?;

		assert_eq!(value_arc_box.extract_tensor().1, &v);
		assert_eq!(value_box.extract_tensor().1, &v);
		assert_eq!(value_vec.extract_tensor().1, &v);
		assert_eq!(value_slice.extract_tensor().1, &v);

		Ok(())
	}

	#[test]
	fn test_tensor_index() -> crate::Result<()> {
		let mut tensor = Tensor::new(&Allocator::default(), Shape::new([1, 3, 224, 224]))?;

		tensor[[0, 2, 42, 42]] = 1.0;
		assert_eq!(tensor[[0, 2, 42, 42]], 1.0);

		for y in 0..224 {
			for x in 0..224 {
				tensor[[0, 1, y, x]] = -1.0;
			}
		}
		assert_eq!(tensor[[0, 1, 0, 0]], -1.0);
		assert_eq!(tensor[[0, 1, 223, 223]], -1.0);

		assert_eq!(tensor[[0, 2, 42, 42]], 1.0);

		Ok(())
	}
}
