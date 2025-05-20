//! ONNX Runtime doesn't (currently) expose an API for inter-device copies, so we instead use a dummy model to copy the
//! tensor & `IoBinding` to configure where the copy ends up.

use alloc::{format, string::ToString};
use core::ops::{Deref, DerefMut};

use super::DefiniteTensorValueTypeMarker;
use crate::{
	Error, OnceLock, Result, execution_providers as ep,
	io_binding::IoBinding,
	memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
	session::{NoSelectedOutputs, RunOptions, Session, builder::GraphOptimizationLevel},
	util::{MiniMap, Mutex, MutexGuard},
	value::Value
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct IdentitySessionKey {
	src_device: AllocationDevice,
	src_device_id: i32,
	target_device: AllocationDevice,
	target_device_id: i32,
	dtype: ort_sys::ONNXTensorElementDataType
}

struct IdentitySession {
	session: Session,
	binding: IoBinding
}

/// A simple graph in `.ort` format with a single `Identity` node between the input & output.
static IDENTITY_MODEL: &[u8] = include_bytes!("./identity.ort");
static SESSIONS: OnceLock<Mutex<MiniMap<IdentitySessionKey, IdentitySession>>> = OnceLock::new();
/// `RunOptions` with [`RunOptions::disable_device_sync`], shared across `to_async()` calls to reduce allocations.
static IDENTITY_RUN_OPTIONS: OnceLock<RunOptions<NoSelectedOutputs>> = OnceLock::new();

fn ep_for_device(device: AllocationDevice, device_id: i32) -> Result<ep::ExecutionProviderDispatch> {
	Ok(match device {
		AllocationDevice::CPU => ep::CPUExecutionProvider::default().with_arena_allocator(false).build(),
		AllocationDevice::CUDA | AllocationDevice::CUDA_PINNED => ep::CUDAExecutionProvider::default()
			.with_device_id(device_id)
			.with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
			.with_conv_max_workspace(false)
			.with_conv_algorithm_search(ep::cuda::CuDNNConvAlgorithmSearch::Default)
			.build(),
		AllocationDevice::DIRECTML => ep::DirectMLExecutionProvider::default().with_device_id(device_id).build(),
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
		_ => return Err(crate::Error::new("Unsupported allocation device {device} for tensor copy target"))
	})
}

impl<Type: DefiniteTensorValueTypeMarker + ?Sized> Value<Type> {
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
		self.copy_to_inner(device, device_id, |identity_session| {
			let target_memory_info = MemoryInfo::new(device, device_id, AllocatorType::Device, MemoryType::Default)?;
			identity_session.binding.bind_output_to_device("output", &target_memory_info)?;

			let output = identity_session
				.session
				.run_binding(&identity_session.binding)?
				.remove("output")
				.expect("identity model should have single output");
			Ok(unsafe { output.transmute_type() })
		})
	}

	/// Asynchronously copies the contents of this tensor to another device.
	///
	/// Unlike [`Tensor::to`][crate::value::Tensor::to], the device's stream will *not* be synchronized (like via
	/// `cudaStreamSynchronize`); thus this function is most useful for host-to-device transfers.
	///
	/// ```
	/// # use ort::{memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType}, session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// # let tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	/// # if false {
	/// let cuda_tensor = tensor.to_async(AllocationDevice::CUDA, 0)?;
	/// // pass to other CUDA code, or to session input
	/// # }
	/// # Ok(())
	/// # }
	/// ```
	pub fn to_async(&self, device: AllocationDevice, device_id: i32) -> Result<Value<Type>> {
		self.copy_to_inner(device, device_id, |identity_session| {
			let target_memory_info = MemoryInfo::new(device, device_id, AllocatorType::Device, MemoryType::Default)?;
			identity_session.binding.bind_output_to_device("output", &target_memory_info)?;

			let options = IDENTITY_RUN_OPTIONS.get_or_try_init(|| -> Result<RunOptions> {
				let mut options = RunOptions::new()?;
				options.disable_device_sync()?;
				Ok(options)
			})?;
			let output = identity_session
				.session
				.run_binding_with_options(&identity_session.binding, options)?
				.remove("output")
				.expect("identity model should have single output");
			Ok(unsafe { output.transmute_type() })
		})
	}

	/// Copies the contents of this tensor to another tensor potentially residing on a separate device.
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
	/// let mut cpu_tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;;
	///
	/// cuda_tensor.copy_into(&mut cpu_tensor)?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn copy_into(&self, target: &mut Value<Type>) -> Result<()> {
		if self.dtype() != target.dtype() {
			return Err(Error::new("target data type does not match source data type"));
		} else if self.shape() != target.shape() {
			return Err(Error::new("target shape does not match source shape"));
		}

		let target_memory_info = target.memory_info();
		self.copy_to_inner(target_memory_info.allocation_device(), target_memory_info.device_id(), |identity_session| {
			unsafe { identity_session.binding.bind_output_mut("output", target) }?;
			identity_session.session.run_binding(&identity_session.binding)?;
			Ok(())
		})
	}

	/// Asynchronously copies the contents of this tensor to another tensor.
	///
	/// Unlike [`Tensor::copy_into`][crate::value::Tensor::copy_into], the device's stream will *not* be synchronized
	/// (like via `cudaStreamSynchronize`); thus this function is most useful for host-to-device transfers.
	///
	/// ```
	/// # use ort::{memory::{Allocator, AllocatorType, AllocationDevice, MemoryInfo, MemoryType}, session::Session, value::Tensor};
	/// # fn main() -> ort::Result<()> {
	/// let cpu_tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;;
	/// # if false {
	/// # let session = Session::builder()?.commit_from_file("tests/data/upsample.onnx")?;
	/// let cuda_allocator = Allocator::new(
	/// 	&session,
	/// 	MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?
	/// )?;
	/// let mut cuda_tensor = Tensor::<f32>::new(&cuda_allocator, [1_usize, 3, 224, 224])?;
	/// # }
	/// # let mut cuda_tensor = Tensor::<f32>::new(&Allocator::default(), [1_usize, 3, 224, 224])?;
	///
	/// cpu_tensor.copy_into_async(&mut cuda_tensor)?;
	/// # Ok(())
	/// # }
	/// ```
	pub fn copy_into_async(&self, target: &mut Value<Type>) -> Result<()> {
		if self.dtype() != target.dtype() {
			return Err(Error::new("target data type does not match source data type"));
		} else if self.shape() != target.shape() {
			return Err(Error::new("target shape does not match source shape"));
		}

		let target_memory_info = target.memory_info();
		self.copy_to_inner(target_memory_info.allocation_device(), target_memory_info.device_id(), |identity_session| {
			unsafe { identity_session.binding.bind_output_mut("output", target) }?;
			let options = IDENTITY_RUN_OPTIONS.get_or_try_init(|| -> Result<RunOptions> {
				let mut options = RunOptions::new()?;
				options.disable_device_sync()?;
				Ok(options)
			})?;
			identity_session.session.run_binding_with_options(&identity_session.binding, options)?;
			Ok(())
		})
	}

	fn copy_to_inner<F, T>(&self, device: AllocationDevice, device_id: i32, runner: F) -> Result<T>
	where
		F: FnOnce(&mut IdentitySession) -> Result<T>
	{
		let source_memory_info = self.memory_info();
		let tensor_type = ort_sys::ONNXTensorElementDataType::from(*self.data_type());

		let mut identity_session = IdentitySessionHandle::new(source_memory_info, device, device_id, tensor_type)?;
		identity_session.binding.bind_input("input", self)?;
		runner(&mut identity_session)
	}
}

impl<Type: DefiniteTensorValueTypeMarker + ?Sized> Clone for Value<Type> {
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

struct IdentitySessionHandle {
	inner: &'static mut IdentitySession,
	_guard: MutexGuard<'static, MiniMap<IdentitySessionKey, IdentitySession>>
}

impl IdentitySessionHandle {
	fn new(
		source_memory_info: &MemoryInfo,
		target_device: AllocationDevice,
		target_device_id: i32,
		tensor_type: ort_sys::ONNXTensorElementDataType
	) -> Result<Self> {
		let session_key = IdentitySessionKey {
			src_device: source_memory_info.allocation_device(),
			src_device_id: source_memory_info.device_id(),
			target_device,
			target_device_id,
			dtype: tensor_type
		};
		let mut sessions = SESSIONS.get_or_init(|| Mutex::new(MiniMap::new())).lock();
		let identity_session = match sessions.get_mut(&session_key) {
			Some(entry) => entry,
			None => {
				let mut model_bytes = IDENTITY_MODEL.to_vec();
				// Override the expected element type of the input & output nodes, respectively.
				model_bytes[544] = tensor_type as u8;
				model_bytes[604] = tensor_type as u8;

				let (source_ep, target_ep) = (
					// We enable `.error_on_failure()` here since `IoBinding::bind_output_to_device` will silently fall back to binding to CPU if the target
					// device doesn't have an EP registered.
					ep_for_device(source_memory_info.allocation_device(), source_memory_info.device_id())?.error_on_failure(),
					ep_for_device(target_device, target_device_id)?.error_on_failure()
				);

				let mut builder = Session::builder()?
					.with_optimization_level(GraphOptimizationLevel::Disable)?
					// since these sessions are persistent for the lifetime of the program, keep them as lean as possible
					// by disabling threading & memory optimizations (there's only 1 operation in the graph anyway)
					.with_intra_threads(1)?
					.with_inter_threads(1)?
					.with_inter_op_spinning(false)?
					.with_intra_op_spinning(false)?
					.with_memory_pattern(false)?
					.with_allocator(MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?)?
					.with_no_environment_execution_providers()?;
				// Avoid registering the same EP twice, since that's an error.
				if source_ep.inner.name() != target_ep.inner.name() {
					builder = builder.with_execution_providers([source_ep, target_ep])?;
				} else {
					builder = builder.with_execution_providers([source_ep])?;
				}

				let session = builder.commit_from_memory(&model_bytes)?;
				let binding = session.create_binding()?;
				sessions.insert(session_key.clone(), IdentitySession { session, binding });
				sessions.get_mut(&session_key).expect("insert should have worked")
			}
		};

		Ok(Self {
			inner: unsafe { core::mem::transmute::<&mut IdentitySession, &'static mut IdentitySession>(identity_session) },
			_guard: sessions
		})
	}
}

impl Deref for IdentitySessionHandle {
	type Target = IdentitySession;
	fn deref(&self) -> &Self::Target {
		self.inner
	}
}

impl DerefMut for IdentitySessionHandle {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.inner
	}
}

#[cfg(test)]
mod tests {
	use crate::value::Tensor;

	#[test]
	#[cfg(feature = "cuda")]
	fn test_clone_tensor() -> crate::Result<()> {
		let tensor = Tensor::<f32>::from_array(([1, 5], vec![2.167892, 333., 1.0, -0.0, f32::EPSILON]))?;
		let clone = tensor.clone();
		assert_eq!(tensor.extract_tensor(), clone.extract_tensor());
		Ok(())
	}

	#[test]
	#[cfg(feature = "cuda")]
	fn test_copy_cuda() -> crate::Result<()> {
		use crate::memory::AllocationDevice;

		let tensor = Tensor::<f32>::from_array(([1, 5], vec![2.167892, 333., 1.0, -0.0, f32::EPSILON]))?;

		let cuda_tensor = tensor.to(AllocationDevice::CUDA, 0)?;
		let memory = cuda_tensor.memory_info();
		assert_eq!(memory.allocation_device(), AllocationDevice::CUDA);
		assert!(!memory.is_cpu_accessible());

		let cpu_tensor = cuda_tensor.to(AllocationDevice::CPU, 0)?;
		assert!(cpu_tensor.memory_info().is_cpu_accessible());

		assert_eq!(tensor.extract_tensor(), cpu_tensor.extract_tensor());

		Ok(())
	}

	#[test]
	#[cfg(feature = "cuda")]
	fn test_copy_cuda_async() -> crate::Result<()> {
		use crate::memory::AllocationDevice;

		let tensor = Tensor::<f32>::from_array(([1, 5], vec![2.167892, 333., 1.0, -0.0, f32::EPSILON]))?;

		let cuda_tensor = tensor.to_async(AllocationDevice::CUDA, 0)?;
		let cpu_tensor = cuda_tensor.to(AllocationDevice::CPU, 0)?;
		assert_eq!(tensor.extract_tensor(), cpu_tensor.extract_tensor());

		Ok(())
	}

	#[test]
	#[cfg(feature = "cuda")]
	fn test_copy_into_cuda() -> crate::Result<()> {
		use crate::{
			execution_providers::CUDAExecutionProvider,
			memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
			session::Session
		};

		let dummy_session = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.commit_from_file("tests/data/upsample.ort")?;

		let allocator = Allocator::new(&dummy_session, MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?)?;
		let tensor = Tensor::<f32>::from_array(([1, 5], vec![2.167892, 333., 1.0, -0.0, f32::EPSILON]))?;
		let mut cuda_tensor = Tensor::<f32>::new(&allocator, [1_i64, 5])?;

		tensor.copy_into(&mut cuda_tensor)?;
		let cpu_tensor = cuda_tensor.to(AllocationDevice::CPU, 0)?;
		assert_eq!(tensor.extract_tensor(), cpu_tensor.extract_tensor());

		Ok(())
	}

	#[test]
	#[cfg(feature = "cuda")]
	fn test_copy_into_async_cuda() -> crate::Result<()> {
		use crate::{
			execution_providers::CUDAExecutionProvider,
			memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
			session::Session
		};

		let dummy_session = Session::builder()?
			.with_execution_providers([CUDAExecutionProvider::default().build()])?
			.commit_from_file("tests/data/upsample.ort")?;

		let allocator = Allocator::new(&dummy_session, MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default)?)?;
		let tensor = Tensor::<f32>::from_array(([1, 5], vec![2.167892, 333., 1.0, -0.0, f32::EPSILON]))?;
		let mut cuda_tensor = Tensor::<f32>::new(&allocator, [1_i64, 5])?;

		tensor.copy_into_async(&mut cuda_tensor)?;
		let cpu_tensor = cuda_tensor.to(AllocationDevice::CPU, 0)?;
		assert_eq!(tensor.extract_tensor(), cpu_tensor.extract_tensor());

		Ok(())
	}
}
