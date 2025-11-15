#[allow(unused)]
use ort::ep::*;

pub fn init() -> ort::Result<()> {
	#[cfg(feature = "backend-candle")]
	ort::set_api(ort_candle::api());
	#[cfg(feature = "backend-tract")]
	ort::set_api(ort_tract::api());

	#[cfg(all(not(feature = "backend-candle"), not(feature = "backend-tract")))]
	ort::init()
		.with_execution_providers([
			#[cfg(feature = "tensorrt")]
			TensorRT::default().build(),
			#[cfg(feature = "cuda")]
			CUDA::default().build(),
			#[cfg(feature = "onednn")]
			OneDNN::default().build(),
			#[cfg(feature = "acl")]
			ACL::default().build(),
			#[cfg(feature = "openvino")]
			OpenVINO::default().build(),
			#[cfg(feature = "coreml")]
			CoreML::default().build(),
			#[cfg(feature = "rocm")]
			ROCm::default().build(),
			#[cfg(feature = "cann")]
			CANN::default().build(),
			#[cfg(feature = "directml")]
			DirectML::default().build(),
			#[cfg(feature = "tvm")]
			TVM::default().build(),
			#[cfg(feature = "nnapi")]
			NNAPI::default().build(),
			#[cfg(feature = "qnn")]
			QNN::default().build(),
			#[cfg(feature = "xnnpack")]
			XNNPACK::default().build(),
			#[cfg(feature = "armnn")]
			ArmNN::default().build(),
			#[cfg(feature = "migraphx")]
			MIGraphX::default().build(),
			#[cfg(feature = "vitis")]
			VitisAI::default().build(),
			#[cfg(feature = "rknpu")]
			RKNPU::default().build(),
			#[cfg(feature = "webgpu")]
			WebGPU::default().build()
		])
		.commit();

	Ok(())
}
