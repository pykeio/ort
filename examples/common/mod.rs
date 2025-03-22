#[allow(unused)]
use ort::execution_providers::*;

pub fn init() -> ort::Result<()> {
	#[cfg(feature = "backend-candle")]
	ort::set_api(ort_candle::api());
	#[cfg(feature = "backend-tract")]
	ort::set_api(ort_tract::api());

	#[cfg(all(not(feature = "backend-candle"), not(feature = "backend-tract")))]
	ort::init()
		.with_execution_providers([
			#[cfg(feature = "tensorrt")]
			TensorRTExecutionProvider::default().build(),
			#[cfg(feature = "cuda")]
			CUDAExecutionProvider::default().build(),
			#[cfg(feature = "onednn")]
			OneDNNExecutionProvider::default().build(),
			#[cfg(feature = "acl")]
			ACLExecutionProvider::default().build(),
			#[cfg(feature = "openvino")]
			OpenVINOExecutionProvider::default().build(),
			#[cfg(feature = "coreml")]
			CoreMLExecutionProvider::default().build(),
			#[cfg(feature = "rocm")]
			ROCmExecutionProvider::default().build(),
			#[cfg(feature = "cann")]
			CANNExecutionProvider::default().build(),
			#[cfg(feature = "directml")]
			DirectMLExecutionProvider::default().build(),
			#[cfg(feature = "tvm")]
			TVMExecutionProvider::default().build(),
			#[cfg(feature = "nnapi")]
			NNAPIExecutionProvider::default().build(),
			#[cfg(feature = "qnn")]
			QNNExecutionProvider::default().build(),
			#[cfg(feature = "xnnpack")]
			XNNPACKExecutionProvider::default().build(),
			#[cfg(feature = "armnn")]
			ArmNNExecutionProvider::default().build(),
			#[cfg(feature = "migraphx")]
			MIGraphXExecutionProvider::default().build(),
			#[cfg(feature = "vitis")]
			VitisAIExecutionProvider::default().build(),
			#[cfg(feature = "rknpu")]
			RKNPUExecutionProvider::default().build(),
			#[cfg(feature = "webgpu")]
			WebGPUExecutionProvider::default().build()
		])
		.commit()?;

	Ok(())
}
