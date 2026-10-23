#![allow(non_snake_case)]

use ort_sys::*;

mod env;
mod memory;
mod run_options;
mod session;
mod session_options;
mod tensor;
mod type_info;

use self::{env::*, memory::*, run_options::*, session::*, session_options::*, tensor::*, type_info::*};

unsafe extern "system" fn GetBuildInfoString() -> *const c_char {
	concat!("ORT Build Info: backend=ort-web, version=", env!("CARGO_PKG_VERSION"), ", with <3\0")
		.as_ptr()
		.cast()
}

pub const fn api() -> OrtApi {
	OrtApi {
		CreateEnv,
		CreateEnvWithCustomLogger,
		EnableTelemetryEvents,
		DisableTelemetryEvents,
		CreateSession,
		CreateSessionFromArray,
		Run,
		RunAsync,
		CreateSessionOptions,
		CloneSessionOptions,
		SetSessionGraphOptimizationLevel,
		SetIntraOpNumThreads,
		SetInterOpNumThreads,
		SetSessionExecutionMode,
		EnableMemPattern,
		DisableMemPattern,
		EnableCpuMemArena,
		DisableCpuMemArena,
		EnableProfiling,
		DisableProfiling,
		SetSessionLogId,
		SetSessionLogSeverityLevel,
		SetSessionLogVerbosityLevel,
		AddFreeDimensionOverrideByName,
		AddSessionConfigEntry,
		SessionGetInputCount,
		SessionGetOutputCount,
		SessionGetOverridableInitializerCount,
		SessionGetInputTypeInfo,
		SessionGetOutputTypeInfo,
		SessionGetInputName,
		SessionGetOutputName,
		SessionGetMemoryInfoForInputs,
		SessionGetMemoryInfoForOutputs,
		CreateTensorAsOrtValue,
		CreateTensorWithDataAsOrtValue,
		IsTensor,
		HasValue,
		GetTensorMutableData,
		GetTensorData,
		GetTensorSizeInBytes,
		TensorAt,
		FillStringTensor,
		GetStringTensorDataLength,
		GetStringTensorContent,
		CastTypeInfoToTensorInfo,
		GetOnnxTypeFromTypeInfo,
		CreateTensorTypeAndShapeInfo,
		SetTensorElementType,
		SetDimensions,
		SetSymbolicDimensions,
		GetTensorElementType,
		GetDimensionsCount,
		GetDimensions,
		GetSymbolicDimensions,
		GetTensorShapeElementCount,
		TensorTypeAndShape_HasShape,
		GetTensorTypeAndShape,
		GetTypeInfo,
		GetValueType,
		CreateMemoryInfo,
		CreateCpuMemoryInfo,
		CompareMemoryInfo,
		MemoryInfoGetName,
		MemoryInfoGetId,
		MemoryInfoGetMemType,
		MemoryInfoGetType,
		MemoryInfoGetDeviceType,
		MemoryInfoGetDeviceMemType,
		AllocatorAlloc,
		AllocatorFree,
		AllocatorGetInfo,
		GetAllocatorWithDefaultOptions,
		ReleaseEnv,
		ReleaseMemoryInfo,
		ReleaseSession,
		ReleaseValue,
		ReleaseTypeInfo,
		ReleaseTensorTypeAndShapeInfo,
		ReleaseSessionOptions,
		CreateAllocator,
		ReleaseAllocator,
		GetTensorMemoryInfo,
		GetBuildInfoString,
		CreateRunOptions,
		ReleaseRunOptions,
		RunOptionsSetRunLogVerbosityLevel,
		RunOptionsSetRunLogSeverityLevel,
		RunOptionsGetRunLogVerbosityLevel,
		RunOptionsGetRunLogSeverityLevel,
		RunOptionsSetRunTag,
		RunOptionsGetRunTag,
		RunOptionsSetTerminate,
		RunOptionsUnsetTerminate,
		AddRunConfigEntry,
		SessionOptionsAppendExecutionProvider,
		GetTensorElementTypeAndShapeDataReference,
		..ort_sys::stub::api()
	}
}
