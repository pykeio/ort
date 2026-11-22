#![allow(non_snake_case)]

use ort_sys::*;

mod env;
mod io_binding;
mod memory;
mod metadata;
mod run_options;
mod session;
mod session_options;
mod tensor;
mod type_info;

use self::{env::*, io_binding::*, memory::*, metadata::*, run_options::*, session::*, session_options::*, tensor::*, type_info::*};

unsafe extern "system" fn GetBuildInfoString() -> *const c_char {
	concat!("ORT Build Info: backend=ort-candle, version=", env!("CARGO_PKG_VERSION"), "\0")
		.as_ptr()
		.cast()
}

pub const fn api() -> OrtApi {
	OrtApi {
		CreateEnv,
		CreateEnvWithCustomLogger,
		CreateEnvWithGlobalThreadPools,
		CreateEnvWithCustomLoggerAndGlobalThreadPools,
		CreateThreadingOptions,
		ReleaseThreadingOptions,
		CreateSession,
		CreateSessionFromArray,
		CreatePrepackedWeightsContainer,
		ReleasePrepackedWeightsContainer,
		CreateSessionWithPrepackedWeightsContainer,
		CreateSessionFromArrayWithPrepackedWeightsContainer,
		Run,
		CreateSessionOptions,
		CloneSessionOptions,
		AddSessionConfigEntry,
		AddInitializer,
		AddExternalInitializers,
		AddFreeDimensionOverride,
		AddFreeDimensionOverrideByName,
		SessionGetInputCount,
		SessionGetOutputCount,
		SessionGetOverridableInitializerCount,
		SessionGetInputTypeInfo,
		SessionGetOutputTypeInfo,
		SessionGetInputName,
		SessionGetOutputName,
		SessionGetMemoryInfoForInputs,
		SessionGetMemoryInfoForOutputs,
		SessionGetModelMetadata,
		ReleaseModelMetadata,
		ModelMetadataGetProducerName,
		ModelMetadataGetGraphName,
		ModelMetadataGetDomain,
		ModelMetadataGetDescription,
		ModelMetadataGetGraphDescription,
		ModelMetadataGetVersion,
		ModelMetadataLookupCustomMetadataMap,
		ModelMetadataGetCustomMetadataMapKeys,
		CreateRunOptions,
		ReleaseRunOptions,
		RunOptionsSetRunLogVerbosityLevel,
		RunOptionsSetRunLogSeverityLevel,
		RunOptionsSetRunTag,
		RunOptionsGetRunLogVerbosityLevel,
		RunOptionsGetRunLogSeverityLevel,
		RunOptionsGetRunTag,
		RunOptionsSetTerminate,
		RunOptionsUnsetTerminate,
		AddRunConfigEntry,
		CreateIoBinding,
		ReleaseIoBinding,
		BindInput,
		BindOutput,
		BindOutputToDevice,
		ClearBoundInputs,
		ClearBoundOutputs,
		RunWithBinding,
		GetBoundOutputNames,
		GetBoundOutputValues,
		CreateTensorAsOrtValue,
		CreateTensorWithDataAsOrtValue,
		CreateTensorWithDataAndDeleterAsOrtValue,
		IsTensor,
		HasValue,
		GetTensorMutableData,
		GetTensorData,
		GetTensorSizeInBytes,
		TensorAt,
		CopyTensors,
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
		ReleaseMemoryInfo,
		AllocatorAlloc,
		AllocatorFree,
		AllocatorGetInfo,
		GetAllocatorWithDefaultOptions,
		CreateAllocator,
		ReleaseAllocator,
		GetTensorMemoryInfo,
		ReleaseEnv,
		ReleaseSession,
		ReleaseValue,
		ReleaseTypeInfo,
		ReleaseTensorTypeAndShapeInfo,
		ReleaseSessionOptions,
		GetBuildInfoString,
		GetTensorElementTypeAndShapeDataReference,
		..ort_sys::stub::api()
	}
}
