use alloc::string::String;
use core::fmt;

use super::{ExecutionProviderOptions, SimpleExecutionProvider};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredLayout {
	NCHW,
	NHWC
}

impl fmt::Display for PreferredLayout {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			PreferredLayout::NCHW => "NCHW",
			PreferredLayout::NHWC => "NHWC"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DawnBackendType {
	Vulkan,
	D3D12
}

impl fmt::Display for DawnBackendType {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			DawnBackendType::Vulkan => "Vulkan",
			DawnBackendType::D3D12 => "D3D12"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferCacheMode {
	Disabled,
	LazyRelease,
	Simple,
	Bucket
}

impl fmt::Display for BufferCacheMode {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			BufferCacheMode::Disabled => "disabled",
			BufferCacheMode::LazyRelease => "lazyRelease",
			BufferCacheMode::Simple => "simple",
			BufferCacheMode::Bucket => "bucket"
		})
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationMode {
	Disabled,
	WgpuOnly,
	Basic,
	Full
}

impl fmt::Display for ValidationMode {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.write_str(match self {
			ValidationMode::Disabled => "disabled",
			ValidationMode::WgpuOnly => "wgpuOnly",
			ValidationMode::Basic => "basic",
			ValidationMode::Full => "full"
		})
	}
}

#[derive(Debug, Default, Clone)]
pub struct WebGPU(ExecutionProviderOptions);

super::impl_ep!(arbitrary; WebGPU);

impl WebGPU {
	super::define_options! {
		pub fn with_preferred_layout(mut self, layout: PreferredLayout) -> Self = "preferredLayout";

		pub fn with_enable_graph_capture(mut self, enable: bool) -> Self = "enableGraphCapture";

		pub fn with_dawn_proc_table(mut self, table: String) -> Self = "dawnProcTable";

		pub fn with_dawn_backend_type(mut self, backend_type: DawnBackendType) -> Self = "dawnBackendType";

		pub fn with_device_id(mut self, id: i32) -> Self = "deviceId";

		pub fn with_storage_buffer_cache_mode(mut self, mode: BufferCacheMode) -> Self = "storageBufferCacheMode";

		pub fn with_uniform_buffer_cache_mode(mut self, mode: BufferCacheMode) -> Self = "uniformBufferCacheMode";

		pub fn with_query_resolve_buffer_cache_mode(mut self, mode: BufferCacheMode) -> Self = "queryResolveBufferCacheMode";

		pub fn with_default_buffer_cache_mode(mut self, mode: BufferCacheMode) -> Self = "defaultBufferCacheMode";

		pub fn with_validation_mode(mut self, mode: ValidationMode) -> Self = "validationMode";

		pub fn with_force_cpu_node_names(mut self, names: String) -> Self = "forceCpuNodeNames";

		pub fn with_enable_pix_capture(mut self, enable: bool) -> Self = "enablePIXCapture";
	}
}

impl SimpleExecutionProvider for WebGPU {
	const CANONICAL_NAME: &'static str = "WebGpuExecutionProvider";
	const SHORT_NAME: &'static str = "WebGPU";

	fn options(&self) -> &ExecutionProviderOptions {
		&self.0
	}
}
