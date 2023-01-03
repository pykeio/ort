#![allow(unused_imports)]

use std::{collections::HashMap, ffi::CString, os::raw::c_char};

use crate::{error::status_to_result, ortsys, sys, OrtApiError, OrtResult};

extern "C" {
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_CPU(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "acl")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_ACL(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
	#[cfg(feature = "onednn")]
	pub(crate) fn OrtSessionOptionsAppendExecutionProvider_Dnnl(options: *mut sys::OrtSessionOptions, use_arena: std::os::raw::c_int) -> sys::OrtStatusPtr;
}

#[derive(Debug, Clone)]
pub struct ExecutionProvider {
	provider: String,
	options: HashMap<String, String>
}

macro_rules! ep_providers {
	($($fn_name:ident = $name:expr),*) => {
		$(
			/// Creates a new `
			#[doc = $name]
			#[doc = "` configuration object."]
			pub fn $fn_name() -> Self {
				Self::new($name)
			}
		)*
	}
}

macro_rules! ep_if_available {
	($($fn_name:ident($original:ident): $name:expr),*) => {
		$(
			/// Creates a new
			#[doc = $name]
			#[doc = " execution provider if available, otherwise falling back to CPU."]
			pub fn $fn_name() -> Self {
				let o = Self::$original();
				if o.is_available() { o } else { Self::cpu() }
			}
		)*
	}
}

macro_rules! ep_options {
	($(
		$(#[$meta:meta])*
		pub fn $fn_name:ident($opt_type:ty) = $option_name:ident;
	)*) => {
		$(
			$(#[$meta])*
			pub fn $fn_name(mut self, v: $opt_type) -> Self {
				self = self.with(stringify!($option_name), v.to_string());
				self
			}
		)*
	}
}

impl ExecutionProvider {
	pub fn new(provider: impl Into<String>) -> Self {
		Self {
			provider: provider.into(),
			options: HashMap::new()
		}
	}

	ep_providers! {
		acl = "AclExecutionProvider",
		dnnl = "DnnlExecutionProvider",
		onednn = "DnnlExecutionProvider",
		cuda = "CUDAExecutionProvider",
		tensorrt = "TensorRTExecutionProvider",
		cpu = "CPUExecutionProvider"
	}

	pub fn is_available(&self) -> bool {
		let mut providers: *mut *mut c_char = std::ptr::null_mut();
		let mut num_providers = 0;
		if status_to_result(ortsys![unsafe GetAvailableProviders(&mut providers, &mut num_providers)]).is_err() {
			return false;
		}

		for i in 0..num_providers {
			let avail = unsafe { std::ffi::CStr::from_ptr(*providers.offset(i as isize)) }
				.to_string_lossy()
				.into_owned();
			if self.provider == avail {
				return true;
			}
		}

		false
	}

	ep_if_available! {
		tensorrt_if_available(tensorrt): "TensorRT",
		cuda_if_available(cuda): "CUDA",
		acl_if_available(acl): "ACL",
		dnnl_if_available(dnnl): "oneDNN",
		onednn_if_available(dnnl): "oneDNN"
	}

	/// Configure this execution provider with the given option.
	pub fn with(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
		self.options.insert(k.into(), v.into());
		self
	}

	ep_options! {
		/// Whether or not to use CPU arena allocator.
		pub fn with_use_arena(bool) = use_arena;
	}
}

#[tracing::instrument(skip_all)]
pub(crate) fn apply_execution_providers(options: *mut sys::OrtSessionOptions, execution_providers: impl AsRef<[ExecutionProvider]>) {
	let status_to_result_and_log = |ep: &'static str, status: *mut sys::OrtStatus| {
		let result = status_to_result(status);
		tracing::debug!("{ep} execution provider registration {status:?}");
		result
	};
	for ep in execution_providers.as_ref() {
		let init_args = ep.options.clone();
		match ep.provider.as_str() {
			"CPUExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map(|s| s.parse::<bool>().unwrap_or(false)).unwrap_or(false);
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_CPU(options, use_arena.into()) };
				if status_to_result_and_log("CPU", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "cuda")]
			"CUDAExecutionProvider" => {
				let mut cuda_options: *mut sys::OrtCUDAProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result_and_log("CUDA", ortsys![unsafe CreateCUDAProviderOptions(&mut cuda_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(k.as_str()).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(v.as_str()).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateCUDAProviderOptions(cuda_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result_and_log("CUDA", status).is_err() {
					ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda_options)];
				ortsys![unsafe ReleaseCUDAProviderOptions(cuda_options)];
				if status_to_result_and_log("CUDA", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "tensorrt")]
			"TensorRTExecutionProvider" => {
				let mut tensorrt_options: *mut sys::OrtTensorRTProviderOptionsV2 = std::ptr::null_mut();
				if status_to_result_and_log("TensorRT", ortsys![unsafe CreateTensorRTProviderOptions(&mut tensorrt_options)]).is_err() {
					continue; // next EP
				}
				let keys: Vec<CString> = init_args.keys().map(|k| CString::new(k.as_str()).unwrap()).collect();
				let values: Vec<CString> = init_args.values().map(|v| CString::new(v.as_str()).unwrap()).collect();
				assert_eq!(keys.len(), values.len()); // sanity check
				let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
				let value_ptrs: Vec<*const c_char> = values.iter().map(|v| v.as_ptr()).collect();
				let status = ortsys![unsafe UpdateTensorRTProviderOptions(tensorrt_options, key_ptrs.as_ptr(), value_ptrs.as_ptr(), keys.len())];
				if status_to_result_and_log("TensorRT", status).is_err() {
					ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
					continue; // next EP
				}
				let status = ortsys![unsafe SessionOptionsAppendExecutionProvider_TensorRT_V2(options, tensorrt_options)];
				ortsys![unsafe ReleaseTensorRTProviderOptions(tensorrt_options)];
				if status_to_result_and_log("TensorRT", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "acl")]
			"AclExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map(|s| s.parse::<bool>().unwrap_or(false)).unwrap_or(false);
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_ACL(options, use_arena.into()) };
				if status_to_result_and_log("ACL", status).is_ok() {
					return; // EP found
				}
			}
			#[cfg(feature = "onednn")]
			"DnnlExecutionProvider" => {
				let use_arena = init_args.get("use_arena").map(|s| s.parse::<bool>().unwrap_or(false)).unwrap_or(false);
				let status = unsafe { OrtSessionOptionsAppendExecutionProvider_Dnnl(options, use_arena.into()) };
				if status_to_result_and_log("oneDNN", status).is_ok() {
					return; // EP found
				}
			}
			_ => {}
		};
	}
}
