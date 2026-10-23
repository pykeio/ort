use core::{mem, pin::Pin};

use ort_sys::{stub::Error, *};

use crate::{
	SyncDirection, binding,
	env::{Environment, TelemetryEvent},
	memory::{DEFAULT_CPU_MEMORY_INFO, MemoryInfo, ort_alloc_cstring},
	session::{RunOptions, Session, SessionOptions},
	tensor::{Tensor, TypeInfo},
	util::{trys, value_to_string}
};

pub(super) unsafe fn CreateSession(
	env: *const OrtEnv,
	model_path: &str,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let fut = Box::pin(async move {
		match Session::from_url(model_path, options).await {
			Ok(session) => {
				let ptr = (Box::leak(Box::new(session))) as *mut Session;
				unsafe { out.write(ptr.cast()) };

				{
					let env = unsafe { Environment::cast_from_sys(env) };
					env.send_telemetry_event(TelemetryEvent::SessionInit);
				}

				OrtStatusPtr::default()
			}
			Err(e) => e.into_sys()
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { mem::transmute(fut) }
}

pub(super) unsafe fn CreateSessionFromArray(
	env: *const OrtEnv,
	model_data: &[u8],
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let fut = Box::pin(async move {
		match Session::from_bytes(model_data, options).await {
			Ok(session) => {
				let ptr = (Box::leak(Box::new(session))) as *mut Session;
				unsafe { out.write(ptr.cast()) };

				{
					let env = unsafe { Environment::cast_from_sys(env) };
					env.send_telemetry_event(TelemetryEvent::SessionInit);
				}

				OrtStatusPtr::default()
			}
			Err(e) => e.into_sys()
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { core::mem::transmute(fut) }
}

pub(super) unsafe extern "system" fn Run(
	_session: *mut OrtSession,
	_run_options: *const OrtRunOptions,
	_input_names: *const *const c_char,
	_inputs: *const *const OrtValue,
	_input_len: usize,
	_output_names: *const *const c_char,
	_output_names_len: usize,
	_output_ptrs: *mut *mut OrtValue
) -> OrtStatusPtr {
	Error::new_sys(OrtErrorCode::ORT_FAIL, "Synchronous `Session::run` is not supported in ort-web; use `run_async()`.")
}

async fn run_async(
	session: &Session,
	run_options: Option<&RunOptions>,
	input_names: &[&str],
	inputs: &[*const OrtValue],
	output_names: &[&str],
	output_ptrs: &mut [*mut OrtValue]
) -> Result<(), Error> {
	let mut feeds = Vec::with_capacity(inputs.len());
	for (&name, &input) in input_names.iter().zip(inputs) {
		if !session.input_names.iter().any(|n| n == name) {
			return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid input name: {name}")));
		}
		let input = unsafe { &*input.cast::<Tensor>() };
		let tensor = input
			.as_runnable()
			.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, format!("Failed to prepare input '{name}': {}", value_to_string(&e))))?;
		feeds.push((name, tensor));
	}
	for &name in output_names {
		if !session.output_names.iter().any(|n| n == name) {
			return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid output name: {name}")));
		}
	}

	let fetches = output_names.iter().zip(output_ptrs.iter()).map(|(&name, &ptr)| {
		let preallocated = unsafe { ptr.cast::<Tensor>().as_ref() };
		(name, preallocated.filter(|t| !t.memory_info.is_cpu()).map(|t| &t.js))
	});

	let default_run_options = RunOptions::default();
	let run_options = run_options.unwrap_or(&default_run_options);
	let mut outputs = session
		.js
		.run(feeds.into_iter(), fetches, &run_options.js)
		.await
		.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, value_to_string(&e)))?;

	let mut results = Vec::with_capacity(output_names.len());
	for &name in output_names {
		match outputs.iter().position(|(n, _)| n == name) {
			Some(i) => results.push(outputs.swap_remove(i).1),
			None => return Err(Error::new(OrtErrorCode::ORT_FAIL, format!("Output '{name}' was not computed")))
		}
	}

	for (result, &output_ptr) in results.iter().zip(output_ptrs.iter()) {
		if let Some(preallocated) = unsafe { output_ptr.cast::<Tensor>().as_mut() }
			&& preallocated.memory_info.is_cpu()
		{
			preallocated.copy_from(result).await?;
			let _ = result.dispose();
		}
	}

	for (result, output_ptr) in results.into_iter().zip(output_ptrs.iter_mut()) {
		if !output_ptr.is_null() {
			continue;
		}

		let mut tensor = Tensor::from_session_output(result);
		if !session.disable_sync {
			tensor
				.sync(SyncDirection::Rust)
				.await
				.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, format!("Failed to synchronize output: {e}")))?;
		}
		*output_ptr = Box::into_raw(Box::new(tensor)).cast();
	}

	Ok(())
}

pub(super) unsafe fn RunAsync(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: &[&str],
	inputs: &[*const OrtValue],
	output_names: &[&str],
	output_ptrs: &mut [*mut OrtValue]
) -> Pin<Box<dyn Future<Output = OrtStatusPtr>>> {
	let session = unsafe { &*session.cast::<Session>() };
	let run_options = unsafe { run_options.cast::<RunOptions>().as_ref() };

	let fut = Box::pin(async move {
		match run_async(session, run_options, input_names, inputs, output_names, output_ptrs).await {
			Ok(()) => OrtStatusPtr::default(),
			Err(e) => e.into_sys()
		}
	}) as Pin<Box<dyn Future<Output = OrtStatusPtr>>>;
	unsafe { mem::transmute(fut) }
}

pub(super) unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.input_names.len()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.output_names.len()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOverridableInitializerCount(_session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	unsafe { out.write(0) };
	OrtStatusPtr::default()
}

fn metadata_type_info(metadata: Result<Vec<binding::ValueMetadata>, serde_wasm_bindgen::Error>, index: usize) -> Result<TypeInfo, Error> {
	let metadata = metadata.map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, format!("Failed to get metadata: {e}")))?;
	let Some(metadata) = metadata.get(index) else {
		return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid index #{}", index + 1)));
	};
	if !metadata.is_tensor {
		return Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, "non-tensor types are not currently supported"));
	}
	Ok(TypeInfo::from_value_metadata(metadata))
}

pub(super) unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let info = trys!(metadata_type_info(session.js.input_metadata(), index));
	unsafe { type_info.write(info.into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let info = trys!(metadata_type_info(session.js.output_metadata(), index));
	unsafe { type_info.write(info.into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(name) = session.input_names.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid input #{}", index + 1));
	};
	unsafe { value.write(trys!(ort_alloc_cstring(allocator, name))) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(name) = session.output_names.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid output #{}", index + 1));
	};
	unsafe { value.write(trys!(ort_alloc_cstring(allocator, name))) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetMemoryInfoForInputs(
	session: *const OrtSession,
	inputs_memory_info: *mut *const OrtMemoryInfo,
	num_inputs: usize
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	if num_inputs != session.input_names.len() {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "Number of inputs does not match the session's input count");
	}
	for i in 0..num_inputs {
		unsafe { inputs_memory_info.add(i).write((&DEFAULT_CPU_MEMORY_INFO as *const MemoryInfo).cast()) };
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetMemoryInfoForOutputs(
	session: *const OrtSession,
	outputs_memory_info: *mut *const OrtMemoryInfo,
	num_outputs: usize
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	if num_outputs != session.output_names.len() {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "Number of outputs does not match the session's output count");
	}
	for i in 0..num_outputs {
		unsafe { outputs_memory_info.add(i).write((&DEFAULT_CPU_MEMORY_INFO as *const MemoryInfo).cast()) };
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseSession(input: *mut OrtSession) {
	drop(unsafe { Box::<Session>::from_raw(input.cast()) });
}
