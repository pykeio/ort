use std::{ffi::OsString, fs, slice};

use ort_sys::{stub::Error, *};

use crate::{
	memory::{DEFAULT_CPU_MEMORY_INFO, MemoryInfo, ort_alloc_cstring},
	session::{RunOptions, Session, SessionOptions, write_outputs},
	tensor::Tensor,
	util::{str_from_ptr, trys}
};

unsafe fn path_from_os_char(model_path: *const os_char) -> OsString {
	let len = (0..).take_while(|&i| unsafe { *model_path.offset(i) } != 0).count();
	let path = unsafe { slice::from_raw_parts(model_path, len) };
	#[cfg(target_os = "windows")]
	{
		use std::os::windows::ffi::OsStringExt;
		OsString::from_wide(path)
	}
	#[cfg(not(target_os = "windows"))]
	unsafe {
		OsString::from_encoded_bytes_unchecked(path.iter().map(|c| *c as u8).collect::<Vec<_>>())
	}
}

pub(super) unsafe extern "system" fn CreateSession(
	_env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let path = unsafe { path_from_os_char(model_path) };
	let buf = match fs::read(&path) {
		Ok(buf) => buf,
		Err(e) => return Error::new_sys(OrtErrorCode::ORT_NO_SUCHFILE, format!("failed to read model file: {e}"))
	};

	let session = trys!(Session::from_buffer(options, &buf));
	unsafe { out.write((Box::leak(Box::new(session)) as *mut Session).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateSessionFromArray(
	_env: *const OrtEnv,
	model_data: *const c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	let options = unsafe { &*options.cast::<SessionOptions>() };

	let buf = unsafe { slice::from_raw_parts(model_data.cast::<u8>(), model_data_length) };

	let session = trys!(Session::from_buffer(options, buf));
	unsafe { out.write((Box::leak(Box::new(session)) as *mut Session).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateSessionWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_path: *const os_char,
	options: *const OrtSessionOptions,
	_prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	unsafe { CreateSession(env, model_path, options, out) }
}

pub(super) unsafe extern "system" fn CreateSessionFromArrayWithPrepackedWeightsContainer(
	env: *const OrtEnv,
	model_data: *const c_void,
	model_data_length: usize,
	options: *const OrtSessionOptions,
	_prepacked_weights_container: *mut OrtPrepackedWeightsContainer,
	out: *mut *mut OrtSession
) -> OrtStatusPtr {
	unsafe { CreateSessionFromArray(env, model_data, model_data_length, options, out) }
}

unsafe fn collect_names<'s>(names: *const *const c_char, len: usize) -> Result<Vec<&'s str>, Error> {
	if len == 0 {
		return Ok(Vec::new());
	}

	unsafe { slice::from_raw_parts(names, len) }
		.iter()
		.map(|&name| unsafe { str_from_ptr(name) })
		.collect()
}

pub(super) unsafe extern "system" fn Run(
	session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	input_names: *const *const c_char,
	inputs: *const *const OrtValue,
	input_len: usize,
	output_names: *const *const c_char,
	output_names_len: usize,
	output_ptrs: *mut *mut OrtValue
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let run_options = unsafe { run_options.cast::<RunOptions>().as_ref() };

	let input_names = trys!(unsafe { collect_names(input_names, input_len) });
	let inputs: Vec<(&str, &Tensor)> = if input_len == 0 {
		Vec::new()
	} else {
		input_names
			.into_iter()
			.zip(unsafe { slice::from_raw_parts(inputs, input_len) })
			.map(|(name, &input)| (name, unsafe { &*input.cast::<Tensor>() }))
			.collect()
	};
	let output_names = trys!(unsafe { collect_names(output_names, output_names_len) });

	let outputs = trys!(session.run(&inputs, &output_names, run_options));
	if output_names_len > 0 {
		trys!(unsafe { write_outputs(outputs, slice::from_raw_parts_mut(output_ptrs, output_names_len)) });
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetInputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.inputs.len()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputCount(session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(session.outputs.len()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOverridableInitializerCount(_session: *const OrtSession, out: *mut usize) -> OrtStatusPtr {
	unsafe { out.write(0) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetInputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(input) = session.inputs.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid input #{}", index + 1));
	};
	unsafe { type_info.write(input.type_info().into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputTypeInfo(session: *const OrtSession, index: usize, type_info: *mut *mut OrtTypeInfo) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(output) = session.outputs.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid output #{}", index + 1));
	};
	unsafe { type_info.write(output.type_info().into_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetInputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(input) = session.inputs.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid input #{}", index + 1));
	};
	unsafe { value.write(trys!(ort_alloc_cstring(allocator, &input.name))) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetOutputName(
	session: *const OrtSession,
	index: usize,
	allocator: *mut OrtAllocator,
	value: *mut *mut c_char
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	let Some(output) = session.outputs.get(index) else {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("Invalid output #{}", index + 1));
	};
	unsafe { value.write(trys!(ort_alloc_cstring(allocator, &output.name))) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetMemoryInfoForInputs(
	session: *const OrtSession,
	inputs_memory_info: *mut *const OrtMemoryInfo,
	num_inputs: usize
) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	if num_inputs != session.inputs.len() {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "number of inputs must exactly match the session's input count");
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
	if num_outputs != session.outputs.len() {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, "number of outputs must exactly match the session's output count");
	}
	for i in 0..num_outputs {
		unsafe { outputs_memory_info.add(i).write((&DEFAULT_CPU_MEMORY_INFO as *const MemoryInfo).cast()) };
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn SessionGetModelMetadata(session: *const OrtSession, out: *mut *mut OrtModelMetadata) -> OrtStatusPtr {
	let session = unsafe { &*session.cast::<Session>() };
	unsafe { out.write(Box::into_raw(Box::new(session.metadata.clone())).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseSession(input: *mut OrtSession) {
	drop(unsafe { Box::<Session>::from_raw(input.cast()) });
}
