use std::{mem, ptr};

use ort_sys::{stub::Error, *};

use crate::{
	memory::ort_alloc,
	session::{BoundOutput, IoBinding, RunOptions},
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateIoBinding(session: *mut OrtSession, out: *mut *mut OrtIoBinding) -> OrtStatusPtr {
	let binding = IoBinding {
		session: session.cast_const().cast(),
		inputs: Vec::new(),
		outputs: Vec::new()
	};
	unsafe { out.write(Box::into_raw(Box::new(binding)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseIoBinding(input: *mut OrtIoBinding) {
	drop(unsafe { Box::from_raw(input.cast::<IoBinding>()) });
}

pub(super) unsafe extern "system" fn BindInput(binding_ptr: *mut OrtIoBinding, name: *const c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	let binding = unsafe { &mut *binding_ptr.cast::<IoBinding>() };
	let name = trys!(unsafe { str_from_ptr(name) });
	let session = unsafe { &*binding.session };
	if !session.inputs.iter().any(|i| i.name == name) {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("unknown input '{name}'"));
	}
	binding.bind_input(name.to_string(), val_ptr.cast());
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn BindOutput(binding_ptr: *mut OrtIoBinding, name: *const c_char, val_ptr: *const OrtValue) -> OrtStatusPtr {
	let binding = unsafe { &mut *binding_ptr.cast::<IoBinding>() };
	let name = trys!(unsafe { str_from_ptr(name) });
	let session = unsafe { &*binding.session };
	if !session.outputs.iter().any(|i| i.name == name) {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("unknown output '{name}'"));
	}
	binding.bind_output(name.to_string(), BoundOutput::Value(val_ptr.cast_mut().cast()));
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn BindOutputToDevice(
	binding_ptr: *mut OrtIoBinding,
	name: *const c_char,
	_mem_info_ptr: *const OrtMemoryInfo
) -> OrtStatusPtr {
	let binding = unsafe { &mut *binding_ptr.cast::<IoBinding>() };
	let name = trys!(unsafe { str_from_ptr(name) });
	let session = unsafe { &*binding.session };
	if !session.outputs.iter().any(|i| i.name == name) {
		return Error::new_sys(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("unknown output '{name}'"));
	}
	// outputs are always on CPU
	binding.bind_output(name.to_string(), BoundOutput::Device(None));
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ClearBoundInputs(binding_ptr: *mut OrtIoBinding) {
	unsafe { &mut *binding_ptr.cast::<IoBinding>() }.inputs.clear();
}

pub(super) unsafe extern "system" fn ClearBoundOutputs(binding_ptr: *mut OrtIoBinding) {
	unsafe { &mut *binding_ptr.cast::<IoBinding>() }.outputs.clear();
}

pub(super) unsafe extern "system" fn RunWithBinding(
	_session: *mut OrtSession,
	run_options: *const OrtRunOptions,
	binding_ptr: *const OrtIoBinding
) -> OrtStatusPtr {
	let binding = unsafe { &mut *binding_ptr.cast_mut().cast::<IoBinding>() };
	let run_options = unsafe { run_options.cast::<RunOptions>().as_ref() };
	trys!(binding.run(run_options));
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetBoundOutputNames(
	binding_ptr: *const OrtIoBinding,
	allocator: *mut OrtAllocator,
	buffer: *mut *mut c_char,
	lengths: *mut *mut usize,
	count: *mut usize
) -> OrtStatusPtr {
	let binding = unsafe { &*binding_ptr.cast::<IoBinding>() };
	unsafe { count.write(binding.outputs.len()) };
	if binding.outputs.is_empty() {
		unsafe {
			buffer.write(ptr::null_mut());
			lengths.write(ptr::null_mut());
		}
		return OrtStatusPtr::default();
	}

	let total_len: usize = binding.outputs.iter().map(|(name, _)| name.len()).sum();
	let names_buf = trys!(unsafe { ort_alloc(allocator, total_len.max(1)) });
	let lengths_buf = match unsafe { ort_alloc(allocator, binding.outputs.len() * mem::size_of::<usize>()) } {
		Ok(buf) => buf.cast::<usize>(),
		Err(e) => {
			unsafe { (*allocator).Free.unwrap()(allocator, names_buf.cast()) };
			return e.into_sys();
		}
	};
	let mut offset = 0;
	for (i, (name, _)) in binding.outputs.iter().enumerate() {
		unsafe {
			ptr::copy_nonoverlapping(name.as_ptr(), names_buf.add(offset), name.len());
			lengths_buf.add(i).write(name.len());
		}
		offset += name.len();
	}
	unsafe {
		buffer.write(names_buf.cast());
		lengths.write(lengths_buf);
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetBoundOutputValues(
	binding_ptr: *const OrtIoBinding,
	allocator: *mut OrtAllocator,
	output: *mut *mut *mut OrtValue,
	output_count: *mut usize
) -> OrtStatusPtr {
	let binding = unsafe { &*binding_ptr.cast::<IoBinding>() };
	let values = trys!(binding.output_values());
	unsafe { output_count.write(values.len()) };
	if values.is_empty() {
		unsafe { output.write(ptr::null_mut()) };
		return OrtStatusPtr::default();
	}

	let values_buf = trys!(unsafe { ort_alloc(allocator, values.len() * mem::size_of::<*mut OrtValue>()) }).cast::<*mut OrtValue>();
	for (i, value) in values.into_iter().enumerate() {
		unsafe { values_buf.add(i).write(Box::into_raw(Box::new(value)).cast()) };
	}
	unsafe { output.write(values_buf) };
	OrtStatusPtr::default()
}
