use std::ffi::CStr;

use ort_sys::{stub::Error, *};

use crate::{
	session::SessionOptions,
	tensor::Tensor,
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateSessionOptions(options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	unsafe { options.write((Box::leak(Box::new(SessionOptions::default())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CloneSessionOptions(in_options: *const OrtSessionOptions, out_options: *mut *mut OrtSessionOptions) -> OrtStatusPtr {
	let options = unsafe { &*in_options.cast::<SessionOptions>() };
	unsafe { out_options.write((Box::leak(Box::new(options.clone())) as *mut SessionOptions).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddSessionConfigEntry(
	options: *mut OrtSessionOptions,
	config_key: *const c_char,
	config_value: *const c_char
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let key = trys!(unsafe { str_from_ptr(config_key) });
	let value = unsafe { CStr::from_ptr(config_value) };
	options.config.insert(key.to_string(), value.to_owned());
	OrtStatusPtr::default()
}

pub(super) unsafe fn add_initializer(options: &mut SessionOptions, name: *const c_char, value: *const OrtValue) -> Result<(), Error> {
	let name = unsafe { str_from_ptr(name) }?;
	let value = unsafe { &*value.cast::<Tensor>() };
	options.add_initializer(name, value)
}

pub(super) unsafe extern "system" fn AddInitializer(options: *mut OrtSessionOptions, name: *const c_char, val: *const OrtValue) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	trys!(unsafe { add_initializer(options, name, val) });
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddExternalInitializers(
	options: *mut OrtSessionOptions,
	initializer_names: *const *const c_char,
	initializers: *const *const OrtValue,
	initializers_num: usize
) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	for i in 0..initializers_num {
		trys!(unsafe { add_initializer(options, *initializer_names.add(i), *initializers.add(i)) });
	}
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddFreeDimensionOverride(options: *mut OrtSessionOptions, dim_denotation: *const c_char, dim_value: i64) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let denotation = trys!(unsafe { str_from_ptr(dim_denotation) });
	options.dimension_overrides_by_denotation.insert(denotation.to_string(), dim_value);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AddFreeDimensionOverrideByName(options: *mut OrtSessionOptions, dim_name: *const c_char, dim_value: i64) -> OrtStatusPtr {
	let options = unsafe { &mut *options.cast::<SessionOptions>() };
	let name = trys!(unsafe { str_from_ptr(dim_name) });
	options.dimension_overrides_by_name.insert(name.to_string(), dim_value);
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseSessionOptions(input: *mut OrtSessionOptions) {
	drop(unsafe { Box::from_raw(input.cast::<SessionOptions>()) });
}

struct PrepackedWeightsContainer;

pub(super) unsafe extern "system" fn CreatePrepackedWeightsContainer(out: *mut *mut OrtPrepackedWeightsContainer) -> OrtStatusPtr {
	unsafe { out.write(Box::into_raw(Box::new(PrepackedWeightsContainer)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleasePrepackedWeightsContainer(input: *mut OrtPrepackedWeightsContainer) {
	drop(unsafe { Box::from_raw(input.cast::<PrepackedWeightsContainer>()) });
}
