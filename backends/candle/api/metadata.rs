use std::{mem, ptr};

use ort_sys::*;

use crate::{
	memory::{ort_alloc, ort_alloc_cstring},
	session::ModelMetadata,
	util::{str_from_ptr, trys}
};

macro_rules! metadata_string_getter {
	($name:ident, $field:ident) => {
		pub(super) unsafe extern "system" fn $name(
			model_metadata: *const OrtModelMetadata,
			allocator: *mut OrtAllocator,
			value: *mut *mut c_char
		) -> OrtStatusPtr {
			let metadata = unsafe { &*model_metadata.cast::<ModelMetadata>() };
			unsafe { value.write(trys!(ort_alloc_cstring(allocator, &metadata.$field))) };
			OrtStatusPtr::default()
		}
	};
}

metadata_string_getter!(ModelMetadataGetProducerName, producer_name);
metadata_string_getter!(ModelMetadataGetGraphName, graph_name);
metadata_string_getter!(ModelMetadataGetDomain, domain);
metadata_string_getter!(ModelMetadataGetDescription, description);
metadata_string_getter!(ModelMetadataGetGraphDescription, graph_description);

pub(super) unsafe extern "system" fn ModelMetadataGetVersion(model_metadata: *const OrtModelMetadata, value: *mut i64) -> OrtStatusPtr {
	let metadata = unsafe { &*model_metadata.cast::<ModelMetadata>() };
	unsafe { value.write(metadata.version) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ModelMetadataLookupCustomMetadataMap(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	key: *const c_char,
	value: *mut *mut c_char
) -> OrtStatusPtr {
	let metadata = unsafe { &*model_metadata.cast::<ModelMetadata>() };
	let key = trys!(unsafe { str_from_ptr(key) });
	unsafe {
		value.write(match metadata.custom_metadata.iter().find(|(k, _)| k == key) {
			Some((_, v)) => trys!(ort_alloc_cstring(allocator, v)),
			None => ptr::null_mut()
		})
	};
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ModelMetadataGetCustomMetadataMapKeys(
	model_metadata: *const OrtModelMetadata,
	allocator: *mut OrtAllocator,
	keys: *mut *mut *mut c_char,
	num_keys: *mut i64
) -> OrtStatusPtr {
	let metadata = unsafe { &*model_metadata.cast::<ModelMetadata>() };
	let count = metadata.custom_metadata.len();
	unsafe { num_keys.write(count as i64) };
	if count == 0 {
		unsafe { keys.write(ptr::null_mut()) };
		return OrtStatusPtr::default();
	}

	let key_array = trys!(unsafe { ort_alloc(allocator, count * mem::size_of::<*mut c_char>()) }).cast::<*mut c_char>();
	for (i, (key, _)) in metadata.custom_metadata.iter().enumerate() {
		match unsafe { ort_alloc_cstring(allocator, key) } {
			Ok(key) => unsafe { key_array.add(i).write(key) },
			Err(e) => {
				for j in 0..i {
					unsafe { (*allocator).Free.unwrap()(allocator, (*key_array.add(j)).cast()) };
				}
				unsafe { (*allocator).Free.unwrap()(allocator, key_array.cast()) };
				return e.into_sys();
			}
		}
	}

	unsafe { keys.write(key_array) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseModelMetadata(input: *mut OrtModelMetadata) {
	drop(unsafe { Box::from_raw(input.cast::<ModelMetadata>()) });
}
