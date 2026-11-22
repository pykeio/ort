use ort_sys::*;

use crate::{
	memory::{Allocator, MemoryInfo, ort_alloc},
	util::{str_from_ptr, trys}
};

pub(super) unsafe extern "system" fn CreateMemoryInfo(
	name: *const c_char,
	type_: OrtAllocatorType,
	id: c_int,
	mem_type: OrtMemType,
	out: *mut *mut OrtMemoryInfo
) -> OrtStatusPtr {
	let device_name = trys!(unsafe { str_from_ptr(name) });
	let info = trys!(MemoryInfo::new(device_name, id as usize, type_, mem_type));
	unsafe { out.write(Box::into_raw(Box::new(info)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn CreateCpuMemoryInfo(type_: OrtAllocatorType, mem_type: OrtMemType, out: *mut *mut OrtMemoryInfo) -> OrtStatusPtr {
	unsafe { CreateMemoryInfo(c"Cpu".as_ptr(), type_, 0, mem_type, out) }
}

pub(super) unsafe extern "system" fn CompareMemoryInfo(info1: *const OrtMemoryInfo, info2: *const OrtMemoryInfo, out: *mut c_int) -> OrtStatusPtr {
	let info1 = unsafe { &*info1.cast::<MemoryInfo>() };
	let info2 = unsafe { &*info2.cast::<MemoryInfo>() };
	unsafe { out.write(if info1 == info2 { 0 } else { -1 }) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn MemoryInfoGetName(ptr: *const OrtMemoryInfo, out: *mut *const c_char) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.device_name_sys()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn MemoryInfoGetId(ptr: *const OrtMemoryInfo, out: *mut c_int) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.device_id() as c_int) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn MemoryInfoGetMemType(ptr: *const OrtMemoryInfo, out: *mut OrtMemType) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.memory_type) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn MemoryInfoGetType(ptr: *const OrtMemoryInfo, out: *mut OrtAllocatorType) -> OrtStatusPtr {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.allocator_type) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn MemoryInfoGetDeviceType(ptr: *const OrtMemoryInfo, out: *mut OrtMemoryInfoDeviceType) {
	let info = unsafe { &*ptr.cast::<MemoryInfo>() };
	unsafe { out.write(info.device_type()) };
}

pub(super) unsafe extern "system" fn ReleaseMemoryInfo(input: *mut OrtMemoryInfo) {
	drop(unsafe { Box::from_raw(input.cast::<MemoryInfo>()) });
}

pub(super) unsafe extern "system" fn CreateAllocator(_session: *const OrtSession, mem_info: *const OrtMemoryInfo, out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	let mem_info = unsafe { &*mem_info.cast::<MemoryInfo>() };
	unsafe { out.write(Box::into_raw(Box::new(Allocator::new(mem_info.clone()))).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AllocatorAlloc(ort_allocator: *mut OrtAllocator, size: usize, out: *mut *mut c_void) -> OrtStatusPtr {
	unsafe { out.write(trys!(ort_alloc(ort_allocator, size)).cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AllocatorFree(ort_allocator: *mut OrtAllocator, p: *mut c_void) -> OrtStatusPtr {
	unsafe { (*ort_allocator).Free.unwrap()(ort_allocator, p) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn AllocatorGetInfo(ort_allocator: *const OrtAllocator, out: *mut *const OrtMemoryInfo) -> OrtStatusPtr {
	unsafe { out.write((*ort_allocator).Info.unwrap()(ort_allocator)) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn GetAllocatorWithDefaultOptions(out: *mut *mut OrtAllocator) -> OrtStatusPtr {
	unsafe { out.write((&crate::memory::DEFAULT_CPU_ALLOCATOR as *const Allocator).cast_mut().cast()) };
	OrtStatusPtr::default()
}

pub(super) unsafe extern "system" fn ReleaseAllocator(input: *mut OrtAllocator) {
	drop(unsafe { Box::from_raw(input.cast::<Allocator>()) });
}
