use ort::memory::{AllocationDevice, Allocator, AllocatorType, DeviceType, MemoryInfo, MemoryType};

#[test]
fn test_memory_info_apis() {
	unsafe { ort::set_api(ort_candle::api()) };

	let allocator = Allocator::default();

	let memory_info = allocator.memory_info();
	assert_eq!(memory_info.allocation_device(), AllocationDevice::CPU);
	assert_eq!(memory_info.device_type(), DeviceType::CPU);
	assert_eq!(memory_info.device_id(), 0);
	assert_eq!(memory_info.memory_type(), MemoryType::Default);
	assert_eq!(memory_info.allocator_type(), AllocatorType::Device);

	let memory_info_clone = memory_info.clone();
	assert_eq!(memory_info, &memory_info_clone);

	let arena_info = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Arena, MemoryType::Default).unwrap();
	assert_eq!(arena_info.allocator_type(), AllocatorType::Arena);
	assert_ne!(memory_info, &arena_info);

	assert!(MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).is_err());
}

#[test]
fn test_allocator() -> ort::Result<()> {
	unsafe { ort::set_api(ort_candle::api()) };

	let allocator = Allocator::default();
	let mut block = allocator.alloc::<u64>(16)?;
	let ptr = block.as_mut_ptr().cast::<u64>();
	assert_eq!(ptr as usize % std::mem::align_of::<u64>(), 0);
	for i in 0..16 {
		unsafe { ptr.add(i).write(i as u64) };
	}
	assert_eq!(unsafe { *ptr.add(15) }, 15);
	drop(block);

	Ok(())
}
