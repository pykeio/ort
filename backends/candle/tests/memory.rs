use ort::memory::{AllocationDevice, Allocator, DeviceType};

#[test]
fn test_memory_info_apis() {
	ort::set_api(ort_candle::api());

	let allocator = Allocator::default();

	let memory_info = allocator.memory_info();
	assert_eq!(memory_info.allocation_device(), AllocationDevice::CPU);
	assert_eq!(memory_info.device_type(), DeviceType::CPU);
	assert_eq!(memory_info.device_id(), 0);

	let memory_info_clone = memory_info.clone();
	assert_eq!(memory_info, memory_info_clone);
}
