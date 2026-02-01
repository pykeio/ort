static API_BASE: ort_sys::OrtApiBase = ort_sys::OrtApiBase {
	GetVersionString: get_version_string,
	GetApi: get_api
};
static API: ort_sys::OrtApi = ort_candle::api();

unsafe extern "system" fn get_version_string() -> *const ort_sys::c_char {
	c"1.23.2+candle@0.9-wrapper@0.2.0".as_ptr()
}

unsafe extern "system" fn get_api(version: u32) -> *const ort_sys::OrtApi {
	if version <= ort_sys::ORT_API_VERSION { &API as *const _ } else { core::ptr::null() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn OrtGetApiBase() -> *const ort_sys::OrtApiBase {
	&API_BASE as *const _
}
