static API_BASE: ort_sys::OrtApiBase = ort_sys::OrtApiBase {
	GetVersionString: get_version_string,
	GetApi: get_api
};
static API: ort_sys::OrtApi = ort_candle::api();

unsafe extern "system" fn get_version_string() -> *const ort_sys::c_char {
	c"1.20.0+candle@0.8-wrapper@0.1.0".as_ptr()
}

unsafe extern "system" fn get_api(version: u32) -> *const ort_sys::OrtApi {
	if version <= 20 { &API as *const _ } else { core::ptr::null() }
}

#[no_mangle]
pub unsafe extern "C" fn OrtGetApiBase() -> *const ort_sys::OrtApiBase {
	&API_BASE as *const _
}
