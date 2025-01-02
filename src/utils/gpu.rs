use std::ffi::CStr;

pub struct GPUInfo {
    pub vendor: String,
    pub renderer: String,
    pub version: String,
}

pub unsafe fn get_gpu_info() -> GPUInfo {
    let vendor = {
        let ptr = gl::GetString(gl::VENDOR);
        if !ptr.is_null() {
            CStr::from_ptr(ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        } else {
            "Unknown".to_string()
        }
    };

    let renderer = {
        let ptr = gl::GetString(gl::RENDERER);
        if !ptr.is_null() {
            CStr::from_ptr(ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        } else {
            "Unknown".to_string()
        }
    };

    let version = {
        let ptr = gl::GetString(gl::VERSION);
        if !ptr.is_null() {
            CStr::from_ptr(ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        } else {
            "Unknown".to_string()
        }
    };

    GPUInfo {
        vendor,
        renderer,
        version,
    }
}