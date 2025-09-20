macro_rules! debug {
	($fmt:expr) => {
		println!("[ort-sys] [DEBUG] {}", format!($fmt))
	};
	($fmt:expr, $($args:tt)*) => {
		println!("[ort-sys] [DEBUG] {}", format!($fmt, $($args)*))
	};
}
pub(crate) use debug;

macro_rules! warning {
	($fmt:expr) => {
		println!("cargo:warning=[ort-sys] [WARN] {}", format!($fmt))
	};
	($fmt:expr, $($args:tt)*) => {
		println!("cargo:warning=[ort-sys] [WARN] {}", format!($fmt, $($args)*))
	};
}
pub(crate) use warning;
