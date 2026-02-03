const V18: u32 = cfg!(feature = "api-18") as u32;
const V19: u32 = cfg!(feature = "api-19") as u32;
const V20: u32 = cfg!(feature = "api-20") as u32;
const V21: u32 = cfg!(feature = "api-21") as u32;
const V22: u32 = cfg!(feature = "api-22") as u32;
const V23: u32 = cfg!(feature = "api-23") as u32;

#[rustfmt::skip]
pub const ORT_API_VERSION: u32 = 17 // minimum version
	+ V18 + V19 + V20 + V21 + V22 + V23; // We can do this because each API also enables the one before it.
