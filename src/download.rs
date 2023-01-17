pub mod language;
pub mod vision;

pub trait ModelUrl {
	fn fetch_url(&self) -> &'static str;
}
