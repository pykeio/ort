use super::ModelUrl;

pub mod machine_comprehension;

pub use machine_comprehension::MachineComprehension;

#[derive(Debug, Clone)]
pub enum Language {
	MachineComprehension(MachineComprehension)
}

impl ModelUrl for Language {
	fn fetch_url(&self) -> &'static str {
		match self {
			Language::MachineComprehension(v) => v.fetch_url()
		}
	}
}
