use std::{
	collections::hash_map::RandomState,
	hash::{BuildHasher, Hasher}
};

pub mod dirs;

pub fn random_identifier() -> String {
	let mut state = RandomState::new().build_hasher().finish();
	std::iter::repeat_with(move || {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		state
	})
	.take(12)
	.map(|i| b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[i as usize % 62] as char)
	.collect()
}
