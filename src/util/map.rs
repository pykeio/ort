use core::{borrow::Borrow, fmt, mem};

use smallvec::SmallVec;

// generally as performant or faster than HashMap<K, V> for <50 items. good enough for #[no_std]
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct MiniMap<K, V> {
	values: SmallVec<[(K, V); 6]>
}

impl<K, V> Default for MiniMap<K, V> {
	fn default() -> Self {
		Self { values: SmallVec::new() }
	}
}

impl<K, V> MiniMap<K, V> {
	pub fn new() -> Self {
		Self { values: SmallVec::new() }
	}
}

impl<K: Eq, V> MiniMap<K, V> {
	pub fn get<Q>(&self, key: &Q) -> Option<&V>
	where
		K: Borrow<Q>,
		Q: Eq + ?Sized
	{
		self.values.iter().find(|(k, _)| key.eq(k.borrow())).map(|(_, v)| v)
	}

	pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
	where
		K: Borrow<Q>,
		Q: Eq + ?Sized
	{
		self.values.iter_mut().find(|(k, _)| key.eq(k.borrow())).map(|(_, v)| v)
	}

	pub fn insert(&mut self, key: K, value: V) -> Option<V> {
		match self.get_mut(&key) {
			Some(v) => Some(mem::replace(v, value)),
			None => {
				self.values.push((key, value));
				None
			}
		}
	}

	pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> + '_ {
		self.values.drain(..)
	}

	pub fn len(&self) -> usize {
		self.values.len()
	}

	pub fn iter(&self) -> impl Iterator<Item = &(K, V)> + '_ {
		self.values.iter()
	}
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for MiniMap<K, V> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_map().entries(self.values.iter().map(|(k, v)| (k, v))).finish()
	}
}
