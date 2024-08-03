use crate::error::Result;

#[allow(clippy::len_without_is_empty)]
pub trait DataLoader<I, L> {
	/// Synchronously loads the batch at index `idx`.
	fn load(&mut self, idx: usize) -> Result<(I, L)>;

	/// The total number of batches in this data loader. The default implementation returns `None`, which indicates the
	/// data loader is 'infinite'.
	///
	/// If `len` does not return `Some` (i.e., it is 'infinite'), you will not be able to use configuration options
	/// based on epochs.
	fn len(&self) -> Option<usize> {
		None
	}
}

/// A definitively-sized [`DataLoader`] created from any type that implements [`Iterator`].
///
/// To create an iterable data loader, use [`iterable_data_loader`].
pub struct IterableDataLoader<T, I, L, C: Fn(&T) -> Result<(I, L)>> {
	items: Box<[T]>,
	collator: C
}

impl<T, I, L, C: Fn(&T) -> Result<(I, L)>> DataLoader<I, L> for IterableDataLoader<T, I, L, C> {
	fn load(&mut self, idx: usize) -> Result<(I, L)> {
		(self.collator)(&self.items[idx])
	}

	fn len(&self) -> Option<usize> {
		Some(self.items.len())
	}
}

/// Creates a definitively-sized [`DataLoader`] from an [`Iterator`] and a corresponding collator function.
pub fn iterable_data_loader<T, I, L, C: Fn(&T) -> Result<(I, L)>>(iterable: impl Iterator<Item = T>, collator: C) -> IterableDataLoader<T, I, L, C> {
	IterableDataLoader {
		items: iterable.collect::<Vec<T>>().into_boxed_slice(),
		collator
	}
}

impl<I, L, F: FnMut(usize) -> Result<(I, L)>> DataLoader<I, L> for F {
	fn load(&mut self, idx: usize) -> Result<(I, L)> {
		(self)(idx)
	}

	fn len(&self) -> Option<usize> {
		None
	}
}
