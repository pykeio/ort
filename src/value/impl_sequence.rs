use std::{
	marker::PhantomData,
	ptr::{self, NonNull}
};

use crate::{memory::Allocator, ortsys, Error, Result, Value, ValueRef, ValueType};

impl Value {
	pub fn extract_sequence<'s>(&'s self, allocator: &Allocator) -> Result<Vec<ValueRef<'s>>> {
		match self.dtype()? {
			ValueType::Sequence(_) => {
				let mut len: ort_sys::size_t = 0;
				ortsys![unsafe GetValueCount(self.ptr(), &mut len) -> Error::ExtractSequence];

				let mut vec = Vec::with_capacity(len as usize);
				for i in 0..len {
					let mut value_ptr = ptr::null_mut();
					ortsys![unsafe GetValue(self.ptr(), i as _, allocator.ptr.as_ptr(), &mut value_ptr) -> Error::ExtractSequence; nonNull(value_ptr)];

					vec.push(ValueRef {
						inner: unsafe { Value::from_ptr(NonNull::new_unchecked(value_ptr), None) },
						lifetime: PhantomData
					});
				}
				Ok(vec)
			}
			t => Err(Error::NotSequence(t))
		}
	}
}
