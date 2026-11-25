use alloc::sync::Arc;
use core::{
	ops::Deref,
	ptr::{self, NonNull}
};

use super::{PrepackedWeights, SessionBuilder};
use crate::{AsPointer, Error, Result, editor::Model, ortsys, session::Session};

pub struct EditableSession<'b> {
	session: Session,
	builder: &'b mut SessionBuilder,
	prepacked_weights: Option<PrepackedWeights>
}

impl<'b> EditableSession<'b> {
	pub(crate) fn new(session: NonNull<ort_sys::OrtSession>, builder: &'b mut SessionBuilder) -> Result<Self> {
		// Prepacked weights are passed to `FinalizeModelEditorSession`; steal them from the builder so we can add them
		// later.
		let prepacked_weights = builder.prepacked_weights.clone();
		Ok(Self {
			session: builder.commit_finalize(session, false)?,
			builder,
			prepacked_weights
		})
	}

	pub fn apply_model(&mut self, model: &Model) -> Result<()> {
		model.ensure_has_graph()?;

		ortsys![@editor:
			unsafe ApplyModelToModelEditorSession(
				self.session.ptr_mut(),
				model.ptr().cast_mut()
			)?
		];
		Ok(())
	}

	pub fn into_session(mut self) -> Result<Session> {
		ortsys![@editor:
			unsafe FinalizeModelEditorSession(
				self.session.ptr_mut(),
				self.builder.ptr(),
				self.prepacked_weights.as_ref().map(|p| p.ptr()).unwrap_or_else(ptr::null)
			)?
		];

		let session_ptr = self.session.inner.session_ptr;
		let Some(inner) = Arc::get_mut(&mut self.session.inner) else {
			return Err(Error::new("Expected to have exclusive access to session inner"));
		};

		inner.initialized = true;
		inner.allocator = self.builder.session_allocator(session_ptr)?;
		if let Some(prepacked_weights) = self.prepacked_weights {
			// add to extras so it outlives the session
			inner._extras.push(prepacked_weights.inner);
		}
		(self.session.inputs, self.session.outputs) = SessionBuilder::session_outlets(session_ptr)?;

		Ok(self.session)
	}
}

impl Deref for EditableSession<'_> {
	type Target = Session;

	fn deref(&self) -> &Self::Target {
		&self.session
	}
}
