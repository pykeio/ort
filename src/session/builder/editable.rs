use alloc::{boxed::Box, sync::Arc};
use core::{
	ops::Deref,
	ptr::{self, NonNull}
};

use super::{PrepackedWeights, SessionBuilder};
use crate::{AsPointer, Error, Result, editor::Model, ortsys, session::Session};

pub struct EditableSession {
	session: Session,
	builder: SessionBuilder,
	prepacked_weights: Option<PrepackedWeights>
}

impl EditableSession {
	pub(crate) fn new(session: NonNull<ort_sys::OrtSession>, mut builder: SessionBuilder) -> Result<Self> {
		// Prepacked weights are passed to `FinalizeModelEditorSession`; steal them from the builder so we can add them later.
		let prepacked_weights = builder.prepacked_weights.take();
		Ok(Self {
			session: builder.commit_finalize(session)?,
			builder,
			prepacked_weights
		})
	}

	pub fn apply_model(&mut self, model: &Model) -> Result<()> {
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

		if let Some(prepacked_weights) = self.prepacked_weights {
			let Some(inner) = Arc::get_mut(&mut self.session.inner) else {
				return Err(Error::new("Expected to have exclusive access to session inner"));
			};

			// add to extras so it outlives the session
			inner._extras.push(Box::new(prepacked_weights));
		}

		Ok(self.session)
	}
}

impl Deref for EditableSession {
	type Target = Session;

	fn deref(&self) -> &Self::Target {
		&self.session
	}
}
