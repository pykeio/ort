use alloc::{ffi::CString, sync::Arc};
use core::{
	ffi::c_char,
	mem::{self, ManuallyDrop},
	ptr::{self, NonNull}
};

use smallvec::SmallVec;

use crate::{
	AsPointer, Error, OnceLock, Result,
	environment::get_environment,
	operator::attribute::Attribute,
	ortsys,
	session::{Session, builder::SessionBuilder},
	util::{run_on_drop, with_cstr, with_cstr_ptr_array},
	value::{TensorValueTypeMarker, Value, ValueType}
};

#[cfg(test)]
mod tests;

pub const ONNX_DOMAIN: &str = "";
pub const ONNX_ML_DOMAIN: &str = "ai.onnx.ml";

/// Returns a pointer to the global [`ort_sys::OrtModelEditorApi`] object, or errors if the Model Editor API is not
/// supported.
pub fn editor_api() -> Result<&'static ort_sys::OrtModelEditorApi> {
	struct ModelEditorApiPointer(*const ort_sys::OrtModelEditorApi);
	unsafe impl Send for ModelEditorApiPointer {}
	unsafe impl Sync for ModelEditorApiPointer {}

	static EDITOR_API: OnceLock<ModelEditorApiPointer> = OnceLock::new();

	let ptr = NonNull::new(
		EDITOR_API
			.get_or_init(|| {
				let api = ortsys![unsafe GetModelEditorApi()];
				ModelEditorApiPointer(api)
			})
			.0
			.cast_mut()
	)
	.ok_or_else(|| Error::new("The Model Editor API is not supported with this backend."))?;
	Ok(unsafe { ptr.as_ref() })
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Outlet(NonNull<ort_sys::OrtValueInfo>);

impl Outlet {
	pub fn new(name: impl AsRef<str>, dtype: &ValueType) -> Result<Self> {
		let type_info = dtype.to_type_info()?;
		let _guard = run_on_drop(|| ortsys![unsafe ReleaseTypeInfo(type_info)]);

		let ptr = with_cstr(name.as_ref().as_bytes(), &|name| {
			let mut ptr: *mut ort_sys::OrtValueInfo = ptr::null_mut();
			ortsys![@editor: unsafe CreateValueInfo(name.as_ptr(), type_info, &mut ptr)?; nonNull(ptr)];
			Ok(ptr)
		})?;
		Ok(Self(ptr))
	}

	pub(crate) fn consume(self) -> NonNull<ort_sys::OrtValueInfo> {
		ManuallyDrop::new(self).0
	}
}

impl AsPointer for Outlet {
	type Sys = ort_sys::OrtValueInfo;
	fn ptr(&self) -> *const Self::Sys {
		self.0.as_ptr()
	}
}

impl Drop for Outlet {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseValueInfo(self.0.as_ptr())];
	}
}

pub struct Node(NonNull<ort_sys::OrtNode>);

impl Node {
	pub fn new<I: AsRef<str>, O: AsRef<str>>(
		operator_name: impl AsRef<str>,
		domain_name: impl AsRef<str>,
		node_name: impl AsRef<str>,
		inputs: impl AsRef<[I]>,
		outputs: impl AsRef<[O]>,
		attributes: impl AsRef<[Attribute]>
	) -> Result<Self> {
		// This code is garbage and we should probably just be allocating.
		with_cstr(operator_name.as_ref().as_bytes(), &|operator_name| {
			with_cstr(domain_name.as_ref().as_bytes(), &|domain_name| {
				with_cstr(node_name.as_ref().as_bytes(), &|node_name| {
					with_cstr_ptr_array(inputs.as_ref(), &|inputs| {
						with_cstr_ptr_array(outputs.as_ref(), &|outputs| {
							let attributes = attributes.as_ref();
							let mut out = ptr::null_mut();
							ortsys![@editor:
								unsafe CreateNode(
									operator_name.as_ptr(),
									domain_name.as_ptr(),
									node_name.as_ptr(),
									inputs.as_ptr(),
									inputs.len(),
									outputs.as_ptr(),
									outputs.len(),
									attributes.as_ptr() as *mut _,
									attributes.len(),
									&mut out
								)?;
								nonNull(out)
							];
							Ok(Self(out))
						})
					})
				})
			})
		})
	}

	pub(crate) fn consume(self) -> *mut ort_sys::OrtNode {
		ManuallyDrop::new(self).0.as_ptr()
	}
}

impl AsPointer for Node {
	type Sys = ort_sys::OrtNode;
	fn ptr(&self) -> *const Self::Sys {
		self.0.as_ptr()
	}
}

impl Drop for Node {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseNode(self.0.as_ptr())];
	}
}

pub struct Graph(*mut ort_sys::OrtGraph);

impl Graph {
	pub fn new() -> Result<Self> {
		let mut out = ptr::null_mut();
		ortsys![@editor: unsafe CreateGraph(&mut out)?];
		Ok(Self(out))
	}

	pub fn set_inputs(&mut self, inputs: impl IntoIterator<Item = Outlet>) -> Result<()> {
		let inputs: SmallVec<NonNull<ort_sys::OrtValueInfo>, 4> = inputs.into_iter().map(|input| input.consume()).collect();
		// no need to drop consumed pointers since this only errors on null pointers
		ortsys![@editor: unsafe SetGraphInputs(self.0, inputs.as_ptr() as *mut _, inputs.len())?];
		Ok(())
	}

	pub fn set_outputs(&mut self, outputs: impl IntoIterator<Item = Outlet>) -> Result<()> {
		let outputs: SmallVec<NonNull<ort_sys::OrtValueInfo>, 4> = outputs.into_iter().map(|output| output.consume()).collect();
		// no need to drop consumed pointers since this only errors on null pointers
		ortsys![@editor: unsafe SetGraphOutputs(self.0, outputs.as_ptr() as *mut _, outputs.len())?];
		Ok(())
	}

	pub fn add_node(&mut self, node: Node) -> Result<()> {
		let node = node.consume();
		ortsys![@editor: unsafe AddNodeToGraph(self.0, node)?]; // infallible
		Ok(())
	}

	pub fn add_initializer<T: TensorValueTypeMarker>(&mut self, name: impl AsRef<str>, mut initializer: Value<T>, as_external: bool) -> Result<()> {
		let Some(value_inner) = Arc::get_mut(&mut initializer.inner) else {
			return Err(Error::new("Initializers must be unique"))?;
		};
		if value_inner._backing.is_some() {
			// `AddInitializerToGraph` wants to take ownership of the value, so the memory needs to be managed by ONNX Runtime.
			// The documentation technically recommends using non-managed memory when `as_external = true`, but it doesn't seem like
			// that matters.
			return Err(Error::new("Initializers must be created via `Tensor::new`, not created from an array (try passing a `.clone()` of the value)"))?;
		}

		with_cstr(name.as_ref().as_bytes(), &|name| {
			ortsys![@editor: unsafe AddInitializerToGraph(self.0, name.as_ptr(), value_inner.ptr.as_ptr(), as_external)?];
			Ok(())
		})?;

		// Allow the `ValueInner` to be dropped so the `Arc` strong count is decremented, but don't allow it to actually
		// *release* the underlying value, since it's now owned by ONNX Runtime.
		value_inner.drop = false;

		Ok(())
	}

	pub(crate) fn consume(self) -> *mut ort_sys::OrtGraph {
		let ptr = self.0;
		mem::forget(self);
		ptr
	}
}

impl AsPointer for Graph {
	type Sys = ort_sys::OrtGraph;
	fn ptr(&self) -> *const Self::Sys {
		self.0
	}
}

impl Drop for Graph {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseGraph(self.0)];
	}
}

#[derive(Debug, Clone)]
pub struct Opset {
	domain_name: CString,
	version: u32
}

impl Opset {
	pub fn new(domain_name: impl AsRef<str>, version: u32) -> Result<Self> {
		let domain_name = CString::new(domain_name.as_ref())?;
		Ok(Self { domain_name, version })
	}
}

#[derive(Debug)]
#[repr(transparent)]
pub struct Model(NonNull<ort_sys::OrtModel>);

impl Model {
	pub fn new(opsets: impl AsRef<[Opset]>) -> Result<Self> {
		let opsets = opsets.as_ref();
		let domain_names: SmallVec<*const c_char, 4> = opsets.iter().map(|p| p.domain_name.as_ptr()).collect();
		let opset_versions: SmallVec<i32, 4> = opsets.iter().map(|p| p.version as i32).collect();

		let mut ptr = ptr::null_mut();
		ortsys![@editor: unsafe CreateModel(domain_names.as_ptr(), opset_versions.as_ptr(), opsets.len(), &mut ptr)?; nonNull(ptr)];
		Ok(Self(ptr))
	}

	pub fn add_graph(&mut self, graph: Graph) -> Result<()> {
		let graph = graph.consume();
		ortsys![@editor: unsafe AddGraphToModel(self.0.as_ptr(), graph)?]; // infallible (errors on null pointer)
		Ok(())
	}

	pub fn into_session(self, mut options: SessionBuilder) -> Result<Session> {
		let env = get_environment()?;
		let mut session_ptr = ptr::null_mut();
		ortsys![@editor:
			unsafe CreateSessionFromModel(
				env.ptr(),
				self.0.as_ptr(),
				options.ptr(),
				&mut session_ptr
			)?;
			nonNull(session_ptr)
		];
		options.commit_finalize(session_ptr)
	}
}

impl AsPointer for Model {
	type Sys = ort_sys::OrtModel;
	fn ptr(&self) -> *const Self::Sys {
		self.0.as_ptr()
	}
}

impl Drop for Model {
	fn drop(&mut self) {
		ortsys![unsafe ReleaseModel(self.0.as_ptr())];
	}
}
