use std::{
	collections::{HashMap, HashSet},
	ffi::CString,
	ptr,
	sync::atomic::{AtomicBool, Ordering}
};

use candle_onnx::onnx::{ModelProto, TensorProto, ValueInfoProto, tensor_shape_proto::dimension, type_proto};
use ort_sys::{ONNXTensorElementDataType, OrtErrorCode};
use prost::Message;

use crate::{
	Error,
	tensor::{Tensor, TypeInfo},
	util::{convert_dtype_to_sys, convert_proto_to_sys, convert_sys_to_dtype}
};

#[derive(Default, Clone)]
pub(crate) struct SessionOptions {
	pub(crate) config: HashMap<String, CString>,
	pub(crate) initializers: Vec<TensorProto>,
	pub(crate) dimension_overrides_by_name: HashMap<String, i64>,
	pub(crate) dimension_overrides_by_denotation: HashMap<String, i64>
}

impl SessionOptions {
	pub(crate) fn add_initializer(&mut self, name: &str, value: &Tensor) -> Result<(), Error> {
		let Some(data) = value.host_bytes() else {
			return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, "initializers must be on CPU"));
		};

		let tensor_proto = TensorProto {
			name: name.to_string(),
			dims: value.shape.clone(),
			data_type: value.dtype as i32,
			raw_data: data.to_vec(),
			..TensorProto::default()
		};
		match self.initializers.iter_mut().find(|i| i.name == name) {
			Some(existing) => *existing = tensor_proto,
			None => self.initializers.push(tensor_proto)
		}
		Ok(())
	}

	fn apply_to_proto(&self, proto: &mut ModelProto) {
		let Some(graph) = proto.graph.as_mut() else {
			return;
		};

		for tensor_proto in &self.initializers {
			match graph.initializer.iter_mut().find(|i| i.name == tensor_proto.name) {
				Some(initializer) => *initializer = tensor_proto.clone(),
				None => graph.initializer.push(tensor_proto.clone())
			}
		}

		if !self.dimension_overrides_by_name.is_empty() || !self.dimension_overrides_by_denotation.is_empty() {
			for input in &mut graph.input {
				let Some(type_proto::Value::TensorType(tensor_type)) = input.r#type.as_mut().and_then(|t| t.value.as_mut()) else {
					continue;
				};
				let Some(shape) = tensor_type.shape.as_mut() else {
					continue;
				};
				for dim in &mut shape.dim {
					let value = match &dim.value {
						Some(dimension::Value::DimParam(name)) => self.dimension_overrides_by_name.get(name),
						_ => None
					}
					.or_else(|| self.dimension_overrides_by_denotation.get(&dim.denotation));
					if let Some(value) = value {
						dim.value = Some(dimension::Value::DimValue(*value));
					}
				}
			}
		}
	}
}

#[derive(Clone)]
pub(crate) struct ModelMetadata {
	pub(crate) producer_name: String,
	pub(crate) graph_name: String,
	pub(crate) domain: String,
	pub(crate) description: String,
	pub(crate) graph_description: String,
	pub(crate) version: i64,
	pub(crate) custom_metadata: Vec<(String, String)>
}

impl ModelMetadata {
	fn from_proto(proto: &ModelProto) -> Self {
		Self {
			producer_name: proto.producer_name.clone(),
			graph_name: proto.graph.as_ref().map(|g| g.name.clone()).unwrap_or_default(),
			domain: proto.domain.clone(),
			description: proto.doc_string.clone(),
			graph_description: proto.graph.as_ref().map(|g| g.doc_string.clone()).unwrap_or_default(),
			version: proto.model_version,
			custom_metadata: proto.metadata_props.iter().map(|e| (e.key.clone(), e.value.clone())).collect()
		}
	}
}

pub(crate) struct Outlet {
	pub(crate) name: String,
	pub(crate) dtype: ONNXTensorElementDataType,
	pub(crate) shape: Option<Vec<(i64, CString)>>
}

impl Outlet {
	fn from_proto(value: &ValueInfoProto) -> Result<Self, Error> {
		let Some(type_proto::Value::TensorType(tensor_type)) = value.r#type.as_ref().and_then(|t| t.value.as_ref()) else {
			return Err(Error::new(OrtErrorCode::ORT_NOT_IMPLEMENTED, format!("'{}' is not a tensor; only tensors are supported", value.name)));
		};
		Ok(Self {
			name: value.name.clone(),
			dtype: convert_proto_to_sys(tensor_type.elem_type)?,
			shape: tensor_type.shape.as_ref().map(|shape| {
				shape
					.dim
					.iter()
					.map(|dim| match &dim.value {
						Some(dimension::Value::DimValue(v)) => (*v, CString::default()),
						Some(dimension::Value::DimParam(name)) => (-1, CString::new(name.as_str()).unwrap_or_default()),
						None => (-1, CString::default())
					})
					.collect()
			})
		})
	}

	pub(crate) fn type_info(&self) -> TypeInfo {
		match &self.shape {
			Some(shape) => TypeInfo {
				dtype: self.dtype,
				shape: Some(shape.iter().map(|(d, _)| *d).collect()),
				symbolic_dims: shape.iter().map(|(_, s)| s.clone()).collect()
			},
			None => TypeInfo {
				dtype: self.dtype,
				shape: None,
				symbolic_dims: Vec::new()
			}
		}
	}
}

pub(crate) struct Session {
	pub(crate) model: ModelProto,
	pub(crate) inputs: Vec<Outlet>,
	pub(crate) outputs: Vec<Outlet>,
	pub(crate) metadata: ModelMetadata
}

impl Session {
	pub(crate) fn from_buffer(options: &SessionOptions, data: &[u8]) -> Result<Session, Error> {
		let mut model = ModelProto::decode(data).map_err(|e| Error::new(OrtErrorCode::ORT_INVALID_PROTOBUF, format!("Failed to parse model: {e}")))?;
		options.apply_to_proto(&mut model);

		let Some(graph) = model.graph.as_ref() else {
			return Err(Error::new(OrtErrorCode::ORT_NO_MODEL, "graph is missing?"));
		};

		// graph inputs that have an initializer are always set to the initializer by candle so they arent exposed as
		// inputs
		let initializers: HashSet<&str> = graph.initializer.iter().map(|i| i.name.as_str()).collect();
		let inputs = graph
			.input
			.iter()
			.filter(|i| !initializers.contains(i.name.as_str()))
			.map(Outlet::from_proto)
			.collect::<Result<_, _>>()?;
		let outputs = graph.output.iter().map(Outlet::from_proto).collect::<Result<_, _>>()?;

		Ok(Session {
			metadata: ModelMetadata::from_proto(&model),
			model,
			inputs,
			outputs
		})
	}

	pub(crate) fn run(
		&self,
		inputs: &[(&str, &Tensor)],
		output_names: &[&str],
		run_options: Option<&RunOptions>
	) -> Result<Vec<(candle_core::Tensor, ONNXTensorElementDataType)>, Error> {
		let mut input_values = HashMap::with_capacity(inputs.len());
		for (name, tensor) in inputs {
			if !self.inputs.iter().any(|c| c.name == *name) {
				return Err(Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("unknown input '{name}'")));
			}
			input_values.insert(name.to_string(), tensor.to_candle()?);
		}

		let outputs = output_names
			.iter()
			.map(|name| {
				self.outputs
					.iter()
					.find(|c| c.name == *name)
					.ok_or_else(|| Error::new(OrtErrorCode::ORT_INVALID_ARGUMENT, format!("unknown output '{name}'")))
			})
			.collect::<Result<Vec<_>, Error>>()?;

		if let Some(run_options) = run_options
			&& run_options.terminate.load(Ordering::Acquire)
		{
			return Err(Error::new(OrtErrorCode::ORT_FAIL, "Terminated"));
		}

		let mut results =
			candle_onnx::simple_eval(&self.model, input_values).map_err(|e| Error::new(OrtErrorCode::ORT_FAIL, format!("Failed to run session: {e}")))?;
		outputs
			.into_iter()
			.map(|outlet| {
				let tensor = results
					.remove(&outlet.name)
					.ok_or_else(|| Error::new(OrtErrorCode::ORT_FAIL, format!("Output '{}' was not computed", outlet.name)))?;
				let dtype = match convert_sys_to_dtype(outlet.dtype) {
					// for bool-as-u8
					Ok(dtype) if dtype == tensor.dtype() => outlet.dtype,
					_ => convert_dtype_to_sys(tensor.dtype())?
				};
				Ok((tensor, dtype))
			})
			.collect()
	}
}

pub(crate) unsafe fn write_outputs(
	outputs: Vec<(candle_core::Tensor, ONNXTensorElementDataType)>,
	output_ptrs: &mut [*mut ort_sys::OrtValue]
) -> Result<(), Error> {
	// copy into preallocated outputs first so nothing is allocated if a copy fails
	for ((output, _), output_ptr) in outputs.iter().zip(output_ptrs.iter()) {
		if !output_ptr.is_null() {
			let preallocated = unsafe { &mut *(*output_ptr).cast::<Tensor>() };
			preallocated.copy_from(output)?;
		}
	}

	let mut created = Vec::new();
	for ((output, dtype), output_ptr) in outputs.into_iter().zip(output_ptrs.iter_mut()) {
		if output_ptr.is_null() {
			match Tensor::from_candle(output, dtype) {
				Ok(tensor) => {
					*output_ptr = Box::into_raw(Box::new(tensor)).cast();
					created.push(output_ptr);
				}
				Err(e) => {
					for ptr in created {
						drop(unsafe { Box::from_raw(ptr.cast::<Tensor>()) });
						*ptr = ptr::null_mut();
					}
					return Err(e);
				}
			}
		}
	}
	Ok(())
}

#[derive(Default)]
pub(crate) struct RunOptions {
	pub(crate) tag: CString,
	pub(crate) log_severity_level: i32,
	pub(crate) log_verbosity_level: i32,
	pub(crate) terminate: AtomicBool,
	pub(crate) config: HashMap<String, CString>
}

pub(crate) enum BoundOutput {
	Value(*mut Tensor),
	Device(Option<Tensor>)
}

pub(crate) struct IoBinding {
	pub(crate) session: *const Session,
	pub(crate) inputs: Vec<(String, *const Tensor)>,
	pub(crate) outputs: Vec<(String, BoundOutput)>
}

impl IoBinding {
	pub(crate) fn bind_input(&mut self, name: String, value: *const Tensor) {
		match self.inputs.iter_mut().find(|(n, _)| *n == name) {
			Some((_, v)) => *v = value,
			None => self.inputs.push((name, value))
		}
	}

	pub(crate) fn bind_output(&mut self, name: String, output: BoundOutput) {
		match self.outputs.iter_mut().find(|(n, _)| *n == name) {
			Some((_, v)) => *v = output,
			None => self.outputs.push((name, output))
		}
	}

	pub(crate) fn run(&mut self, run_options: Option<&RunOptions>) -> Result<(), Error> {
		let session = unsafe { &*self.session };
		let inputs: Vec<(&str, &Tensor)> = self.inputs.iter().map(|(name, value)| (name.as_str(), unsafe { &**value })).collect();
		let output_names: Vec<&str> = self.outputs.iter().map(|(name, _)| name.as_str()).collect();
		let results = session.run(&inputs, &output_names, run_options)?;
		for ((_, output), (res, dtype)) in self.outputs.iter_mut().zip(results) {
			match output {
				BoundOutput::Value(value) => unsafe { &mut **value }.copy_from(&res)?,
				BoundOutput::Device(slot) => *slot = Some(Tensor::from_candle(res, dtype)?)
			}
		}
		Ok(())
	}

	pub(crate) fn output_values(&self) -> Result<Vec<Tensor>, Error> {
		self.outputs
			.iter()
			.map(|(name, output)| match output {
				BoundOutput::Value(value) => unsafe { &**value }.copy(),
				BoundOutput::Device(Some(tensor)) => tensor.copy(),
				BoundOutput::Device(None) => Err(Error::new(OrtErrorCode::ORT_FAIL, format!("output '{name}' has not been computed")))
			})
			.collect()
	}
}
