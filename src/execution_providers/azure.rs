use super::{ExecutionProvider, ExecutionProviderOptions, RegisterError};
use crate::{error::Result, session::builder::SessionBuilder};

/// [Azure Execution Provider](https://onnxruntime.ai/docs/execution-providers/Azure-ExecutionProvider.html) enables
/// operators that invoke Azure cloud models.
///
/// The Azure EP enables the use of [`onnxruntime-extensions`](https://github.com/microsoft/onnxruntime-extensions)'
/// [Azure operators](https://github.com/microsoft/onnxruntime-extensions/blob/v0.14.0/docs/custom_ops.md#azure-operators).
///
/// # Example
/// Using an example model generated in Python with:
/// ```python
/// from onnx import *
///
/// azure_model_uri = "https://myname-aoai-test.openai.azure.com/openai/deployments/mydeploy/chat/completions?api-version=2023-05-15"
///
/// auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
/// chat = helper.make_tensor_value_info('chat', TensorProto.STRING, [-1])
/// response = helper.make_tensor_value_info('response', TensorProto.STRING, [-1])
///
/// invoker = helper.make_node(
/// 	'AzureTextToText',
/// 	[ 'auth_token', 'chat' ],
/// 	[ 'response' ],
/// 	domain='com.microsoft.extensions',
/// 	name='chat_invoker',
/// 	model_uri=azure_model_uri
/// )
///
/// graph = helper.make_graph([ invoker ], 'graph', [ auth_token, chat ], [ response ])
/// model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
///
/// onnx.save(model, 'azure_chat.onnx')
/// ```
///
/// To use this model in `ort`:
/// ```no_run
/// # use ort::{execution_providers::azure::AzureExecutionProvider, session::Session, value::Tensor};
/// # fn main() -> ort::Result<()> {
/// let mut session = Session::builder()?
/// 	// note: session must be initialized with `onnxruntime-extensions`
/// 	.with_extensions()?
/// 	.with_execution_providers([AzureExecutionProvider::default().build()])?
/// 	.commit_from_file("azure_chat.onnx")?;
///
/// let auth_token = Tensor::from_string_array(([1], &*vec!["..."]))?;
/// let input = Tensor::from_string_array((
/// 	[1],
/// 	&*vec![
/// 		r#"
/// 			{
/// 				"messages": [
/// 					{ "role": "system", "content": "You are a helpful assistant." },
/// 					{ "role": "user", "content": "Does Azure OpenAI support customer managed keys?" },
/// 					{ "role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI." },
/// 					{ "role": "user", "content": "Do other Azure AI services support this too?" }
/// 				]
/// 			}
/// 		"#,
/// 	]
/// ))?;
/// let outputs = session.run(ort::inputs![input])?;
/// let (_, response_json_strings) = &outputs[0].try_extract_strings()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone)]
pub struct AzureExecutionProvider {
	options: ExecutionProviderOptions
}

super::impl_ep!(arbitrary; AzureExecutionProvider);

impl ExecutionProvider for AzureExecutionProvider {
	fn name(&self) -> &'static str {
		"AzureExecutionProvider"
	}

	fn supported_by_platform(&self) -> bool {
		cfg!(any(target_os = "linux", target_os = "windows", target_os = "android"))
	}

	#[allow(unused, unreachable_code)]
	fn register(&self, session_builder: &mut SessionBuilder) -> Result<(), RegisterError> {
		#[cfg(any(feature = "load-dynamic", feature = "azure"))]
		{
			use crate::{AsPointer, ortsys};

			let ffi_options = self.options.to_ffi();
			ortsys![unsafe SessionOptionsAppendExecutionProvider(
				session_builder.ptr_mut(),
				c"AZURE".as_ptr().cast::<core::ffi::c_char>(),
				ffi_options.key_ptrs(),
				ffi_options.value_ptrs(),
				ffi_options.len(),
			)?];
			return Ok(());
		}

		Err(RegisterError::MissingFeature)
	}
}
