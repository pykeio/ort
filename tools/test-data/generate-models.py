import inspect

import numpy as np
import onnx
import onnx.helper as G
import onnxruntime as ort

def make_tensor_from_np(name: str, arr: np.ndarray) -> onnx.TensorProto:
	match arr.dtype:
		case np.float32:
			dtype = onnx.TensorProto.FLOAT
	shape = list(arr.shape)
	return G.make_tensor(name, dtype, shape, arr)

factories = []
def model_factory(func):
	model_name = f'{func.__name__}.onnx'
	def wrapper():
		try:
			model = func()
		except Exception as e:
			print(f'Failed to create `{model_name}`: {e}')
			return
		if isinstance(model, onnx.GraphProto):
			model = G.make_model(model, opset_imports=[onnx.OperatorSetIdProto(domain=None, version=21)])
		try:
			onnx.checker.check_model(model)
		except Exception as e:
			print(f'`{model_name}` is invalid: {e}')
		onnx.save_model(model, f'tests/data/{model_name}')
	factories.append(wrapper)
	return wrapper
def misc_factory(func):
	def wrapper():
		try:
			func()
		except Exception as e:
			pass
	factories.append(wrapper)
	return wrapper

class Models:
	@model_factory
	def lora_model():
		input_x = G.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [4, 4])

		lora_param_a_input = G.make_tensor_value_info('lora_param_a', onnx.TensorProto.FLOAT, [4, 'dim'])
		lora_param_b_input = G.make_tensor_value_info('lora_param_b', onnx.TensorProto.FLOAT, ['dim', 4])

		output = onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [4, 4])

		weight_x = make_tensor_from_np('weight_x', np.array(range(1, 17)).reshape(4, 4).astype(np.float32))

		lora_param_a = make_tensor_from_np('lora_param_a', np.zeros([4, 0], dtype=np.float32))
		lora_param_b = make_tensor_from_np('lora_param_b', np.zeros([0, 4], dtype=np.float32))

		matmul_x = G.make_node('MatMul', ['input', 'weight_x'], ['mm_output_x'])
		matmul_a = G.make_node('MatMul', ['input', 'lora_param_a'], ['mm_output_a'])
		matmul_b = G.make_node('MatMul', ['mm_output_a', 'lora_param_b'], ['mm_output_b'])
		add_node = G.make_node('Add', ['mm_output_x', 'mm_output_b'], ['output'])

		return G.make_graph(
			nodes=[matmul_x, matmul_a, matmul_b, add_node],
			inputs=[input_x, lora_param_a_input, lora_param_b_input],
			outputs=[output],
			initializer=[weight_x, lora_param_a, lora_param_b],
			name='lora_test'
		)
	
	@misc_factory
	def lora_adapter():
		param_a = ort.OrtValue.ortvalue_from_numpy(np.array([[3], [4], [5], [6]], dtype=np.float32))
		param_b = ort.OrtValue.ortvalue_from_numpy(np.array([[7, 8, 9, 10]], dtype=np.float32))

		adapter = ort.AdapterFormat()
		adapter.set_parameters({
			'lora_param_a': param_a,
			'lora_param_b': param_b
		})
		adapter.export_adapter('tests/data/adapter.orl')

if __name__ == '__main__':
	for model_factory in factories:
		model_factory()
