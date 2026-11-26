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
			model = G.make_model(model, ir_version=13, opset_imports=[onnx.OperatorSetIdProto(domain=None, version=21)])
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

	@model_factory
	def mul_add():
		input_x = G.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['batch', 3])
		output_y = onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ['batch', 3])
		output_z = onnx.helper.make_tensor_value_info('z', onnx.TensorProto.FLOAT, ['batch', 3])

		add_bias = make_tensor_from_np('bias', np.array([1, 2, 3], dtype=np.float32))
		mul_weight = make_tensor_from_np('weight', np.array([2], dtype=np.float32))

		mul = G.make_node('Mul', ['x', 'weight'], ['y'])
		add = G.make_node('Add', ['x', 'bias'], ['z'])

		return G.ModelProto(
			ir_version=13,
			graph=G.make_graph(
				nodes=[mul, add],
				inputs=[input_x],
				outputs=[output_y, output_z],
				initializer=[add_bias, mul_weight],
				name='mul_add'
			),
			opset_import=[onnx.OperatorSetIdProto(domain=None, version=13)],
			domain='io.pyke.ort',
			model_version=42,
			producer_name='generate-models.py'
		)

	@model_factory
	def passthrough():
		input_x = G.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['n'])
		output_y = G.make_tensor_value_info('y', onnx.TensorProto.FLOAT, ['n'])

		return G.ModelProto(
			ir_version=13,
			graph=G.make_graph(
				nodes=[G.make_node('Neg', ['x'], ['y'])],
				inputs=[input_x],
				outputs=[input_x, output_y],
				name='passthrough'
			),
			opset_import=[onnx.OperatorSetIdProto(domain=None, version=13)],
			producer_name='generate-models.py'
		)

	@misc_factory
	def add_constant_ort():
		import os, tempfile

		input_x = G.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [64])
		output_y = G.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [64])
		bias = make_tensor_from_np('bias', np.arange(64, dtype=np.float32))
		model = G.ModelProto(
			ir_version=13,
			graph=G.make_graph(
				nodes=[G.make_node('Add', ['x', 'bias'], ['y'])],
				inputs=[input_x],
				outputs=[output_y],
				initializer=[bias],
				name='add_constant'
			),
			opset_import=[onnx.OperatorSetIdProto(domain=None, version=13)],
			producer_name='generate-models.py'
		)
		onnx.checker.check_model(model)

		with tempfile.TemporaryDirectory() as temp_dir:
			onnx_path = os.path.join(temp_dir, 'add_constant.onnx')
			onnx.save_model(model, onnx_path)

			options = ort.SessionOptions()
			options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
			options.optimized_model_filepath = 'tests/data/add_constant.ort'
			options.add_session_config_entry('session.save_model_format', 'ORT')
			_ = ort.InferenceSession(onnx_path, options, providers=['CPUExecutionProvider'])

	@model_factory
	def custom_op_inplace():
		input_x = G.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [4])
		output_y = G.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [4])

		return G.ModelProto(
			ir_version=10,
			graph=G.make_graph(
				nodes=[
					G.make_node('Neg', ['x'], ['neg_x']),
					G.make_node('AddOne', ['neg_x'], ['added'], name='add_one', domain='test.customop'),
					G.make_node('Neg', ['added'], ['y'])
				],
				inputs=[input_x],
				outputs=[output_y],
				name='custom_op_inplace'
			),
			opset_import=[
				onnx.OperatorSetIdProto(domain='', version=19),
				onnx.OperatorSetIdProto(domain='test.customop', version=1)
			],
			producer_name='generate-models.py'
		)

if __name__ == '__main__':
	for model_factory in factories:
		model_factory()
