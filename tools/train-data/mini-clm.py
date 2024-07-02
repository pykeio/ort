import math

import onnx
from onnxruntime.training import artifacts
import torch
from torch import nn, Tensor
from torch.nn import functional as F

class RMSNorm(nn.Module):
	def __init__(self, dim: int, *, eps: float = 1e-6):
		super().__init__()

		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def forward(self, x: Tensor) -> Tensor:
		if x.dtype != torch.float32:
			xf = x.to(dtype=torch.float32)
		else:
			xf = x
		output = (xf * torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))
		if x.dtype != torch.float32:
			output = output.to(dtype=x.dtype)
		return output * self.weight

class RoPE(nn.Module):
	def __init__(self, embedding_dim: int, *, max_seq_length: int = 2048, base: float = 10000.0):
		super().__init__()

		pe = torch.zeros(max_seq_length, embedding_dim)
		position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, embedding_dim, step=2).float() * (-math.log(base) / embedding_dim))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe, persistent=False)

	@torch.no_grad()
	def forward(self, x: Tensor) -> Tensor:
		return x + self.pe[:, :x.shape[1], :]

class Attention(nn.Module):
	def __init__(self, embedding_dim: int, *, rope: RoPE, max_seq_length: int = 2048, n_heads: int = 4):
		super().__init__()

		self.embedding_dim = embedding_dim
		self.n_heads = n_heads
		self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
		self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
		self.rope = rope
		self.register_buffer('bias', torch.tril(torch.ones(max_seq_length, max_seq_length))[None, None, :, :], persistent=False)

	def forward(self, x: Tensor) -> Tensor:
		b, t, c = x.size()

		x = self.rope(x)

		q, k, v = self.qkv(x).split(self.embedding_dim, dim=2)
		q = q.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
		k = k.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
		v = v.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)

		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		y = att @ v
		y = y.transpose(1, 2).contiguous().view(b, t, c)

		return self.proj(y)

class FFN(nn.Module):
	def __init__(self, embedding_dim: int, intermediate_dim: int | None = None):
		super().__init__()

		intermediate_dim = intermediate_dim or embedding_dim * 4

		self.w1 = nn.Linear(embedding_dim, intermediate_dim * 2, bias=False)
		self.w2 = nn.Linear(intermediate_dim, embedding_dim, bias=False)

	def forward(self, x: Tensor) -> Tensor:
		x, gate = self.w1(x).chunk(2, dim=-1)
		return self.w2(F.gelu(gate) * x)

class Layer(nn.Module):
	def __init__(self, embedding_dim: int, rope: RoPE):
		super().__init__()

		self.attn = Attention(embedding_dim, rope=rope)
		self.norm1 = RMSNorm(embedding_dim)
		self.ffn = FFN(embedding_dim)
		self.norm2 = RMSNorm(embedding_dim)

	def forward(self, x: Tensor) -> Tensor:
		x = x + self.attn(self.norm1(x))
		x = x + self.ffn(self.norm2(x))
		return x

class CLM(nn.Module):
	def __init__(self, embedding_dim: int, n_layers: int, *, vocab_size: int):
		super().__init__()

		rope = RoPE(embedding_dim)
		self.layers = nn.ModuleList([Layer(embedding_dim, rope=rope) for _ in range(n_layers)])
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.norm = RMSNorm(embedding_dim)
		self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

	def forward(self, x: Tensor) -> Tensor:
		x = self.word_embeddings(x)
		for layer in self.layers:
			x = layer(x)
		logits = self.lm_head(self.norm(x))
		return logits.view(-1, logits.size(-1))

lm = CLM(256, 4, vocab_size=50257)
torch.onnx.export(
	lm,
	torch.randint(0, 50256, (1, 64)),
	f'tools/train-data/mini-clm/model.onnx',
	input_names=['input_ids'],
	output_names=['probs'],
	dynamic_axes={
		'input_ids': {0: 'batch', 1: 'seq'},
		'probs': {0: 'batch_seq'}
	},
	opset_version=14
)

onnx_model = onnx.load('tools/train-data/mini-clm/model.onnx')
requires_grad = [param.name for param in onnx_model.graph.initializer]

artifacts.generate_artifacts(
	onnx_model,
	requires_grad=requires_grad,
	frozen_params=[],
	loss=artifacts.LossType.CrossEntropyLoss,
	optimizer=artifacts.OptimType.AdamW,
	artifact_directory='tools/train-data/mini-clm'
)
