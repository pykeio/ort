# Phi-3 Vision ONNX Example

This example demonstrates the usage of Microsoft's [Phi-3 Vision model](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu)

Phi-3 Vision ONNX is a multimodal model that combines vision and language processing. It uses three interconnected ONNX models:

- Vision model: Processes images to extract visual features
- Text embedding model: Embeds input text into a format compatible with the model
- Text generation model: Produces text outputs based on the combined visual and textual inputs

This multi-model structure requires a coordinated process:

1. Image Processing:
   - Preprocess the input image
   - Pass it through the vision ONNX model for visual features

2. Text Embedding:
   - Tokenize input text
   - Process it with the text embedding ONNX model

3. Multimodal Fusion:
   - Combine visual features and text embeddings into a single input

4. Text Generation:
   - The combined input is fed into the text generation ONNX model.
   - The model generates text tokens one by one in an autoregressive manner.
   - For each token, the model uses past key/value states to maintain context.

The specific configuration for the model can be found in `data/genai_config.json`.

## Limitations and Performance

This example currently only supports single image input.

The performance of ONNX-based LLM inference can be relatively slow, especially on CPU:

- On an Apple M1 Pro:
  - For image+text input (about 300 tokens): ~7 tokens/s
  - For text-only input (about 10 tokens): ~5 tokens/s

## Run this Example

Before running the example, you'll need to download the ONNX model files to the `data` directory. At present, the `SessionBuilder.commit_from_url` method doesn't support initialization for models split into `.onnx` and `.onnx.data` files, which is the case for Phi-3 Vision models.

To get started, use the `/data/download.sh` script to download the following three model files:

1. `phi-3-v-128k-instruct-vision.onnx` and `phi-3-v-128k-instruct-vision.onnx.data`
2. `phi-3-v-128k-instruct-text-embedding.onnx` and `phi-3-v-128k-instruct-text-embedding.onnx.data`
3. `phi-3-v-128k-instruct-text.onnx` and `phi-3-v-128k-instruct-text.onnx.data`
4. `tokenizer.json`

Once the model files are downloaded, you can run the example using Cargo:

```bash
cargo run -p phi-3-vision
```
