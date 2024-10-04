#!/bin/bash

BASE_URL="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cpu/resolve/main/cpu-int4-rtn-block-32-acc-level-4/"
FILES=(
    "phi-3-v-128k-instruct-text-embedding.onnx"
    "phi-3-v-128k-instruct-text-embedding.onnx.data"
    "phi-3-v-128k-instruct-text.onnx"
    "phi-3-v-128k-instruct-text.onnx.data"
    "phi-3-v-128k-instruct-vision.onnx"
    "phi-3-v-128k-instruct-vision.onnx.data"
    "tokenizer.json"
)

for FILE in "${FILES[@]}"; do
    wget "${BASE_URL}${FILE}"
done
