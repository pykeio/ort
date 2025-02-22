# Training Examples

## `train-clm`
This example trains a tiny causal language model on a small subset of pyke's [**OshiChats v2**](https://huggingface.co/datasets/pykeio/oshichats-v2), a dataset of live text chat messages collected from various [VTuber](https://en.wikipedia.org/wiki/VTuber) live streams. The model is not particularly useful or interesting (due to both the low-quality dataset and small model size), but it showcases that entire language models can be trained from scratch entirely in Rust on (almost) any device.

To get started, create a Python virtual environment and install the following packages:
```
pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu==1.18.0 onnx~=1.17 torch~=2.3
```

We're installing the CPU version of the `onnxruntime-training` & `torch` packages because we only need to use Python to *create* the initial graph which will be used for training. Run `python tools/train-data/mini-clm.py` from the root directory of the `ort` repo to create the training artifacts.

Next, we need to convert our dataset into tokens to feed the model. This can be achieved by downloading the `oshicats-v2.jsonl` file from the OshiChats v2 dataset and running `cargo run -p example-training --example pretokenize -- ~/oshichats-v2.jsonl`, or if you (rightfully) don't wish to waste 30 GB worth of disk space and bandwidth on brainrot, you may download a [1 MB pre-tokenized subset of the dataset](https://cdn.pyke.io/0/pyke:ort-rs/example-auxiliary-data@0.0.0/train-clm-dataset.bin) (make sure `train-clm-dataset.bin` is in the root of the `ort` repo).

Finally, we can train our model! Run `cargo run -p example-training --example train-clm` to start training. If you have an NVIDIA GPU, add `--features cuda` to enable CUDA, though it's not required and you can train directly on CPU instead. **This will use ~8 GB of (V)RAM!** You can lower the memory usage by adjusting the `BATCH_SIZE` and `SEQUENCE_LENGTH` constants in `train-clm.rs`, though note that changing the batch size may require adjustments to the learning rate.

While training, the progress bar will show the cross-entropy loss at each training step. At the end of training, the final trained model will be saved to `trained-clm.onnx`, and the program will use the model to generate a small snippet of text:
```
100%|██████████████████████████████████████| 5000/5000 [06:29<00:00, 12.83it/s, loss=3.611]
I'm so much better than the game<|endoftext|>I think you can't see it<|endoftext|>I think you can't see it<|endoftext|>I think so it's a new game<|endoftext|>I think I'm sure you can't see what you can't see it<|endoftext|>
```

Not bad, considering the model & dataset size! This example can easily be scaled up to pre-train or fine-tune (both full-parameter and PEFT) larger language models like Llama/Phi, so long as you have enough compute.

## `train-clm-simple`
This example is functionally identical to `train-clm`, except it uses ort's "simple" Trainer API instead of implementing a manual training loop. The simple API is more akin to 🤗 Transformer's [`Trainer`](https://huggingface.co/docs/transformers/en/main_classes/trainer) API or [PyTorch Lightning](https://lightning.ai/pytorch-lightning). With the simple API, all you have to do is pass a data loader & parameters, and let `ort` handle training for you!
