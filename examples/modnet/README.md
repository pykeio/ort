# Example of using ModNet to matt images
This example shows how to use ModNet to matt images. The example is based on the [ModNet](https://github.com/ZHKKKe/MODNet) repository.

## Installation
Download the onnx model from [here](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing).

## Usage
```bash
cargo run --example modnet -- /path/to/modnet_photographic_portrait_matting.onnx 
```
e.g.
```bash
cargo run --example modnet -- /Volumes/Extremessd/ort-modnet/models/modnet_photographic_portrait_matting.onnx
```