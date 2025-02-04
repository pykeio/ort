# WASM-Emscripten Example
Example how to use `ort` to run `onnxruntime` in the Web with multi-threaded execution. Here inferencing the YoloV8 model.

## Prepare environment
1. Requires a recent Rust installation via `rustup`, `git`, `cmake` and `build-essentials`, `libssl-dev`, and `pkg-config` under Ubuntu or `xcode-select` under macOS.
1. Install the Rust nightly toolchain with `rustup install nightly`.
1. Add Emscripten as Rust target with `rustup target add wasm32-unknown-emscripten --toolchain nightly`.
1. Clone Emscripten SDK via `git clone https://github.com/emscripten-core/emsdk.git --depth 1`.
1. Install Emscripten SDK 3.1.59 locally to [match version used in ONNX runtime](https://github.com/microsoft/onnxruntime/blob/1d97d6ef55433298dee58634b0ea59f736e8a72e/.gitmodules#L10) via `./emsdk/emsdk install 3.1.59`.
1. Prepare local Emscripten SDK via `./emsdk/emsdk activate 3.1.59`.

Environment tested on Ubuntu 24.04 and macOS 14.7.1.

## Build example
1. Set local Emscripten SDK in current session via `source ./emsdk/emsdk_env.sh`.
1. Build the example via `cargo build --release`.

## Serve example
1. Serve the example via `python3 serve.py`. Pre-installed Python 3 should be sufficient.