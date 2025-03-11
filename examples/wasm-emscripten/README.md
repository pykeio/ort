# WASM-Emscripten Example
Example how to use `ort` to run `onnxruntime` in the Web with multi-threaded execution. Here inferencing the YoloV8 model.

## Prepare environment
1. Requires a recent Rust installation via `rustup`, `git`, `cmake` and `build-essentials`, `libssl-dev`, and `pkg-config` under Ubuntu or `xcode-select` under macOS.
1. Install the Rust nightly toolchain with `rustup install nightly`.
1. Add Emscripten as Rust target with `rustup target add wasm32-unknown-emscripten --toolchain nightly`.
1. Clone Emscripten SDK via `git clone https://github.com/emscripten-core/emsdk.git --depth 1`.
1. Install Emscripten SDK 4.0.4 locally to [match version used in ONNX runtime](https://github.com/microsoft/onnxruntime/blob/fe7634eb6f20b656a3df978a6a2ef9b3ea00c59d/.gitmodules#L10) via `./emsdk/emsdk install 4.0.4`.
1. Prepare local Emscripten SDK via `./emsdk/emsdk activate 4.0.4`.

Environment tested on Ubuntu 24.04 and macOS 14.7.1.

## Build example
1. Set local Emscripten SDK in current session via `source ./emsdk/emsdk_env.sh`.
1. Build the example via `cargo build` for a debug build or `cargo build --release` for a release build.

## Serve example
1. Serve a debug build via `python3 serve.py` or a release build via `python3 serve.py --release`. Pre-installed Python 3 should be sufficient.