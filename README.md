<div align=center>
	<h1><code>ort</code> - ONNX Runtime Rust bindings</h1>
    <a href="https://app.codecov.io/gh/pykeio/ort" target="_blank"><img alt="Coverage Results" src="https://img.shields.io/codecov/c/gh/pykeio/ort?style=for-the-badge"></a> <a href="https://github.com/pykeio/ort/actions/workflows/test.yml"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/workflow/status/pykeio/ort/Run%20cargo%20tests?style=for-the-badge"></a>
</div>

`ort` is yet another ONNX Runtime wrapper for Rust based on [`onnxruntime-rs`](https://github.com/nbigaouette/onnxruntime-rs). `ort` is updated for ONNX Runtime 1.13.1 and contains many API improvements & fixes.

## Cargo features
- `fetch-models`: Enables fetching models from the ONNX Model Zoo; not recommended for production.
- `generate-bindings`: Update/generate ONNX Runtime bindings with `bindgen`. Requires [libclang](https://clang.llvm.org/doxygen/group__CINDEX.html).
- `copy-dylibs`: Copy dynamic libraries to the Cargo `target` folder. Highly recommended on Windows, where the operating system may have an older version of ONNX Runtime bundled.
- `prefer-system-strategy`: Uses the `system` compile strategy by default; requires users to provide ONNX Runtime libraries.
    - `prefer-dynamic-libs`: By default, if the path pointed to by `ORT_LIB_LOCATION` contains static libraries, `ort` will link to them rather than dynamic libraries. This feature prefers linking to dynamic libraries instead.
- `prefer-compile-strategy`: Uses the `compile` strategy by default; will take a *very* long time and is currently untested, but allows for easy static linking, avoiding [the DLL hell](#shared-library-hell).
    - `compile-static`: Compiles ONNX Runtime as a static library.
    - `mimalloc`: Uses the (usually) faster mimalloc memory allocation library instead of the platform default.
    - `experimental`: Compiles Microsoft experimental operators.
    - `training`: Enables training via ONNX Runtime. Currently unavailable through high-level bindings.
    - `minimal-build`: Builds ONNX Runtime without ONNX model loading. Drastically reduces size. Recommended for release builds.
- **Execution providers**: These are required for both building **and** using execution providers. Do not enable any of these features unless you are using the `compile` strategy or you are using the `system` strategy with binaries that support these execution providers, otherwise you'll run into linking errors.
    - `cuda`: Enables the CUDA execution provider for Maxwell (7xx) NVIDIA GPUs and above. Requires CUDA v11.6+.
    - `tensorrt`: Enables the TensorRT execution provider for GeForce 9xx series NVIDIA GPUs and above; requires CUDA v11.6+ and TensorRT v8.4+.
    - `openvino`: Enables the OpenVINO execution provider for 6th+ generation Intel Core CPUs.
    - `onednn`: Enables the oneDNN execution provider for x86/x64 targets.
    - `directml`: Enables the DirectML execution provider for Windows x86/x64 targets with dedicated GPUs supporting DirectX 12.
    - `snpe`: Enables the SNPE execution provider for Qualcomm Snapdragon CPUs & Adreno GPUs.
    - `nnapi`: Enables the Android Neural Networks API (NNAPI) execution provider.
    - `coreml`: Enables the CoreML execution provider for macOS/iOS targets.
    - `xnnpack`: Enables the [XNNPACK](https://github.com/google/XNNPACK) backend for WebAssembly and Android.
    - `rocm`: Enables the ROCm execution provider for AMD ROCm-enabled GPUs.
    - `acl`: Enables the ARM Compute Library execution provider for multi-core ARM v8 processors.
    - `armnn`: Enables the ArmNN execution provider for ARM v8 targets.
    - `tvm`: Enables the **preview** Apache TVM execution provider.
    - `migraphx`: Enables the MIGraphX execution provider for Windows x86/x64 targets with dedicated AMD GPUs.
    - `rknpu`: Enables the RKNPU execution provider for Rockchip NPUs.
    - `vitis`: Enables Xilinx's Vitis-AI execution provider for U200/U250 accelerators.
    - `cann`: Enables the Huawei Compute Architecture for Neural Networks (CANN) execution provider.
- `half`: Builds support for `float16`/`bfloat16` ONNX tensors.
- `use-half-intrinsics`: Use intrinsics in the `half` crate for faster operations when dealing with `float16`/`bfloat16` ONNX tensors.

## Execution providers
To use other execution providers, you must explicitly enable them via their Cargo features. Using the `compile` strategy, everything should just work™️. Using the `system` strategy, ensure that the binaries you are linking to have been built with the execution providers you want to use, otherwise you'll get linking errors. After that, configuring & enabling these execution providers can be done through `SessionBuilder::execution_providers()`.

Requesting an execution provider via e.g. `ExecutionProviderBuilder::cuda()` will silently fail if that EP is not available on the system or encounters an error and falls back to the next requested execution provider or to the CPU provider if no requested providers are available. If you must know why the execution provider is unavailable, use `ExecutionProviderBuilder::try_*()`, e.g. `try_cuda()`.

For prebuilt Microsoft binaries, you can enable the CUDA or TensorRT execution providers for Windows and Linux via the `cuda` and `tensorrt` Cargo features respectively. **No other execution providers are supported** in these binaries and enabling other features will fail. To use other execution providers, you must build ONNX Runtime yourself to be able to use them.

## Shared library hell
Because compiling ONNX Runtime from source takes so long (and static linking is not recommended by Microsoft), it may be easier to compile ONNX Runtime as a shared library or use prebuilt DLLs. However, this can cause some issues with library paths and load orders.

### Windows
Some versions of Windows come bundled with `onnxruntime.dll` in the System32 folder. On Windows 11 build 22598.1, `onnxruntime.dll` is on version 1.10.0, while `ort` requires 1.13.1, so execution will fail because [system DLLs take precedence](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order). Luckily though, DLLs in the same folder as the application have higher priority; `ort` can automatically copy the DLLs to the Cargo target folder when the `copy-dylibs` feature is enabled.

Note that, when running tests/benchmarks for the first time, you'll have to manually copy the `target/debug/onnxruntime*.dll` files to `target/debug/deps/`, or `target/debug/examples/` for examples. It should Just Work™️ when building/running a binary, however.

### Linux
You'll either have to copy `libonnxruntime.so` to a known lib location (e.g. `/usr/lib`) or enable rpath if you have the `copy-dylibs` feature enabled.

In `Cargo.toml`:
```toml
[profile.dev]
rpath = true

[profile.release]
rpath = true

# do this for all profiles
```

In `.cargo/config.toml`:
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN" ]

# do this for all Linux targets as well
```

### macOS
macOS follows the same procedure as Linux, except the rpath should point to `@loader_path` rather than `$ORIGIN`:

```toml
# .cargo/config.toml
[target.x86_64-apple-darwin]
rustflags = [ "-Clink-args=-Wl,-rpath,@loader_path" ]
```
