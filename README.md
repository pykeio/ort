<div align=center>
    <img src="https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/docs/trend-banner.png" width="350px">
    <hr />
    <a href="https://app.codecov.io/gh/pykeio/ort" target="_blank"><img alt="Coverage Results" src="https://img.shields.io/codecov/c/gh/pykeio/ort?style=for-the-badge"></a> <a href="https://crates.io/crates/ort" target="_blank"><img alt="Crates.io" src="https://img.shields.io/crates/d/ort?style=for-the-badge"></a> <a href="https://opencollective.com/pyke-osai" target="_blank"><img alt="Open Collective backers and sponsors" src="https://img.shields.io/opencollective/all/pyke-osai?style=for-the-badge&label=sponsors"></a>
    <br />
    <a href="https://crates.io/crates/ort" target="_blank"><img alt="Crates.io" src="https://img.shields.io/crates/v/ort?style=for-the-badge&label=ort&logo=rust"></a> <img alt="ONNX Runtime" src="https://img.shields.io/badge/onnxruntime-v1.23.2-blue?style=for-the-badge&logo=cplusplus">
</div>

`ort` is a Rust interface for performing hardware-accelerated inference & training on machine learning models in the [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format.

Based on the now-inactive [`onnxruntime-rs`](https://github.com/nbigaouette/onnxruntime-rs) crate, `ort` is primarily a wrapper for Microsoft's [ONNX Runtime](https://onnxruntime.ai/) library, but offers support for [other pure-Rust runtimes](https://ort.pyke.io/backends).

`ort` with ONNX Runtime is super quick - and it supports almost [any hardware accelerator](https://ort.pyke.io/perf/execution-providers) you can think of. Even still, it's light enough to run on your users' devices.

When you need to deploy a PyTorch/TensorFlow/Keras/scikit-learn/PaddlePaddle model either on-device or in the datacenter, `ort` has you covered.

## 📖 Documentation
- [Guide](https://ort.pyke.io/)
- [API reference](https://docs.rs/ort/2.0.0-rc.11/ort/)
- [Examples](https://github.com/pykeio/ort/tree/main/examples)
- [Migrating from v1.x to v2.0](https://ort.pyke.io/migrating/v2)

## 🤔 Support
- [Discord: `#🦀｜ort-general`](https://discord.gg/uQtsNu2xMa)
- [GitHub Discussions](https://github.com/pykeio/ort/discussions)

## 🌠 Sponsor `ort`
<a href="https://opencollective.com/pyke-osai">
<img src="https://opencollective.com/pyke-osai/sponsors.svg" />
<br />
<img src="https://opencollective.com/pyke-osai/backers.svg" />
</a>

## 💖 FOSS projects using `ort`
<sub>[Open a PR](https://github.com/pykeio/ort/pulls) to add your project here 🌟</sub>

<!--
    This section only showcases projects with OSI-approved licenses: https://opensource.org/licenses
    Businesses that sponsor pyke will appear at the top of the README instead!
-->

- **[Koharu](https://github.com/mayocream/koharu)** uses `ort` to detect, OCR, and inpaint manga pages.
- **[BoquilaHUB](https://github.com/boquila/boquilahub/)** uses `ort` for local AI deployment in biodiversity conservation efforts.
- **[Magika](https://github.com/google/magika)** uses `ort` for content type detection.
- **[Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference)** uses `ort` to deliver high-performance ONNX runtime inference for text embedding models.
- **[`sbv2-api`](https://github.com/neodyland/sbv2-api)** is a fast implementation of Style-BERT-VITS2 text-to-speech using `ort`.
- **[CamTrap Detector](https://github.com/bencevans/camtrap-detector)** uses `ort` to detect animals, humans and vehicles in trail camera imagery.
- **[oar-ocr](https://github.com/GreatV/oar-ocr)** A comprehensive OCR library, built in Rust with `ort` for efficient inference.
- **[retto](https://github.com/NekoImageLand/retto)** uses `ort` for reliable, fast ONNX inference of PaddleOCR models on Desktop and WASM platforms.
- **[Ahnlich](https://github.com/deven96/ahnlich)** uses `ort` to power their AI proxy for semantic search applications.
- **[Valentinus](https://github.com/kn0sys/valentinus)** uses `ort` to provide embedding model inference inside LMDB.
- **[edge-transformers](https://github.com/npc-engine/edge-transformers)** uses `ort` for accelerated transformer model inference at the edge.
- **[`FastEmbed-rs`](https://github.com/Anush008/fastembed-rs)** uses `ort` for generating vector embeddings, reranking locally.
- **[Ortex](https://github.com/relaypro-open/ortex)** uses `ort` for safe ONNX Runtime bindings in Elixir.
- **[SilentKeys](https://github.com/gptguy/silentkeys)** uses `ort` for fast, on-device real-time dictation with NVIDIA Parakeet and Silero VAD.
