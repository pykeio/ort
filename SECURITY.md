# Reporting security concerns
Security concerns should be reported to [contact@pyke.io](mailto:contact@pyke.io?subject=ort%20vulnerability%20disclosure). **Do not report vulnerabilities in GitHub issues** or any other public forum (GitHub Discussions, Discord).

When making a report, ensure that the issue is actionable by `ort` or one of its alternative backends - `ort-candle`, `ort-tract`, and `ort-web`. For example: a buffer overflow caused by a bad session input name *is* actionable; an RCE caused by a maliciously crafted `.onnx` file *is not* actionable (as `ort` itself does not handle model loading), and we suggest you report the issue to the underlying runtime instead.

For issues affecting ONNX Runtime in general, see [Microsoft's security disclosure information](https://github.com/microsoft/onnxruntime/blob/main/SECURITY.md).

## Maintained versions
1.x branches of `ort` are not maintained and will not receive security patches.

After version 2.0.0 is stable, all minor versions (2.x) will receive security patches.
