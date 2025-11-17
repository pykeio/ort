# Contributing to `ort`
Thank you for contributing to ort =) Any contribution is welcome, whether its a huge refactor or a simple typo fix!

## ⚠️ AI assistance notice
This project does not allow the usage of artificial intelligence agents to write or architect code or documentation. PRs containing AI-generated or AI-assisted code/docs will be rejected. Special exception is granted to very menial tasks like [filling out long `match` arms](https://github.com/pykeio/ort/blob/d931f0ac509ec075cdc81f3c4872c09b7122d752/src/tensor/types.rs#L109-L166).

Using AI to "spruce up" issue or PR descriptions is discouraged (though machine translation is okay).

## Proposing API changes & features
Please open an issue before starting any PRs that modify `ort`'s API so that any concerns can be resolved before work begins.

## Adding a new example
`ort`'s examples are structured very differently from other Rust projects (they're really weird!). The `gpt2` example is a good reference for creating new examples.

Let's call our new example `agi-net`. To start off, create a directory in `examples/agi-net` and add a `Cargo.toml` which looks like this:
```toml
[package]
publish = false
name = "example-agi-net"
version = "0.0.0"
edition = "2024"

[dependencies]
ort = { path = "../../" }

ort-candle = { path = "../../backends/candle", optional = true }
ort-tract = { path = "../../backends/tract", optional = true }

[features]
load-dynamic = [ "ort/load-dynamic" ]

cuda = [ "ort/cuda" ]
# ...copy the rest of the features from gpt2/Cargo.toml...
azure = [ "ort/azure" ]

backend-candle = [ "ort/alternative-backend", "dep:ort-candle" ]
backend-tract = [ "ort/alternative-backend", "dep:ort-tract" ]

[[example]]
name = "agi-net"
path = "agi-net.rs"
```

You can of course add dependencies and `ort` features as needed.

Next, `examples/agi-net` needs to be added to the `workspace.exclude` field in the root `Cargo.toml` to separate it from the main workspace. You can then open `examples/agi-net` in your IDE.

The main example `examples/agi-net/agi-net.rs` should look like this:
```rs
use ort::{...};

// Include common code for `ort` examples that allows using the various feature flags to enable different EPs and
// backends.
#[path = "../common/mod.rs"]
mod common;

fn main() -> ort::Result<()> {
	// Initialize tracing to receive debug messages from `ort`
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,ort=debug".into()))
		.with(tracing_subscriber::fmt::layer())
		.init();

	// Register EPs based on feature flags - this isn't crucial for usage and can be removed.
	common::init()?;

    ... // The AGI part is left as an exercise for the reader.

    Ok(())
}
```

Make sure to keep the comments annotating `mod common`, the tracing initialization, and `common::init` so their purpose is clear.

The last thing we need to do in order to make the example runnable is to add a Cargo alias to the `.cargo/config.toml` in the root of the repository:
```toml
[alias]
...
example-agi-net = ["run", "--manifest-path", "examples/agi-net/Cargo.toml", "--example", "agi-net", "--target-dir", "target"]
```

Once you're done writing the example, run it, take a screenshot of your masterpiece, and add it to `examples/README.md`!

## Contributing code changes
After making code changes, please do the following before submitting your PR so it can be accepted quicker.

<ol>
<li>

**Format the code**. This requires a nightly Rust toolchain, which can be installed with `rustup toolchain install nightly`. The exact version of the toolchain required can be found in [`.github/workflows/code-quality.yml`](https://github.com/pykeio/ort/blob/main/.github/workflows/code-quality.yml), but it's ok if you have a newer version.

In the root of the repository, run:
```shell
$ cargo +nightly fmt
```

Any warnings about "`fn_call_width`" can be safely ignored. I don't know how to fix that.

</li>
<li>

**Test the code**. Ideally use stable Rust for this step. When testing code, the `fetch-models` feature flag is required, since many tests depend on externally hosted models.

In the root of the repository, run:
```shell
$ cargo test --features fetch-models
```

Generally, you don't need to enable any other features, unless you changed something related to a non-default feature like `load-dynamic`. Don't enable `--all-features`, as this will error; testing with only the default features (+ `fetch-models`) is fine.

Alternative backends aren't part of the main workspace, so if you change code in one of them, you'll have to `cd` into that backend and run `cargo test` there (`fetch-models` is not required).

</li>
</ol>

If you made changes to the public API, then make sure to update all examples as well, since they aren't in the main workspace and thus aren't checked by `cargo check` or `cargo test`.

## Reporting security concerns
See [`SECURITY.md`](https://github.com/pykeio/ort/blob/main/SECURITY.md) for information on reporting security concerns; do not use public forums to report vulnerabilities.
