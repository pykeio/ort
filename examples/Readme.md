# Note

In case you encounter the following error:

```
The given version [16] is not supported, only version 1 to 10 is supported in this build.
thread 'main' panicked at src\lib.rs:158:5:
assertion `left != right` failed
  left: 0x0
 right: 0x0
stack backtrace:
   0: std::panicking::begin_panic_handler
             at /rustc/cc66ad468955717ab92600c770da8c1601a4ff33/library\std\src\panicking.rs:595
   1: core::panicking::panic_fmt
             at /rustc/cc66ad468955717ab92600c770da8c1601a4ff33/library\core\src\panicking.rs:67
   2: core::panicking::assert_failed_inner
             at /rustc/cc66ad468955717ab92600c770da8c1601a4ff33/library\core\src\panicking.rs:269
   3: core::panicking::assert_failed<ptr_mut$<ort_sys::OrtApi>,ptr_mut$<ort_sys::OrtApi> >
             at /rustc/cc66ad468955717ab92600c770da8c1601a4ff33\library\core\src\panicking.rs:229
   4: ort::ort
             at .\src\lib.rs:158
   5: ort::environment::EnvironmentBuilder::build
             at .\src\environment.rs:307
   6: gpt2::main
             at .\examples\gpt2\examples\gpt2.rs:21
   7: core::ops::function::FnOnce::call_once<enum2$<core::result::Result<tuple$<>,enum2$<ort::error::Error> > > (*)(),tuple$<> >
             at /rustc/cc66ad468955717ab92600c770da8c1601a4ff33\library\core\src\ops\function.rs:250
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

Please read the documentation at: https://github.com/pykeio/ort#shared-library-hell