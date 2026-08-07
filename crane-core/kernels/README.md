# Native kernel backends

Native kernels are grouped by execution backend:

- `cuda/` contains CUDA sources compiled to PTX when the crate's `cuda`
  feature is enabled.
- `cpu/` is reserved for future CPU-native kernel implementations.

The CPU path currently uses the portable Rust/Candle implementations in
`src/ops`. Keeping native sources separated by backend ensures a future CPU
build integration cannot accidentally be passed to the CUDA PTX compiler.
