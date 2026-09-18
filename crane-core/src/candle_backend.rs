// Puts the active candle backend into the extern prelude as `candle_core` /
// `candle_nn` / `candle_transformers`, so existing call sites don't change
// per backend. Must be `include!`d at the crate root (not a `mod`) for the
// prelude effect to apply. `sycl` wins if both `sycl` and `rocm` are set.

#[cfg(feature = "sycl")]
pub extern crate candle_core_sycl as candle_core;
#[cfg(feature = "sycl")]
pub extern crate candle_nn_sycl as candle_nn;
#[cfg(feature = "sycl")]
pub extern crate candle_transformers_sycl as candle_transformers;

#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_core_rocm as candle_core;
#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_nn_rocm as candle_nn;
#[cfg(all(feature = "rocm", not(feature = "sycl")))]
pub extern crate candle_transformers_rocm as candle_transformers;

#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_core_upstream as candle_core;
#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_nn_upstream as candle_nn;
#[cfg(not(any(feature = "sycl", feature = "rocm")))]
pub extern crate candle_transformers_upstream as candle_transformers;
