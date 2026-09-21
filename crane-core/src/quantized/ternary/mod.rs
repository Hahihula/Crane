// SPDX-License-Identifier: MIT

//! Prism ternary GGUF support.

mod codec;
mod linear;

pub use codec::{TernaryEncoding, decode_block};
pub use linear::{GdnPermutation, HadamardMode, TernaryLinear, TernaryWeight};
