// SPDX-License-Identifier: MIT

//! Test-only builders for packed GGUF data: raw GGUF files with tensor types
//! Candle's writer cannot emit, and reproducible i-quant blocks.

use half::f16;

use super::iquant::IQuantType;

/// A minimal GGUF v3 file (no metadata, 32-byte alignment) holding `tensors`
/// as `(name, shape outermost-first, ggml type id, raw bytes)`.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn write_gguf(tensors: &[(&str, &[usize], u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend((tensors.len() as u64).to_le_bytes());
    out.extend(0u64.to_le_bytes());
    let mut offset = 0u64;
    for (name, shape, ty, data) in tensors {
        out.extend((name.len() as u64).to_le_bytes());
        out.extend(name.as_bytes());
        out.extend((shape.len() as u32).to_le_bytes());
        for &d in shape.iter().rev() {
            out.extend((d as u64).to_le_bytes());
        }
        out.extend(ty.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset += (data.len() as u64).div_ceil(32) * 32;
    }
    for (_, _, _, data) in tensors {
        out.resize(out.len().div_ceil(32) * 32, 0);
        out.extend(data);
    }
    out
}

/// `n` pseudo-random blocks of `ty` with sane `f16` scales. The generator
/// is fixed: golden values (e.g. `iquant::tests::low_bit_decoders_match_ggml`)
/// were computed from its exact output.
pub(crate) fn random_blocks(ty: IQuantType, n: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(n * ty.block_bytes());
    for i in 0..n {
        #[allow(clippy::cast_precision_loss)]
        let scale = f16::from_f32(0.002 + 0.0001 * (i % 7) as f32).to_le_bytes();
        let start = out.len();
        out.extend(scale);
        for _ in 2..ty.block_bytes() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            #[allow(clippy::cast_possible_truncation)]
            out.push(state as u8);
        }
        // Other `f16` fields would be random bits, possibly NaN.
        let block = &mut out[start..];
        match ty {
            IQuantType::Q4K | IQuantType::Q5K => {
                block[2..4].copy_from_slice(&f16::from_f32(0.001).to_le_bytes());
            },
            IQuantType::Q6K => {
                let end = block.len();
                block[end - 2..].copy_from_slice(&scale);
            },
            _ => {},
        }
    }
    out
}
