// SPDX-License-Identifier: MIT
//! Crane Added 20260731: ONNX `STFT` as a native eval op.

use candle_core::{IndexOp, Result, Tensor, bail};
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `STFT`: computes a real forward short-time Fourier transform.
///
/// `signal` (input 0) is `[batch, signal_length]` or
/// `[batch, signal_length, 1]`; `frame_step` (input 1) is a scalar frame
/// hop size; `window` (input 2) is a 1-D window of length `frame_length`,
/// required here (a windowless STFT isn't supported); `frame_length`
/// (input 3, optional) must match `window`'s length when present, and
/// defaults to it otherwise.
///
/// Frames the signal every `frame_step` samples, applies `window`, and runs
/// an FFT of size `frame_length` per frame via `rustfft`. When the
/// `onesided` attribute (default `1`) is nonzero, only the first
/// `frame_length / 2 + 1` bins are kept. Output shape is
/// `[batch, num_frames, n_bins, 2]` (real, imaginary).
pub(crate) fn stft(
    node: &NodeProto,
    signal: &Tensor,
    frame_step: &Tensor,
    window: Option<&Tensor>,
    frame_length: Option<&Tensor>,
) -> Result<Tensor> {
    let frame_step = scalar_i64(frame_step)?;
    if frame_step <= 0 {
        bail!(
            "STFT node '{}': frame_step must be > 0, got {frame_step}",
            node.name
        );
    }
    // Validated positive above, so this cast never wraps or loses sign.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let frame_step = frame_step as usize;

    let Some(window) = window else {
        bail!(
            "STFT node '{}' has no window input; a windowless STFT isn't supported",
            node.name
        );
    };
    let window = window.to_vec1::<f32>()?;

    let frame_length = match frame_length {
        Some(t) => {
            let v = scalar_i64(t)?;
            if v < 0 {
                bail!(
                    "STFT node '{}': frame_length must be non-negative, got {v}",
                    node.name
                );
            }
            // Validated non-negative above.
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let v = v as usize;
            v
        },
        None => window.len(),
    };
    if frame_length != window.len() {
        bail!(
            "STFT node '{}': frame_length {frame_length} does not match window length {}",
            node.name,
            window.len()
        );
    }
    if frame_length == 0 {
        bail!("STFT node '{}': frame_length must be > 0", node.name);
    }

    let onesided = int_attribute(node, "onesided", 1)? != 0;

    let signal = match signal.dims() {
        [_, _] => signal.clone(),
        [b, l, 1] => signal.reshape((*b, *l))?,
        dims => bail!(
            "STFT node '{}': input has unsupported shape {dims:?}",
            node.name
        ),
    };
    let (batch, signal_length) = signal.dims2()?;
    if signal_length < frame_length {
        bail!(
            "STFT node '{}': signal length {signal_length} is shorter than frame_length {frame_length}",
            node.name
        );
    }
    let num_frames = 1 + (signal_length - frame_length) / frame_step;
    let n_bins = if onesided {
        frame_length / 2 + 1
    } else {
        frame_length
    };

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(frame_length);

    let signal_data = signal.to_vec2::<f32>()?;
    let mut output = vec![0f32; batch * num_frames * n_bins * 2];
    let mut buffer = vec![FftComplex::new(0.0, 0.0); frame_length];
    for (b, row) in signal_data.iter().enumerate() {
        for f in 0..num_frames {
            let start = f * frame_step;
            for (i, sample) in buffer.iter_mut().enumerate() {
                *sample = FftComplex::new(row[start + i] * window[i], 0.0);
            }
            fft.process(&mut buffer);
            for (k, bin) in buffer.iter().take(n_bins).enumerate() {
                let idx = ((b * num_frames + f) * n_bins + k) * 2;
                output[idx] = bin.re;
                output[idx + 1] = bin.im;
            }
        }
    }
    Tensor::from_vec(output, (batch, num_frames, n_bins, 2), signal.device())
}

fn scalar_i64(t: &Tensor) -> Result<i64> {
    if t.rank() > 0 && t.elem_count() == 1 {
        t.flatten_all()?.i(0)?.to_vec0::<i64>()
    } else {
        t.to_vec0::<i64>()
    }
}

fn int_attribute(node: &NodeProto, name: &str, default: i64) -> Result<i64> {
    let Some(attribute) = node
        .attribute
        .iter()
        .find(|attribute| attribute.name == name)
    else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Int {
        bail!(
            "STFT node '{}' has a non-INT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(attribute.i)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::stft;

    fn node() -> NodeProto {
        NodeProto {
            name: "STFT.0".to_string(),
            ..Default::default()
        }
    }

    fn node_with_onesided(value: i64) -> NodeProto {
        NodeProto {
            name: "STFT.0".to_string(),
            attribute: vec![AttributeProto {
                name: "onesided".to_string(),
                r#type: AttributeType::Int as i32,
                i: value,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    // A constant-1.0 signal windowed by a rectangular window of length 4:
    // each frame's FFT has bin 0 == (frame_length, 0) and every other bin
    // == (0, 0), an easy hand-checkable case.
    #[test]
    fn stft_basic_windowed_constant_signal() -> Result<()> {
        let signal = Tensor::new(&[[1f32; 8]], &Device::Cpu)?;
        let frame_step = Tensor::new(2i64, &Device::Cpu)?;
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu)?;

        let y = stft(&node(), &signal, &frame_step, Some(&window), None)?;

        // num_frames = 1 + (8 - 4) / 2 = 3; n_bins = 4/2 + 1 = 3 (onesided).
        assert_eq!(y.dims(), &[1, 3, 3, 2]);
        let data = y.flatten_all()?.to_vec1::<f32>()?;
        for frame in 0..3 {
            let base = frame * 3 * 2;
            assert!((data[base] - 4.0).abs() < 1e-4, "bin 0 real: {data:?}");
            assert!(data[base + 1].abs() < 1e-4, "bin 0 imag: {data:?}");
            for bin in 1..3 {
                let idx = base + bin * 2;
                assert!(
                    data[idx].abs() < 1e-4,
                    "bin {bin} real should be ~0: {data:?}"
                );
                assert!(
                    data[idx + 1].abs() < 1e-4,
                    "bin {bin} imag should be ~0: {data:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn stft_onesided_false_keeps_all_bins() -> Result<()> {
        let signal = Tensor::new(&[[1f32; 8]], &Device::Cpu)?;
        let frame_step = Tensor::new(2i64, &Device::Cpu)?;
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu)?;

        let y = stft(
            &node_with_onesided(0),
            &signal,
            &frame_step,
            Some(&window),
            None,
        )?;

        assert_eq!(y.dims(), &[1, 3, 4, 2]);
        Ok(())
    }

    #[test]
    fn stft_signal_shorter_than_frame_bails() {
        let signal = Tensor::new(&[[1f32, 2., 3.]], &Device::Cpu).unwrap();
        let frame_step = Tensor::new(1i64, &Device::Cpu).unwrap();
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu).unwrap();

        let err = stft(&node(), &signal, &frame_step, Some(&window), None).unwrap_err();

        assert!(err.to_string().contains("shorter than frame_length"));
    }

    #[test]
    fn stft_frame_length_mismatch_bails() {
        let signal = Tensor::new(&[[1f32; 8]], &Device::Cpu).unwrap();
        let frame_step = Tensor::new(1i64, &Device::Cpu).unwrap();
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu).unwrap();
        let frame_length = Tensor::new(8i64, &Device::Cpu).unwrap();

        let err = stft(
            &node(),
            &signal,
            &frame_step,
            Some(&window),
            Some(&frame_length),
        )
        .unwrap_err();

        assert!(err.to_string().contains("does not match window length"));
    }

    #[test]
    fn stft_3d_input_squeezes_trailing_one() -> Result<()> {
        let signal = Tensor::new(
            &[[[1f32], [1.], [1.], [1.], [1.], [1.], [1.], [1.]]],
            &Device::Cpu,
        )?;
        let frame_step = Tensor::new(2i64, &Device::Cpu)?;
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu)?;

        let y = stft(&node(), &signal, &frame_step, Some(&window), None)?;

        assert_eq!(y.dims(), &[1, 3, 3, 2]);
        Ok(())
    }

    #[test]
    fn stft_no_window_bails() {
        let signal = Tensor::new(&[[1f32; 8]], &Device::Cpu).unwrap();
        let frame_step = Tensor::new(2i64, &Device::Cpu).unwrap();

        let err = stft(&node(), &signal, &frame_step, None, None).unwrap_err();

        assert!(err.to_string().contains("no window input"));
    }

    #[test]
    fn stft_zero_frame_step_bails() {
        let signal = Tensor::new(&[[1f32; 8]], &Device::Cpu).unwrap();
        let frame_step = Tensor::new(0i64, &Device::Cpu).unwrap();
        let window = Tensor::new(&[1f32, 1., 1., 1.], &Device::Cpu).unwrap();

        let err = stft(&node(), &signal, &frame_step, Some(&window), None).unwrap_err();

        assert!(err.to_string().contains("frame_step must be > 0"));
    }
}
