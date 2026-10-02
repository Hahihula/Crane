//! SYCL launcher for the fused Gated Delta Net recurrence kernel.
//!
//! The counterpart of [`super::cuda_backend`] / [`super::rocm_backend`], against
//! `kernels/sycl/gdn.cpp` (built into `libcrane_gdn_sycl.so` by `build.rs` with
//! `icpx`). Collapses the per-timestep candle op graph into one submission onto
//! candle's in-order SYCL queue.
//!
//! Inputs must be contiguous f32 SYCL tensors in the layouts documented on
//! [`gdn_recurrence_sycl`]; `q` is expected pre-scaled by `1/sqrt(K)` (the
//! caller does this, matching the CPU reference). No explicit `synchronize`:
//! candle's SYCL queue is in-order, so the wrapped result tensors are correctly
//! ordered against every later op on the same queue.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_gdn_recurrence_sycl(
        queue: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state_in: *const f32,
        state_out: *mut f32,
        y: *mut f32,
        bh: i32,
        s: i32,
        kdim: i32,
        vdim: i32,
        v_tile: i32,
    ) -> i32;
}

/// Run the gated delta rule recurrence on SYCL.
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
///
/// # Errors
///
/// Returns an error if `head_k_dim > 256` (the kernel's staging limit), if any
/// operand is not an f32 SYCL tensor, or if the submission fails.
pub fn gdn_recurrence_sycl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (bh, s, kdim) = q.dims3()?;
    let vdim = v.dim(2)?;
    if kdim > 256 {
        candle_core::bail!("gdn sycl kernel supports head_k_dim <= 256, got {kdim}");
    }

    let dev = q.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();

    // The storage guards must outlive the submission — the raw pointers borrow
    // from them — so they are all bound here rather than inside a helper.
    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (g_s, g_l) = g.storage_and_layout();
    let (beta_s, beta_l) = beta.storage_and_layout();
    let (state_s, state_l) = state.storage_and_layout();

    let ptr = |s: &Storage, offset: usize, name: &str| -> Result<*const f32> {
        match s {
            Storage::Sycl(st) => Ok(unsafe { (st.buf().as_ptr() as *const f32).add(offset) }),
            _ => candle_core::bail!("gdn: {name} must be a sycl tensor"),
        }
    };
    let q_ptr = ptr(&q_s, q_l.start_offset(), "q")?;
    let k_ptr = ptr(&k_s, k_l.start_offset(), "k")?;
    let v_ptr = ptr(&v_s, v_l.start_offset(), "v")?;
    let g_ptr = ptr(&g_s, g_l.start_offset(), "g")?;
    let beta_ptr = ptr(&beta_s, beta_l.start_offset(), "beta")?;
    let state_ptr = ptr(&state_s, state_l.start_offset(), "state")?;

    let y_elems = bh * s * vdim;
    let state_elems = bh * kdim * vdim;
    let y_buf = dev.alloc_bytes(y_elems * std::mem::size_of::<f32>())?;
    let state_out_buf = dev.alloc_bytes(state_elems * std::mem::size_of::<f32>())?;

    let status = unsafe {
        crane_gdn_recurrence_sycl(
            queue,
            q_ptr,
            k_ptr,
            v_ptr,
            g_ptr,
            beta_ptr,
            state_ptr,
            state_out_buf.as_mut_ptr() as *mut f32,
            y_buf.as_mut_ptr() as *mut f32,
            bh as i32,
            s as i32,
            kdim as i32,
            vdim as i32,
            // Per-column kernel only (head sizes the sub-group kernel does not
            // cover): one work-group per (batch*head).
            vdim as i32,
        )
    };
    if status != 0 {
        candle_core::bail!("crane_gdn_recurrence_sycl failed (status {status})");
    }

    let y = Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, y_buf, DType::F32, y_elems)),
        (bh, s, vdim),
        BackpropOp::none(),
        false,
    );
    let state_out = Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(
            &dev,
            state_out_buf,
            DType::F32,
            state_elems,
        )),
        (bh, kdim, vdim),
        BackpropOp::none(),
        false,
    );
    Ok((y, state_out))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    /// The recurrence documented in `kernels/sycl/gdn.cpp`, step by step in
    /// f64: `S *= exp(g)`, `delta = (v - S^T k) * beta`, `S += k delta^T`,
    /// `y = S^T q`. Returns `(y [BH,S,V], state [BH,K,V])`, flattened.
    #[allow(clippy::many_single_char_names)]
    fn reference(
        [q, k, v, g, beta, state]: [&[f32]; 6],
        [bh, s, kd, vd]: [usize; 4],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut y = vec![0f32; bh * s * vd];
        let mut out = vec![0f32; bh * kd * vd];
        for h in 0..bh {
            let mut st: Vec<f64> = state[h * kd * vd..(h + 1) * kd * vd]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            for t in 0..s {
                let decay = f64::from(g[h * s + t]).exp();
                let kt = &k[(h * s + t) * kd..][..kd];
                let qt = &q[(h * s + t) * kd..][..kd];
                for col in 0..vd {
                    let mut kv = 0f64;
                    for r in 0..kd {
                        st[r * vd + col] *= decay;
                        kv += st[r * vd + col] * f64::from(kt[r]);
                    }
                    let delta =
                        (f64::from(v[(h * s + t) * vd + col]) - kv) * f64::from(beta[h * s + t]);
                    let mut acc = 0f64;
                    for r in 0..kd {
                        st[r * vd + col] += f64::from(kt[r]) * delta;
                        acc += st[r * vd + col] * f64::from(qt[r]);
                    }
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        y[(h * s + t) * vd + col] = acc as f32;
                    }
                }
            }
            for (o, &x) in out[h * kd * vd..].iter_mut().zip(&st) {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *o = x as f32;
                }
            }
        }
        (y, out)
    }

    /// Model-like inputs: unit-norm keys and queries (queries also scaled by
    /// `1/sqrt(K)`, as the caller does), log-decays in `(-1, 0)` and write
    /// strengths in `(0, 1)`.
    fn inputs(bh: usize, s: usize, kd: usize, vd: usize) -> candle_core::Result<[Tensor; 6]> {
        let dev = Device::Cpu;
        let unit = |rows: usize| -> candle_core::Result<Tensor> {
            let x = Tensor::randn(0f32, 1.0, (bh, rows, kd), &dev)?;
            x.broadcast_div(&x.sqr()?.sum_keepdim(2)?.sqrt()?)
        };
        #[allow(clippy::cast_precision_loss)]
        let q = (unit(s)? / (kd as f64).sqrt())?;
        Ok([
            q,
            unit(s)?,
            Tensor::randn(0f32, 1.0, (bh, s, vd), &dev)?,
            (Tensor::rand(0f32, 1.0, (bh, s), &dev)? * -1.0)?,
            Tensor::rand(0f32, 1.0, (bh, s), &dev)?,
            Tensor::randn(0f32, 0.1, (bh, kd, vd), &dev)?,
        ])
    }

    #[test]
    fn sycl_recurrence_matches_reference() -> candle_core::Result<()> {
        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        let sycl = Device::new_sycl(0)?;
        // Qwen 3.5's 128x128 heads, other K/V the kernel accepts, decode
        // (S = 1) and an odd prefill length.
        for (bh, s, kd, vd) in [
            (3, 1, 128, 128),
            (3, 37, 128, 128),
            (2, 300, 128, 128),
            (2, 37, 64, 128),
            (2, 37, 256, 64),
            (2, 37, 96, 40),
        ] {
            let ts = inputs(bh, s, kd, vd)?;
            let host: Vec<Vec<f32>> = ts
                .iter()
                .map(|t| t.flatten_all()?.to_vec1::<f32>())
                .collect::<candle_core::Result<_>>()?;
            let (want_y, want_state) =
                reference(std::array::from_fn(|i| host[i].as_slice()), [bh, s, kd, vd]);
            let on: Vec<Tensor> = ts
                .iter()
                .map(|t| t.to_device(&sycl))
                .collect::<candle_core::Result<_>>()?;
            let (y, state) =
                super::gdn_recurrence_sycl(&on[0], &on[1], &on[2], &on[3], &on[4], &on[5])?;
            for (name, got, want) in [("y", y, want_y), ("state", state, want_state)] {
                let got = got.flatten_all()?.to_vec1::<f32>()?;
                let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
                let diff = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    diff / scale < 1e-4,
                    "{name} bh={bh} s={s} k={kd} v={vd}: rel diff {}",
                    diff / scale
                );
            }
        }
        Ok(())
    }
}
