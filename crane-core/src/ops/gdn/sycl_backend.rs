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

    fn crane_gdn_fused_sycl(
        queue: *mut c_void,
        dtype: i32,
        qkv: *const c_void,
        a: *const c_void,
        b: *const c_void,
        neg_a: *const f32,
        dt_bias: *const f32,
        state_in: *const f32,
        state_out: *mut f32,
        y: *mut c_void,
        batch: i32,
        s: i32,
        hk: i32,
        hv: i32,
        kdim: i32,
        vdim: i32,
        conv_dim: i32,
        key_dim: i32,
        chunked: i32,
    ) -> i32;
}

/// Pointer to element `offset` (in elements of `dtype`) of a SYCL tensor's
/// storage.
fn sycl_ptr(s: &Storage, offset: usize, dtype: DType, name: &str) -> Result<*const c_void> {
    match s {
        Storage::Sycl(st) => Ok(unsafe {
            st.buf()
                .as_ptr()
                .cast::<u8>()
                .add(offset * dtype.size_in_bytes())
                .cast::<c_void>()
        }),
        _ => candle_core::bail!("gdn: {name} must be a sycl tensor"),
    }
}

/// One GDN layer's token mixing after the conv, fused into a single launch:
/// Q/K/V split, key-head expansion, Q/K L2 norms, query scale, the `beta` /
/// `g` gates and the recurrence (see `crane_gdn_fused_sycl` in
/// `kernels/sycl/gdn.cpp`). On the unfused path these are ~20 candle ops per
/// layer per token, and SYCL decode is bound by submitting them.
///
/// `qkv` is the conv output `[B, S, conv_dim]`, `a` / `b` the raw gate
/// projections `[B, S, Hv]`, all in the model dtype (F16 or F32);
/// `neg_exp_a_log` / `dt_bias` hold `Hv` f32s and `state` is `[B, Hv, K, V]`
/// f32. Returns `(y [B, S, Hv, V]` in the model dtype`, state_out)`, or `None`
/// when the shapes or dtype are outside what the kernel covers (the caller
/// then takes the unfused path).
///
/// # Errors
///
/// Returns an error if an operand is not a SYCL tensor or the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn gdn_fused_sycl(
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    neg_exp_a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    dims: &super::GdnDims,
) -> Result<Option<(Tensor, Tensor)>> {
    let dtype = qkv.dtype();
    let dtype_tag = match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        _ => return Ok(None),
    };
    let (kd, vd) = (dims.head_k_dim, dims.head_v_dim);
    if !matches!(kd, 64 | 128 | 256) || vd % 4 != 0 {
        return Ok(None);
    }
    let (batch, seq_len, conv_dim) = qkv.dims3()?;
    let hv = dims.num_v_heads;

    let qkv = qkv.contiguous()?;
    let a = a.to_dtype(dtype)?.contiguous()?;
    let b = b.to_dtype(dtype)?.contiguous()?;
    let neg_a = neg_exp_a_log.flatten_all()?.contiguous()?;
    let dt_bias = dt_bias.flatten_all()?.contiguous()?;
    let state = state.to_dtype(DType::F32)?.contiguous()?;

    let dev = qkv.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let (qkv_s, qkv_l) = qkv.storage_and_layout();
    let (a_s, a_l) = a.storage_and_layout();
    let (b_s, b_l) = b.storage_and_layout();
    let (na_s, na_l) = neg_a.storage_and_layout();
    let (dt_s, dt_l) = dt_bias.storage_and_layout();
    let (st_s, st_l) = state.storage_and_layout();

    let y_elems = batch * seq_len * hv * vd;
    let state_elems = batch * hv * kd * vd;
    let y_buf = dev.alloc_bytes(y_elems * dtype.size_in_bytes())?;
    let state_out_buf = dev.alloc_bytes(state_elems * std::mem::size_of::<f32>())?;
    let to_i32 = |n: usize| {
        i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("gdn: {n} exceeds i32")))
    };

    let status = unsafe {
        crane_gdn_fused_sycl(
            queue,
            dtype_tag,
            sycl_ptr(&qkv_s, qkv_l.start_offset(), dtype, "qkv")?,
            sycl_ptr(&a_s, a_l.start_offset(), dtype, "a")?,
            sycl_ptr(&b_s, b_l.start_offset(), dtype, "b")?,
            sycl_ptr(&na_s, na_l.start_offset(), DType::F32, "neg_exp_a_log")?.cast(),
            sycl_ptr(&dt_s, dt_l.start_offset(), DType::F32, "dt_bias")?.cast(),
            sycl_ptr(&st_s, st_l.start_offset(), DType::F32, "state")?.cast(),
            state_out_buf.as_mut_ptr().cast(),
            y_buf.as_mut_ptr(),
            to_i32(batch)?,
            to_i32(seq_len)?,
            to_i32(dims.num_k_heads)?,
            to_i32(hv)?,
            to_i32(kd)?,
            to_i32(vd)?,
            to_i32(conv_dim)?,
            to_i32(dims.key_dim)?,
            i32::from(dims.v_head_order == super::VHeadOrder::Chunked),
        )
    };
    match status {
        0 => {},
        2 => return Ok(None),
        _ => candle_core::bail!("crane_gdn_fused_sycl failed (status {status})"),
    }

    let y = Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, y_buf, dtype, y_elems)),
        (batch, seq_len, hv, vd),
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
        (batch, hv, kd, vd),
        BackpropOp::none(),
        false,
    );
    Ok(Some((y, state_out)))
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

    /// [`super::gdn_fused_sycl`] against the unfused steps on the host: split
    /// the conv output, map key heads to value heads in either order, L2-norm
    /// and scale, compute `beta` / `g`, then [`reference`].
    #[test]
    fn sycl_fused_matches_reference() -> candle_core::Result<()> {
        use super::super::{GdnDims, VHeadOrder};
        use candle_core::DType;

        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        let sycl = Device::new_sycl(0)?;
        let cpu = Device::Cpu;
        for (batch, s, hk, hv, kd, vd, order, dtype) in [
            (1, 1, 2, 4, 128, 128, VHeadOrder::Chunked, DType::F32),
            (1, 37, 2, 4, 128, 128, VHeadOrder::Chunked, DType::F32),
            (2, 37, 2, 4, 128, 64, VHeadOrder::Interleaved, DType::F32),
            (1, 19, 3, 3, 64, 128, VHeadOrder::Chunked, DType::F32),
            (1, 37, 2, 4, 128, 128, VHeadOrder::Chunked, DType::F16),
        ] {
            let dims = GdnDims {
                hidden_size: 0,
                num_k_heads: hk,
                num_v_heads: hv,
                head_k_dim: kd,
                head_v_dim: vd,
                conv_kernel_size: 4,
                key_dim: hk * kd,
                value_dim: hv * vd,
                conv_dim: 2 * hk * kd + hv * vd,
                v_per_group: hv / hk,
                v_head_order: order,
            };
            // Inputs rounded to the test dtype first, so the host reference
            // sees exactly what the kernel reads.
            let round = |t: Tensor| t.to_dtype(dtype)?.to_dtype(DType::F32);
            let qkv = round(Tensor::randn(0f32, 1.0, (batch, s, dims.conv_dim), &cpu)?)?;
            let a = round(Tensor::randn(0f32, 1.0, (batch, s, hv), &cpu)?)?;
            let b = round(Tensor::randn(0f32, 1.0, (batch, s, hv), &cpu)?)?;
            let neg_a = (Tensor::rand(0f32, 1.0, hv, &cpu)? * -1.0)?;
            let dt_bias = Tensor::randn(0f32, 1.0, hv, &cpu)?;
            let state = Tensor::randn(0f32, 0.1, (batch, hv, kd, vd), &cpu)?;

            let host = |t: &Tensor| t.flatten_all()?.to_vec1::<f32>();
            let (xq, xa, xb, xna, xdt) = (
                host(&qkv)?,
                host(&a)?,
                host(&b)?,
                host(&neg_a)?,
                host(&dt_bias)?,
            );
            let bh = batch * hv;
            let (mut q3, mut k3) = (vec![0f32; bh * s * kd], vec![0f32; bh * s * kd]);
            let mut v3 = vec![0f32; bh * s * vd];
            let (mut g2, mut beta2) = (vec![0f32; bh * s], vec![0f32; bh * s]);
            for bi in 0..batch {
                for h in 0..hv {
                    let kh = match order {
                        VHeadOrder::Chunked => h % hk,
                        VHeadOrder::Interleaved => h / (hv / hk),
                    };
                    let row = bi * hv + h;
                    for t in 0..s {
                        let x = &xq[(bi * s + t) * dims.conv_dim..][..dims.conv_dim];
                        let unit = |v: &[f32], scale: f64| -> Vec<f32> {
                            let ss: f64 = v.iter().map(|&e| f64::from(e).powi(2)).sum();
                            let inv = scale / (ss + 1e-6).sqrt();
                            #[allow(clippy::cast_possible_truncation)]
                            v.iter().map(|&e| (f64::from(e) * inv) as f32).collect()
                        };
                        #[allow(clippy::cast_precision_loss)]
                        let qn = unit(&x[kh * kd..][..kd], 1.0 / (kd as f64).sqrt());
                        let kn = unit(&x[dims.key_dim + kh * kd..][..kd], 1.0);
                        q3[(row * s + t) * kd..][..kd].copy_from_slice(&qn);
                        k3[(row * s + t) * kd..][..kd].copy_from_slice(&kn);
                        v3[(row * s + t) * vd..][..vd]
                            .copy_from_slice(&x[2 * dims.key_dim + h * vd..][..vd]);
                        let i = (bi * s + t) * hv + h;
                        let sp = (f64::from(xa[i]) + f64::from(xdt[h])).exp().ln_1p();
                        #[allow(clippy::cast_possible_truncation)]
                        {
                            g2[row * s + t] = (f64::from(xna[h]) * sp) as f32;
                            beta2[row * s + t] = (1.0 / (1.0 + (-f64::from(xb[i])).exp())) as f32;
                        }
                    }
                }
            }
            let (want_y, want_state) = reference(
                [&q3, &k3, &v3, &g2, &beta2, &host(&state)?],
                [bh, s, kd, vd],
            );

            let on = |t: &Tensor| t.to_dtype(dtype)?.to_device(&sycl);
            let (y, st) = super::gdn_fused_sycl(
                &on(&qkv)?,
                &on(&a)?,
                &on(&b)?,
                &neg_a.to_device(&sycl)?,
                &dt_bias.to_device(&sycl)?,
                &state.to_device(&sycl)?,
                &dims,
            )?
            .expect("shapes the fused kernel covers");
            assert_eq!(y.dims(), &[batch, s, hv, vd]);
            assert_eq!(y.dtype(), dtype);
            // The reference's y is [B*Hv, S, V]; the kernel's [B, S, Hv, V].
            let y = y
                .to_dtype(DType::F32)?
                .transpose(1, 2)?
                .contiguous()?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tol = if dtype == DType::F16 { 5e-3 } else { 1e-4 };
            for (name, got, want) in [("y", y, want_y), ("state", host(&st)?, want_state)] {
                let scale = want.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
                let diff = got
                    .iter()
                    .zip(&want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    diff / scale < tol,
                    "{name} {dtype:?} b={batch} s={s} hk={hk} hv={hv} k={kd} v={vd} {order:?}: rel diff {}",
                    diff / scale
                );
            }
        }
        Ok(())
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
