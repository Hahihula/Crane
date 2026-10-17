//! The fused multi-token CUDA path of `GatedDeltaNet` (`gdn_fused.cu`) against
//! the op-by-op chain it replaces: same layer output, same conv / recurrent /
//! checkpoint state, for several lengths, head groupings and checkpoint splits.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use crane_core::candle_core::{DType, Device, Result, Tensor};
use crane_core::candle_nn::VarBuilder;
use crane_core::ops::gdn::{
    GateActivation, GatedDeltaNet, GdnConfig, GdnDims, GdnInputProjectionKind, GdnLayerCache,
};

struct Cfg {
    hidden: usize,
    k_heads: usize,
    v_heads: usize,
}

impl GdnConfig for Cfg {
    fn hidden_size(&self) -> usize {
        self.hidden
    }
    fn rms_norm_eps(&self) -> f64 {
        1e-6
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        4
    }
    fn linear_key_head_dim(&self) -> usize {
        128
    }
    fn linear_value_head_dim(&self) -> usize {
        128
    }
    fn linear_num_key_heads(&self) -> usize {
        self.k_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.v_heads
    }
    fn output_gate_activation(&self) -> Result<GateActivation> {
        Ok(GateActivation::Silu)
    }
}

fn rand(dev: &Device, shape: &[usize], std: f64) -> Tensor {
    Tensor::randn(0f32, std as f32, shape, dev)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
}

fn layer(dev: &Device, cfg: &Cfg) -> (GatedDeltaNet, GdnDims) {
    let dims = GdnDims::new(cfg);
    let mut w: HashMap<String, Tensor> = HashMap::new();
    let mut put = |k: &str, t: Tensor| {
        w.insert(format!("linear_attn.{k}"), t);
    };
    put(
        "in_proj_qkv.weight",
        rand(dev, &[dims.conv_dim, cfg.hidden], 0.04),
    );
    put(
        "in_proj_z.weight",
        rand(dev, &[dims.value_dim, cfg.hidden], 0.04),
    );
    put(
        "in_proj_b.weight",
        rand(dev, &[dims.num_v_heads, cfg.hidden], 0.04),
    );
    put(
        "in_proj_a.weight",
        rand(dev, &[dims.num_v_heads, cfg.hidden], 0.04),
    );
    put("conv1d.weight", rand(dev, &[dims.conv_dim, 1, 4], 0.5));
    put(
        "dt_bias.weight",
        Tensor::zeros(1, DType::BF16, dev).unwrap(),
    );
    put("dt_bias", rand(dev, &[dims.num_v_heads], 1.0));
    put(
        "A_log",
        (rand(dev, &[dims.num_v_heads], 0.5).abs().unwrap() + 0.5)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
    );
    put(
        "norm.weight",
        (rand(dev, &[128], 0.1) + 1.0)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap(),
    );
    put(
        "out_proj.weight",
        rand(dev, &[cfg.hidden, dims.value_dim], 0.04),
    );
    let vb = VarBuilder::from_tensors(w, DType::BF16, dev);
    let gdn = GatedDeltaNet::load(&vb, cfg, GdnInputProjectionKind::Split, None).unwrap();
    (gdn, dims)
}

fn run(
    gdn: &GatedDeltaNet,
    dims: &GdnDims,
    x: &Tensor,
    cache: &GdnLayerCache,
    split: Option<usize>,
    fused: bool,
) -> (Tensor, GdnLayerCache) {
    run_with(gdn, dims, x, cache, split, fused, true)
}

fn run_with(
    gdn: &GatedDeltaNet,
    dims: &GdnDims,
    x: &Tensor,
    cache: &GdnLayerCache,
    split: Option<usize>,
    fused: bool,
    serial: bool,
) -> (Tensor, GdnLayerCache) {
    // SAFETY: single-threaded test body; nothing else reads the environment.
    unsafe { std::env::set_var("CRANE_GDN_FUSED", if fused { "1" } else { "0" }) };
    let mut c = cache.clone();
    c.checkpoint_at = split;
    c.prefer_serial = serial;
    let y = gdn.forward(x, dims, &mut c).unwrap();
    (y, c)
}

fn diff(a: &Tensor, b: &Tensor) -> (usize, usize, f32) {
    let a = a
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let b = b
        .flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let mismatched = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    (mismatched, a.len(), max)
}

#[test]
#[ignore = "needs a CUDA GPU"]
fn fused_matches_op_chain() {
    let dev = Device::new_cuda(0).unwrap();
    for cfg in [
        Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads: 4,
        },
        Cfg {
            hidden: 512,
            k_heads: 4,
            v_heads: 8,
        },
    ] {
        let (gdn, dims) = layer(&dev, &cfg);
        // A realistic non-zero state: run a prefix through the op chain first.
        let mut base = GdnLayerCache::new(&cfg, DType::BF16, &dev).unwrap();
        let prefix = rand(&dev, &[1, 11, cfg.hidden], 1.0);
        let (_, warmed) = run(&gdn, &dims, &prefix, &base, None, false);
        base = warmed;

        for (seq, split) in [
            (2usize, None),
            (7, None),
            (20, None),
            (20, Some(1)),
            (20, Some(5)),
            (16, Some(15)),
            (3, Some(2)),
        ] {
            let x = rand(&dev, &[1, seq, cfg.hidden], 1.0);
            let (ya, ca) = run(&gdn, &dims, &x, &base, split, false);
            let (yb, cb) = run(&gdn, &dims, &x, &base, split, true);
            let (yd, yn, ymax) = diff(&ya, &yb);
            let (cd, cn, cmax) = diff(&ca.conv_state, &cb.conv_state);
            let (rd, rn, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
            println!(
                "v_heads {} S={seq:<2} split={split:?}: out {yd}/{yn} (max {ymax:.3e}) | conv {cd}/{cn} (max {cmax:.1e}) | state {rd}/{rn} (max {rmax:.1e})",
                cfg.v_heads
            );
            if let (Some((cc_a, rc_a)), Some((cc_b, rc_b))) = (&ca.checkpoint, &cb.checkpoint) {
                let (d1, n1, m1) = diff(cc_a, cc_b);
                let (d2, n2, m2) = diff(rc_a, rc_b);
                println!(
                    "      checkpoint: conv {d1}/{n1} (max {m1:.1e}) | state {d2}/{n2} (max {m2:.1e})"
                );
                assert_eq!(d1, 0, "checkpoint conv state differs");
                assert!(m2 < 1e-3, "checkpoint recurrent state differs: {m2}");
            } else {
                assert_eq!(
                    ca.checkpoint.is_some(),
                    cb.checkpoint.is_some(),
                    "checkpoint presence differs"
                );
            }
            assert_eq!(cd, 0, "conv state differs");
            assert!(rmax < 1e-3, "recurrent state differs: {rmax}");
            assert!(ymax < 0.05, "layer output differs: {ymax}");
        }
    }
}

/// Per-pass cost of one layer, fused vs op chain (prints; also proves the two
/// runs above really took different paths).
#[test]
#[ignore = "needs a CUDA GPU"]
fn fused_is_faster_per_pass() {
    let dev = Device::new_cuda(0).unwrap();
    let cfg = Cfg {
        hidden: 1024,
        k_heads: 16,
        v_heads: 16,
    };
    let (gdn, dims) = layer(&dev, &cfg);
    let base = GdnLayerCache::new(&cfg, DType::BF16, &dev).unwrap();
    for (seq, split) in [(16usize, None), (20, Some(5))] {
        let x = rand(&dev, &[1, seq, cfg.hidden], 1.0);
        let mut ms = [0f64; 2];
        for (i, fused) in [false, true].into_iter().enumerate() {
            let warm = std::time::Instant::now();
            while warm.elapsed().as_secs_f64() < 0.5 {
                run(&gdn, &dims, &x, &base, split, fused);
                dev.synchronize().unwrap();
            }
            let t = std::time::Instant::now();
            for _ in 0..100 {
                run(&gdn, &dims, &x, &base, split, fused);
            }
            dev.synchronize().unwrap();
            ms[i] = t.elapsed().as_secs_f64() * 10.0;
        }
        println!(
            "S={seq} split={split:?}: op chain {:.3} ms/layer, fused {:.3} ms/layer ({:.2}x)",
            ms[0],
            ms[1],
            ms[0] / ms[1]
        );
        assert!(ms[1] < ms[0], "fused path is not faster");
    }
}

/// Long passes take the chunked recurrence (bf16 output); the fused front and
/// back ends must still match the op chain around it.
#[test]
#[ignore = "needs a CUDA GPU"]
fn fused_matches_op_chain_with_chunked_recurrence() {
    let dev = Device::new_cuda(0).unwrap();
    let cfg = Cfg {
        hidden: 512,
        k_heads: 4,
        v_heads: 8,
    };
    let (gdn, dims) = layer(&dev, &cfg);
    let base = GdnLayerCache::new(&cfg, DType::BF16, &dev).unwrap();
    for (seq, split) in [(130usize, None), (200, Some(70))] {
        let x = rand(&dev, &[1, seq, cfg.hidden], 1.0);
        let (ya, ca) = run_with(&gdn, &dims, &x, &base, split, false, false);
        let (yb, cb) = run_with(&gdn, &dims, &x, &base, split, true, false);
        let (yd, yn, ymax) = diff(&ya, &yb);
        let (cd, _, _) = diff(&ca.conv_state, &cb.conv_state);
        let (_, _, rmax) = diff(&ca.recurrent_state, &cb.recurrent_state);
        println!(
            "chunked S={seq} split={split:?}: out {yd}/{yn} (max {ymax:.3e}) | conv mismatches {cd} | state max {rmax:.1e}"
        );
        assert_eq!(cd, 0);
        assert!(rmax < 1e-3 && ymax < 0.05);
    }
}
