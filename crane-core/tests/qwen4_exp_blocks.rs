//! Qwen4-Exp building blocks against the transformers reference.
//!
//! Fixtures come from `tests/gen_qwen4_exp_fixtures.py`: a tiny random-weight
//! Qwen4-Exp model and the inputs/outputs of each block, hosted under
//! `qwen4_exp/tiny/` in the `crane-local-ai/test-data` dataset. Run with
//! `CRANE_TEST_DATA_DIR=<checkout> cargo test -p crane-core --test
//! qwen4_exp_blocks -- --ignored`.

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use crane_core::models::modules::moe::SparseMoeBlock;
use crane_core::models::qwen3_5::MRotaryEmbedding;
use crane_core::models::qwen4_exp::TextConfig;
use crane_core::models::qwen4_exp::hyper_connection::{GatedResidual, combine};
use crane_core::models::qwen4_exp::indexer::{IndexerCache, QsaIndexer};
use crane_core::models::qwen4_exp::ple::{PleLayer, PleState};
use crane_core::ops::gdn::{GatedDeltaNet, GdnDims, GdnInputProjectionKind, GdnLayerCache};
use crane_core::test_data::get_test_data_file;
use crane_core::{candle_core, candle_nn};

struct Fixture {
    cfg: TextConfig,
    /// The whole checkpoint (`model.*`, `lm_head.weight`).
    root: VarBuilder<'static>,
    /// Scoped to `model.`, where the blocks live.
    vb: VarBuilder<'static>,
    goldens: HashMap<String, Tensor>,
}

impl Fixture {
    fn load() -> Fixture {
        let file = |name: &str| {
            get_test_data_file(&format!("qwen4_exp/tiny/{name}"))
                .unwrap_or_else(|e| panic!("fixture {name}: {e}"))
        };
        let cfg = TextConfig::from_json(&std::fs::read_to_string(file("config.json")).unwrap())
            .expect("tiny config");
        // SAFETY: read-only mapping of a fixture file nothing else writes.
        let root = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[file("model.safetensors")],
                DType::F32,
                &Device::Cpu,
            )
        }
        .unwrap();
        let vb = root.pp("model");
        let goldens =
            candle_core::safetensors::load(file("goldens.safetensors"), &Device::Cpu).unwrap();
        Fixture {
            cfg,
            root,
            vb,
            goldens,
        }
    }

    fn golden(&self, name: &str) -> &Tensor {
        &self.goldens[name]
    }

    fn input_ids(&self) -> Vec<u32> {
        self.golden("input_ids")
            .to_vec1::<i64>()
            .unwrap()
            .into_iter()
            .map(|t| u32::try_from(t).unwrap())
            .collect()
    }
}

/// Largest absolute difference relative to the reference's largest value.
fn rel_diff(got: &Tensor, want: &Tensor) -> f32 {
    let got = got.to_dtype(DType::F32).unwrap();
    let scale = want
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let diff = (got - want).unwrap().abs().unwrap().max_all().unwrap();
    diff.to_scalar::<f32>().unwrap() / scale.max(1e-12)
}

fn assert_close(got: &Tensor, want: &Tensor, what: &str) {
    assert_eq!(got.dims(), want.dims(), "{what}: shape");
    let d = rel_diff(got, want);
    assert!(d < 1e-5, "{what}: relative difference {d}");
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn hyper_connection_mixers_match_reference() -> Result<()> {
    let f = Fixture::load();
    let c = &f.cfg;
    let hc_in = f.golden("hc_in");

    let block_mixer = GatedResidual::load(
        c.hidden_size,
        c.hc_count,
        c.hc_lowrank,
        c.rms_norm_eps,
        true,
        &f.vb.pp("layers.1.attn_hyper_connection"),
    )?;
    let (mixed, inject) = block_mixer.mix(hc_in)?;
    let inject = inject.expect("block mixers have injection weights");
    assert_close(&mixed, f.golden("hc_mixed"), "block mixer");
    assert_close(
        &inject.weights()?,
        f.golden("hc_inject"),
        "injection weights",
    );

    // Scattering back: every stream gains the block output times its weight.
    let combined = combine(hc_in, &mixed, &inject)?;
    let stream = |t: &Tensor, s: usize| t.narrow(1, s * c.hidden_size, c.hidden_size);
    for s in 0..c.hc_count {
        let want =
            (stream(hc_in, s)? + mixed.broadcast_mul(&inject.weights()?.narrow(1, s, 1)?)?)?;
        assert_close(&stream(&combined, s)?, &want, "combine");
    }

    let final_mixer = GatedResidual::load(
        c.hidden_size,
        c.hc_count,
        c.hc_lowrank,
        c.rms_norm_eps,
        false,
        &f.vb.pp("hyper_connection_mixer"),
    )?;
    let (out, none) = final_mixer.mix(hc_in)?;
    assert!(none.is_none());
    assert_close(&out, f.golden("hc_final"), "final mixer");
    Ok(())
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn ple_matches_reference_in_one_shot_and_chunked() -> Result<()> {
    let f = Fixture::load();
    let layer = f.cfg.ple_layer().expect("tiny model has a PLE layer");
    let ple = PleLayer::load(&f.cfg, 0, &f.vb.pp(format!("layers.{layer}.ple")))?;
    let tokens = f.input_ids();
    let hc_in = f.golden("hc_in");

    // Row ids alone, against the ids the reference fed its table.
    let mut state = PleState::new(ple.hash());
    let want_rows: Vec<u32> = f
        .golden("ple_ids")
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|r| u32::try_from(r).unwrap())
        .collect();
    let rows = ple.hash().rows(&mut ple.hash().new_window(), &tokens)?;
    assert_eq!(rows, want_rows, "hashed rows");

    let emb = ple.embed(&mut state, &tokens, &Device::Cpu)?;
    let out = ple.forward(hc_in, &emb, &mut state)?;
    assert_close(&out, f.golden("ple_out"), "PLE one shot");

    // A 10-token prefill then one token at a time must give the same output.
    let mut state = PleState::new(ple.hash());
    let mut outs = Vec::new();
    let mut start = 0;
    for len in std::iter::once(10).chain(std::iter::repeat_n(1, tokens.len() - 10)) {
        let emb = ple.embed(&mut state, &tokens[start..start + len], &Device::Cpu)?;
        outs.push(ple.forward(&hc_in.narrow(0, start, len)?, &emb, &mut state)?);
        start += len;
    }
    assert_close(&Tensor::cat(&outs, 0)?, f.golden("ple_out"), "PLE chunked");
    Ok(())
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn gated_delta_net_with_sigmoid_gate_matches_reference() -> Result<()> {
    let f = Fixture::load();
    let gdn = GatedDeltaNet::load(
        f.vb.pp("layers.1"),
        &f.cfg,
        GdnInputProjectionKind::Split,
        None,
    )?;
    let dims = GdnDims::new(&f.cfg);
    let mut cache = GdnLayerCache::new(&f.cfg, DType::F32, &Device::Cpu)?;
    let out = gdn.forward(&f.golden("block_in").unsqueeze(0)?, &dims, &mut cache)?;
    assert_close(&out.squeeze(0)?, f.golden("gdn_out"), "GDN");
    Ok(())
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn shared_expert_moe_matches_reference() -> Result<()> {
    let f = Fixture::load();
    let moe = SparseMoeBlock::new(
        &f.cfg.moe_config(),
        1,
        f.cfg.hidden_size,
        f.vb.pp("layers.1.mlp"),
        &Device::Cpu,
    )?;
    let out = moe.forward(&f.golden("block_in").unsqueeze(0)?)?;
    assert_close(&out.squeeze(0)?, f.golden("moe_out"), "MoE");
    Ok(())
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn qsa_indexer_selects_reference_cells_in_one_shot_and_chunked() -> Result<()> {
    let f = Fixture::load();
    let c = &f.cfg;
    let indexer = QsaIndexer::load(
        c.indexer()?,
        c.hidden_size,
        c.rms_norm_eps,
        &f.vb.pp("layers.3.self_attn.indexer"),
    )?;
    let rope = MRotaryEmbedding::from_params(
        c.rot_dim(),
        c.rope_parameters.rope_theta,
        c.max_position_embeddings,
        &c.rope_parameters.mrope_section,
        &Device::Cpu,
    )?;
    let x = f.golden("block_in");
    let seq = x.dim(0)?;
    let (cos, sin) = rope.cos_sin(0, seq)?;
    let want = f.golden("indexer_selected").to_vec2::<u8>()?;
    let selected = |mask: &Tensor| -> Vec<Vec<bool>> {
        mask.to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .map(|r| r.into_iter().map(|v| v == 0.0).collect())
            .collect()
    };
    let want: Vec<Vec<bool>> = want
        .into_iter()
        .map(|r| r.into_iter().map(|v| v != 0).collect())
        .collect();

    let mask = indexer.select(x, &cos, &sin, c.rot_dim(), &mut IndexerCache::default())?;
    assert_eq!(selected(&mask), want, "one shot");

    // Chunked: a 13-token prefill, then single tokens; each query row is
    // compared over the cells that existed when it ran.
    let mut cache = IndexerCache::default();
    let mut start = 0;
    for len in std::iter::once(13).chain(std::iter::repeat_n(1, seq - 13)) {
        let mask = indexer.select(
            &x.narrow(0, start, len)?,
            &cos,
            &sin,
            c.rot_dim(),
            &mut cache,
        )?;
        for (i, row) in selected(&mask).into_iter().enumerate() {
            assert_eq!(row, want[start + i][..start + len], "query {}", start + i);
        }
        start += len;
    }
    Ok(())
}

#[test]
#[ignore = "needs crane-local-ai/test-data (CRANE_TEST_DATA_DIR or network)"]
fn tiny_model_logits_match_reference_one_shot_and_incremental() -> Result<()> {
    use crane_core::models::qwen4_exp::Qwen4ExpTextModel;

    let f = Fixture::load();
    let tokens = f.input_ids();
    let want = f.golden("logits");
    let mut model = Qwen4ExpTextModel::load_hf(f.cfg.clone(), &f.root)?;

    // Every position in one pass.
    assert_close(&model.forward_all(&tokens)?, want, "one shot");

    // One token at a time: every cache (GDN, K/V, indexer, PLE) carries over.
    model.reset()?;
    for (i, &t) in tokens.iter().enumerate() {
        let got = model.forward(&[t])?;
        assert_close(&got, &want.get(i)?, &format!("decode step {i}"));
    }

    // A 10-token prefill, then decode.
    model.reset()?;
    let prefill = model.forward_all(&tokens[..10])?;
    assert_close(&prefill, &want.narrow(0, 0, 10)?, "prefill");
    for (i, &t) in tokens.iter().enumerate().skip(10) {
        assert_close(
            &model.forward(&[t])?,
            &want.get(i)?,
            &format!("after prefill, step {i}"),
        );
    }
    Ok(())
}
