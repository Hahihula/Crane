// SPDX-License-Identifier: MIT

//! Qwen4-Exp text config, from HF's `text_config` or from GGUF metadata
//! (`general.architecture = "qwen4exp"`).

use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;
use candle_core::{Result, bail};
use serde::Deserialize;

use crate::models::modules::moe::MoeConfig;
use crate::models::qwen3_5::RopeParameters;
use crate::ops::gdn::{GateActivation, GdnConfig};
use crate::quantized::gguf_metadata::GgufMetadata;

/// Token mixer of one decoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    /// Gated delta net (recurrent, constant-size state).
    LinearAttention,
    /// Softmax attention restricted by the QSA indexer. The released
    /// checkpoints call it `full_attention`.
    #[serde(alias = "full_attention")]
    IndexedAttention,
}

/// HF `eos_token_id`: a single id or a list whose first entry counts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl TokenIds {
    fn first(&self) -> Option<u32> {
        match self {
            Self::One(id) => Some(*id),
            Self::Many(ids) => ids.first().copied(),
        }
    }
}

/// Text model config. Field names follow HF `Qwen4ExpTextConfig`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    /// GDN output gate activation; falls back to `hidden_act` when absent.
    #[serde(default)]
    pub output_gate_type: Option<String>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    /// Parallel residual streams of the hyper-connections.
    pub hc_count: usize,
    /// Rank of the hyper-connection input mixer.
    pub hc_lowrank: usize,

    /// One-indexed ids of the layers carrying a PLE module (HF convention).
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    /// Width of the concatenated n-gram embedding; HF defaults it to
    /// `hidden_size`.
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv_kernel")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    /// Lower bound of the per-head prime vocab sizes. Only the safetensors
    /// path derives the hash constants from it; GGUF stores them directly.
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    eos_token_id: Option<TokenIds>,

    #[serde(default)]
    pub indexer_n_heads: Option<usize>,
    #[serde(default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(default)]
    pub indexer_head_dim: Option<usize>,
    /// Token budget of complete blocks each query may attend to.
    #[serde(default)]
    pub indexer_budget: Option<usize>,
    /// Tokens mean-pooled into one indexer block.
    #[serde(default)]
    pub indexer_compress_ratio: Option<usize>,
}

fn default_hidden_act() -> String {
    "silu".into()
}
fn default_ple_conv_kernel() -> usize {
    4
}
fn default_ngram_size() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_ngram_vocab_size_base() -> u64 {
    20_000_000
}
fn default_seed() -> u64 {
    1234
}

/// QSA indexer dimensions (present on every indexed-attention layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexerConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl TextConfig {
    /// Parse the `text_config` object of a HF `config.json` (or a bare text
    /// config) and validate it.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON does not match or fails [`Self::validate`].
    pub fn from_json(json: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(json).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let text = value.get("text_config").cloned().unwrap_or(value);
        let cfg: Self =
            serde_json::from_value(text).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Build the config from GGUF metadata of the first shard.
    ///
    /// The file does not carry the n-gram hash seed or vocab base; it stores
    /// the derived constants instead (see `ple::NgramHash::from_gguf`).
    ///
    /// # Errors
    ///
    /// Returns an error if the architecture is not `qwen4exp`, a required key
    /// is missing, or the result fails [`Self::validate`].
    pub fn from_gguf(md: &HashMap<String, Value>) -> Result<Self> {
        let md = GgufMetadata(md);
        let arch = md.string("general.architecture")?;
        if arch != "qwen4exp" {
            bail!("expected GGUF architecture qwen4exp, got {arch}")
        }
        let key = |k: &str| format!("qwen4exp.{k}");
        let usize_of = |k: &str| md.usize(&key(k));

        let num_hidden_layers = usize_of("block_count")?;
        let interval = usize_of("full_attention_interval")?;
        if interval == 0 {
            bail!("qwen4exp.full_attention_interval must be positive")
        }
        let layer_types = (0..num_hidden_layers)
            .map(|i| {
                if (i + 1) % interval == 0 {
                    LayerType::IndexedAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();

        let head_dim = usize_of("attention.key_length")?;
        let rot_dim = usize_of("rope.dimension_count")?;
        let sections = md.usizes(&key("rope.dimension_sections"))?;
        #[allow(clippy::cast_precision_loss)]
        let rope_parameters = RopeParameters {
            rope_theta: f64::from(md.f32(&key("rope.freq_base"))?),
            mrope_section: sections.into_iter().filter(|&s| s > 0).collect(),
            partial_rotary_factor: rot_dim as f64 / head_dim as f64,
            mrope_interleaved: true,
        };

        // Every indexed layer shares one compress ratio; linear layers list 0.
        let ratios = md.usizes(&key("attention.compress_ratios"))?;
        let mut indexed_ratios = ratios.iter().copied().filter(|&r| r > 0);
        let compress_ratio = indexed_ratios.next();
        if indexed_ratios.any(|r| Some(r) != compress_ratio) {
            bail!("qwen4exp.attention.compress_ratios mixes ratios: {ratios:?}")
        }

        let ple_layers = md.usizes(&key("ple.layers")).unwrap_or_default();
        let ngram_size = usize_of("ple.ngram_size").unwrap_or_else(|_| default_ngram_size());
        let heads_per_ngram =
            usize_of("ple.heads_per_ngram").unwrap_or_else(|_| default_heads_per_ngram());
        let ple_embed_dim = if ple_layers.is_empty() {
            None
        } else {
            Some(usize_of("embedding_length_per_layer_input")? * (ngram_size - 1) * heads_per_ngram)
        };
        let eos = if ple_layers.is_empty() {
            None
        } else {
            Some(TokenIds::One(md.u32(&key("ple.eos_token_id"))?))
        };

        let cfg = Self {
            vocab_size: md.array_len("tokenizer.ggml.tokens")?,
            hidden_size: usize_of("embedding_length")?,
            num_hidden_layers,
            num_attention_heads: usize_of("attention.head_count")?,
            num_key_value_heads: usize_of("attention.head_count_kv")?,
            head_dim,
            rms_norm_eps: f64::from(md.f32(&key("attention.layer_norm_rms_epsilon"))?),
            max_position_embeddings: usize_of("context_length")?,
            layer_types,
            rope_parameters,
            tie_word_embeddings: false,
            linear_conv_kernel_dim: usize_of("ssm.conv_kernel")?,
            linear_key_head_dim: usize_of("ssm.state_size")?,
            linear_value_head_dim: usize_of("ssm.state_size")?,
            linear_num_key_heads: usize_of("ssm.group_count")?,
            linear_num_value_heads: usize_of("ssm.time_step_rank")?,
            // llama.cpp's qwen4exp graph hardcodes the sigmoid GDN gate.
            output_gate_type: Some("sigmoid".into()),
            hidden_act: default_hidden_act(),
            num_experts: usize_of("expert_count")?,
            num_experts_per_tok: usize_of("expert_used_count")?,
            moe_intermediate_size: usize_of("expert_feed_forward_length")?,
            shared_expert_intermediate_size: usize_of("expert_shared_feed_forward_length")?,
            hc_count: usize_of("hyper_connection.count")?,
            hc_lowrank: usize_of("hyper_connection.low_rank")?,
            // GGUF lists them zero-indexed.
            ple_layer_ids: ple_layers.iter().map(|l| l + 1).collect(),
            ple_embed_dim,
            ple_conv_kernel_size: usize_of("ple.conv_kernel")
                .unwrap_or_else(|_| default_ple_conv_kernel()),
            ngram_size,
            heads_per_ngram,
            ngram_vocab_size_base: default_ngram_vocab_size_base(),
            seed: default_seed(),
            eos_token_id: eos,
            indexer_n_heads: compress_ratio
                .map(|_| usize_of("attention.indexer.head_count"))
                .transpose()?,
            indexer_kv_heads: compress_ratio.map(|_| 1),
            indexer_head_dim: compress_ratio
                .map(|_| usize_of("attention.indexer.key_length"))
                .transpose()?,
            indexer_budget: compress_ratio
                .map(|_| usize_of("attention.indexer.top_k"))
                .transpose()?,
            indexer_compress_ratio: compress_ratio,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Check the architecture invariants the building blocks rely on
    /// (mirrors HF `Qwen4ExpTextConfig.validate_architecture`).
    ///
    /// # Errors
    ///
    /// Returns an error describing the first violated invariant.
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.num_hidden_layers {
            bail!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )
        }
        if self.hc_count <= 1 {
            bail!("qwen4_exp requires hc_count > 1, got {}", self.hc_count)
        }
        if !(1..=self.num_experts).contains(&self.num_experts_per_tok) {
            bail!(
                "num_experts_per_tok {} must be in [1, num_experts = {}]",
                self.num_experts_per_tok,
                self.num_experts
            )
        }
        self.output_gate_activation()?;
        if self.layer_types.contains(&LayerType::IndexedAttention) {
            let idx = self.indexer()?;
            if self.indexer_kv_heads != Some(1) {
                bail!("the QSA indexer requires exactly one key head")
            }
            if idx.budget % idx.compress_ratio != 0 {
                bail!("indexer_budget must be divisible by indexer_compress_ratio")
            }
            if self.rot_dim() > idx.head_dim {
                bail!(
                    "rotary dim {} does not fit the indexer head dim {}",
                    self.rot_dim(),
                    idx.head_dim
                )
            }
        }
        if let Some(layer) = self.ple_layer() {
            if self.ple_layer_ids.len() != 1 {
                bail!(
                    "only one PLE layer is supported, got {:?}",
                    self.ple_layer_ids
                )
            }
            if self.layer_types.get(layer) != Some(&LayerType::LinearAttention) {
                bail!("PLE layer {layer} (zero-indexed) must be a linear-attention layer")
            }
            if self.ngram_size < 2 || self.heads_per_ngram == 0 {
                bail!("PLE needs ngram_size >= 2 and heads_per_ngram > 0")
            }
            if !self.ple_embed_dim().is_multiple_of(self.ngram_heads()) {
                bail!(
                    "ple_embed_dim {} is not divisible by the {} n-gram heads",
                    self.ple_embed_dim(),
                    self.ngram_heads()
                )
            }
            if self.ple_eos_token_id().is_none() {
                bail!("eos_token_id must be set when a PLE layer is present")
            }
        }
        Ok(())
    }

    /// Partial-rotary width shared by attention and the indexer.
    #[must_use]
    pub fn rot_dim(&self) -> usize {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let rot = (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        rot
    }

    /// QSA indexer dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if any indexer field is missing or zero.
    pub fn indexer(&self) -> Result<IndexerConfig> {
        let field = |v: Option<usize>, name: &str| match v {
            Some(n) if n > 0 => Ok(n),
            _ => Err(candle_core::Error::Msg(format!(
                "indexed attention needs a positive {name}"
            ))),
        };
        Ok(IndexerConfig {
            n_heads: field(self.indexer_n_heads, "indexer_n_heads")?,
            head_dim: field(self.indexer_head_dim, "indexer_head_dim")?,
            budget: field(self.indexer_budget, "indexer_budget")?,
            compress_ratio: field(self.indexer_compress_ratio, "indexer_compress_ratio")?,
        })
    }

    /// Zero-indexed layer carrying the PLE module, if any.
    #[must_use]
    pub fn ple_layer(&self) -> Option<usize> {
        self.ple_layer_ids.first().map(|id| id - 1)
    }

    /// Width of the concatenated n-gram embedding.
    #[must_use]
    pub fn ple_embed_dim(&self) -> usize {
        self.ple_embed_dim.unwrap_or(self.hidden_size)
    }

    /// Hashed embedding heads: `heads_per_ngram` for each n-gram order 2..=n.
    #[must_use]
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// EOS id the n-gram window resets on.
    #[must_use]
    pub fn ple_eos_token_id(&self) -> Option<u32> {
        self.eos_token_id.as_ref().and_then(TokenIds::first)
    }

    /// Routed-expert config for the sparse `MoE` blocks.
    #[must_use]
    pub fn moe_config(&self) -> MoeConfig {
        MoeConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            // HF `Qwen4ExpTextConfig.norm_topk_prob` is always true.
            norm_topk_prob: true,
            decoder_sparse_step: None,
        }
    }
}

impl GdnConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn output_gate_activation(&self) -> Result<GateActivation> {
        GateActivation::from_name(self.output_gate_type.as_deref().unwrap_or(&self.hidden_act))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Text config of `Qwen/Qwen3.8-Flash-Next` (the fields this crate reads).
    const FLASH_NEXT_CONFIG: &str = r#"{
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "vocab_size": 248320, "hidden_size": 2560, "num_hidden_layers": 48,
            "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "rms_norm_eps": 1e-06, "max_position_embeddings": 262144,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention", "linear_attention", "linear_attention", "linear_attention",
                "full_attention"],
            "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "output_gate_type": "sigmoid", "hidden_act": "silu",
            "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "hc_count": 4, "hc_lowrank": 320,
            "ple_layer_ids": [2], "ple_embed_dim": 2560, "ple_conv_kernel_size": 4,
            "ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 20000000,
            "eos_token_id": 248044,
            "indexer_budget": 2048, "indexer_compress_ratio": 4, "indexer_head_dim": 128,
            "indexer_kv_heads": 1, "indexer_n_heads": 4,
            "tie_word_embeddings": false
        }
    }"#;

    #[test]
    fn parses_flash_next_text_config() {
        let cfg = TextConfig::from_json(FLASH_NEXT_CONFIG).unwrap();
        assert_eq!(cfg.layer_types[3], LayerType::IndexedAttention);
        assert_eq!(cfg.layer_types[2], LayerType::LinearAttention);
        assert_eq!(cfg.ple_layer(), Some(1));
        assert_eq!(cfg.ngram_heads(), 16);
        assert_eq!(cfg.ple_embed_dim() / cfg.ngram_heads(), 160);
        assert_eq!(cfg.ple_eos_token_id(), Some(248_044));
        assert_eq!(cfg.rot_dim(), 64);
        assert_eq!(
            cfg.output_gate_activation().unwrap(),
            GateActivation::Sigmoid
        );
        assert_eq!(
            cfg.indexer().unwrap(),
            IndexerConfig {
                n_heads: 4,
                head_dim: 128,
                budget: 2048,
                compress_ratio: 4
            }
        );
    }

    #[test]
    fn rejects_ple_on_attention_layer() {
        let bad = FLASH_NEXT_CONFIG.replace("\"ple_layer_ids\": [2]", "\"ple_layer_ids\": [4]");
        let err = TextConfig::from_json(&bad).unwrap_err().to_string();
        assert!(err.contains("linear-attention"), "{err}");
    }

    /// Metadata of `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`,
    /// minus the tokenizer (the vocab size comes from the token list).
    fn flash_next_gguf_metadata() -> HashMap<String, Value> {
        let u = |v: u32| Value::U32(v);
        let arr = |v: &[u32]| Value::Array(v.iter().map(|&x| Value::U32(x)).collect());
        let mut ratios = vec![0; 48];
        for r in ratios.iter_mut().skip(3).step_by(4) {
            *r = 4;
        }
        let mut md: HashMap<String, Value> = [
            ("block_count", u(48)),
            ("context_length", u(262_144)),
            ("embedding_length", u(2560)),
            ("attention.head_count", u(24)),
            ("attention.head_count_kv", u(2)),
            ("rope.dimension_sections", arr(&[11, 11, 10, 0])),
            ("rope.freq_base", Value::F32(10_000_000.0)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("expert_count", u(256)),
            ("expert_used_count", u(10)),
            ("attention.key_length", u(256)),
            ("attention.value_length", u(256)),
            ("expert_feed_forward_length", u(640)),
            ("expert_shared_feed_forward_length", u(640)),
            ("ssm.conv_kernel", u(4)),
            ("ssm.state_size", u(128)),
            ("ssm.group_count", u(16)),
            ("ssm.time_step_rank", u(48)),
            ("ssm.inner_size", u(6144)),
            ("full_attention_interval", u(4)),
            ("rope.dimension_count", u(64)),
            ("hyper_connection.count", u(4)),
            ("hyper_connection.low_rank", u(320)),
            ("attention.indexer.head_count", u(4)),
            ("attention.indexer.key_length", u(128)),
            ("attention.indexer.top_k", u(2048)),
            ("attention.compress_ratios", arr(&ratios)),
            ("ple.layers", arr(&[1])),
            ("ple.ngram_size", u(3)),
            ("ple.heads_per_ngram", u(8)),
            ("ple.conv_kernel", u(4)),
            ("ple.eos_token_id", u(248_044)),
            ("embedding_length_per_layer_input", u(160)),
        ]
        .into_iter()
        .map(|(k, v)| (format!("qwen4exp.{k}"), v))
        .collect();
        md.insert(
            "general.architecture".into(),
            Value::String("qwen4exp".into()),
        );
        md.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); 248_320]),
        );
        md
    }

    #[test]
    fn gguf_metadata_matches_hf_config() {
        let gguf = TextConfig::from_gguf(&flash_next_gguf_metadata()).unwrap();
        let hf = TextConfig::from_json(FLASH_NEXT_CONFIG).unwrap();
        assert_eq!(gguf.layer_types, hf.layer_types);
        assert_eq!(gguf.ple_layer(), hf.ple_layer());
        assert_eq!(gguf.ple_embed_dim(), hf.ple_embed_dim());
        assert_eq!(gguf.ple_eos_token_id(), hf.ple_eos_token_id());
        assert_eq!(gguf.indexer().unwrap(), hf.indexer().unwrap());
        assert_eq!(gguf.rot_dim(), hf.rot_dim());
        assert_eq!(
            gguf.rope_parameters.mrope_section,
            hf.rope_parameters.mrope_section
        );
        assert_eq!(
            gguf.output_gate_activation().unwrap(),
            hf.output_gate_activation().unwrap()
        );
        assert_eq!(gguf.vocab_size, hf.vocab_size);
        // The pruned release keeps 256 of the 512 experts.
        assert_eq!((gguf.num_experts, hf.num_experts), (256, 512));
    }
}
