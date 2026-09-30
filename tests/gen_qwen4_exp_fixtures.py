"""Generate Qwen4-Exp golden fixtures for crane-core's `qwen4_exp` tests.

Builds a tiny random-weight Qwen4-Exp text model with the reference
implementation (transformers `models/qwen4_exp`) and records the inputs and
outputs of each building block, plus full-model logits. Output layout, meant
for the `crane-local-ai/test-data` dataset (see `crane-core/src/test_data.rs`):

    <out>/qwen4_exp/tiny/config.json         text config of the tiny model
    <out>/qwen4_exp/tiny/model.safetensors   its weights (f32, HF names)
    <out>/qwen4_exp/tiny/goldens.safetensors per-module inputs and outputs

Usage: python tests/gen_qwen4_exp_fixtures.py <out-dir>
Needs torch and a transformers build that ships `Qwen4ExpForCausalLM`.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

SEQ_LEN = 24
# EOS inside the prompt exercises the n-gram window reset.
EOS_POSITION = 9


def tiny_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=300,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        full_attention_interval=4,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        hc_count=4,
        hc_lowrank=24,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        num_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=48,
        output_gate_type="sigmoid",
        rms_norm_eps=1e-6,
        eos_token_id=299,
        bos_token_id=299,
        pad_token_id=None,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )


def randomize(model: torch.nn.Module) -> None:
    """Replace the reference init (zero norms, zero PLE conv) with random
    weights, so every parameter actually shapes the goldens. Integer buffers
    (the n-gram hash constants) are left alone."""
    for name, param in model.named_parameters():
        if name.endswith("A_log") or name.endswith("dt_bias"):
            continue
        std = 1.0 if "embed" in name else 0.1
        param.data.normal_(0.0, std)


@torch.no_grad()
def main(out: Path) -> None:
    torch.manual_seed(0)
    cfg = tiny_config()
    model = Qwen4ExpForCausalLM(cfg).eval()
    randomize(model)
    text = model.model
    hc_hidden = cfg.hc_count * cfg.hidden_size

    input_ids = torch.randint(0, cfg.vocab_size - 1, (1, SEQ_LEN))
    input_ids[0, EOS_POSITION] = cfg.eos_token_id

    goldens = {"input_ids": input_ids[0]}

    # Hyper-connections: one layer's input mixer, and the final mixer.
    hc_in = torch.randn(1, SEQ_LEN, hc_hidden)
    mixed, _, inject = text.layers[1].attn_hyper_connection(hc_in)
    goldens |= {
        "hc_in": hc_in[0],
        "hc_mixed": mixed[0],
        "hc_inject": inject[0],
        "hc_final": text.hyper_connection_mixer(hc_in)[0],
    }

    # PLE on layer index 1 (`ple_layer_ids` is one-indexed), with the hashed
    # row ids captured on their way into the table.
    ple = text.layers[1].ple
    captured = {}
    table = ple.ple_embedding.ngram_embedding
    handle = table.register_forward_hook(lambda _m, args, _o: captured.update(ids=args[0]))
    goldens["ple_out"] = ple(hc_in, input_ids, None)[0]
    handle.remove()
    goldens["ple_ids"] = captured["ids"][0]

    # Gated delta net (sigmoid output gate) and the shared-expert MoE.
    x = torch.randn(1, SEQ_LEN, cfg.hidden_size)
    goldens |= {
        "block_in": x[0],
        "gdn_out": text.layers[1].linear_attn(x)[0],
        "moe_out": text.layers[1].mlp(x)[0],
    }

    # QSA indexer of the first full-attention layer: which cells each query
    # keeps. SEQ_LEN > indexer_budget, so later queries drop whole blocks.
    positions = torch.arange(SEQ_LEN).view(1, 1, -1).expand(3, 1, -1)
    cos_sin = text.rotary_emb(x, positions)
    causal = torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool).tril().view(1, 1, SEQ_LEN, SEQ_LEN)
    selected = text.layers[3].self_attn.indexer(x, cos_sin, causal, None)
    goldens["indexer_selected"] = selected[0, 0].to(torch.uint8)

    goldens["logits"] = model(input_ids).logits[0]

    tiny = out / "qwen4_exp" / "tiny"
    tiny.mkdir(parents=True, exist_ok=True)
    (tiny / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, tiny / "model.safetensors")
    save_file({k: v.contiguous() for k, v in goldens.items()}, tiny / "goldens.safetensors")
    print(f"wrote {tiny}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
