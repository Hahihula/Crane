#!/usr/bin/env python3
"""Cross-check a Qwen 3.5-family checkpoint against what Crane's `qwen3_5`
loader expects — from the safetensors *headers* only, so it costs a few
hundred KB of range requests instead of a 55 GB download.

Manual tool, not part of `cargo test`. Use it before porting effort: if every
tensor name and shape matches, the checkpoint loads through the existing
module and the remaining work is config plumbing, not modeling.

    ./tests/test_qwen35_family_shapes.py                     # Qwen3.8-27B
    ./tests/test_qwen35_family_shapes.py Qwen/Qwen3.6-27B
    ./tests/test_qwen35_family_shapes.py Qwen/Qwen3.5-4B

Exits non-zero on any missing tensor or shape mismatch. Tensors present in the
checkpoint but never requested are reported, not failed — Qwen 3.6/3.8 ship an
`mtp.*` draft head that Crane deliberately does not load.
"""
import json
import struct
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.8-27B"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"


def fetch(name):
    return urllib.request.urlopen(BASE + name).read()


def header(shard):
    """Read a safetensors header: u64 length prefix, then that many JSON bytes."""
    req = urllib.request.Request(BASE + shard, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(req).read())[0]
    req = urllib.request.Request(BASE + shard, headers={"Range": f"bytes=8-{7 + n}"})
    return json.loads(urllib.request.urlopen(req).read())


def load_headers():
    try:
        shards = sorted(set(json.loads(fetch("model.safetensors.index.json"))["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]  # single-shard checkpoint
    hdrs = {}
    with ThreadPoolExecutor(8) as ex:
        for h in ex.map(header, shards):
            hdrs.update(h)
    hdrs.pop("__metadata__", None)
    return hdrs, shards


def expected(cfg):
    """Every tensor `crane-core/src/models/qwen3_5/` asks for, with its shape."""
    t = cfg["text_config"]
    H, HD = t["hidden_size"], t["head_dim"]
    NH, NKV = t["num_attention_heads"], t["num_key_value_heads"]
    NV = t["linear_num_value_heads"]
    KD = t["linear_num_key_heads"] * t["linear_key_head_dim"]
    VD = NV * t["linear_value_head_dim"]
    INTER = t["intermediate_size"]
    interval = t.get("full_attention_interval", 4)
    kinds = t.get("layer_types") or [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(t["num_hidden_layers"])
    ]

    want = {
        "model.language_model.embed_tokens.weight": [t["vocab_size"], H],
        "model.language_model.norm.weight": [H],
    }
    if not t.get("tie_word_embeddings", False):
        want["lm_head.weight"] = [t["vocab_size"], H]

    for i, kind in enumerate(kinds):
        p = f"model.language_model.layers.{i}"
        want[f"{p}.input_layernorm.weight"] = [H]
        want[f"{p}.post_attention_layernorm.weight"] = [H]
        want[f"{p}.mlp.gate_proj.weight"] = [INTER, H]
        want[f"{p}.mlp.up_proj.weight"] = [INTER, H]
        want[f"{p}.mlp.down_proj.weight"] = [H, INTER]
        if kind == "full_attention":
            # q_proj is 2x wide when the output gate is fused in.
            gate = 2 if t.get("attn_output_gate", True) else 1
            want[f"{p}.self_attn.q_proj.weight"] = [gate * NH * HD, H]
            want[f"{p}.self_attn.k_proj.weight"] = [NKV * HD, H]
            want[f"{p}.self_attn.v_proj.weight"] = [NKV * HD, H]
            want[f"{p}.self_attn.o_proj.weight"] = [H, NH * HD]
            want[f"{p}.self_attn.q_norm.weight"] = [HD]
            want[f"{p}.self_attn.k_norm.weight"] = [HD]
        else:
            want[f"{p}.linear_attn.in_proj_qkv.weight"] = [2 * KD + VD, H]
            want[f"{p}.linear_attn.in_proj_z.weight"] = [VD, H]
            want[f"{p}.linear_attn.in_proj_a.weight"] = [NV, H]
            want[f"{p}.linear_attn.in_proj_b.weight"] = [NV, H]
            want[f"{p}.linear_attn.out_proj.weight"] = [H, VD]
            want[f"{p}.linear_attn.conv1d.weight"] = [2 * KD + VD, 1, t["linear_conv_kernel_dim"]]
            want[f"{p}.linear_attn.norm.weight"] = [t["linear_value_head_dim"]]
            want[f"{p}.linear_attn.A_log"] = [NV]
            want[f"{p}.linear_attn.dt_bias"] = [NV]

    v = cfg.get("vision_config")
    if v:
        VH, VI = v["hidden_size"], v["intermediate_size"]
        MERGED = VH * v["spatial_merge_size"] ** 2
        want["model.visual.patch_embed.proj.weight"] = [
            VH, v["in_channels"], v["temporal_patch_size"], v["patch_size"], v["patch_size"]]
        want["model.visual.patch_embed.proj.bias"] = [VH]
        want["model.visual.pos_embed.weight"] = [v["num_position_embeddings"], VH]
        for i in range(v["depth"]):
            p = f"model.visual.blocks.{i}"
            want[f"{p}.attn.qkv.weight"] = [3 * VH, VH]
            want[f"{p}.attn.qkv.bias"] = [3 * VH]
            want[f"{p}.attn.proj.weight"] = [VH, VH]
            want[f"{p}.attn.proj.bias"] = [VH]
            want[f"{p}.mlp.linear_fc1.weight"] = [VI, VH]
            want[f"{p}.mlp.linear_fc1.bias"] = [VI]
            want[f"{p}.mlp.linear_fc2.weight"] = [VH, VI]
            want[f"{p}.mlp.linear_fc2.bias"] = [VH]
            for n in ("norm1", "norm2"):
                want[f"{p}.{n}.weight"] = [VH]
                want[f"{p}.{n}.bias"] = [VH]
        want["model.visual.merger.linear_fc1.weight"] = [MERGED, MERGED]
        want["model.visual.merger.linear_fc1.bias"] = [MERGED]
        want["model.visual.merger.linear_fc2.weight"] = [v["out_hidden_size"], MERGED]
        want["model.visual.merger.linear_fc2.bias"] = [v["out_hidden_size"]]
        # Pre-merge norm, so `hidden_size` — matches PatchMerger::new with
        # use_postshuffle_norm = false.
        want["model.visual.merger.norm.weight"] = [VH]
        want["model.visual.merger.norm.bias"] = [VH]
    return want


def main():
    cfg = json.loads(fetch("config.json"))
    print(f"{REPO}: model_type={cfg.get('model_type')} architectures={cfg.get('architectures')}")

    gate = cfg["text_config"].get("output_gate_type")
    if gate is not None and gate not in ("swish", "silu"):
        print(f"  !! output_gate_type={gate!r} — the GDN gate is swish-only in Crane")

    hdrs, shards = load_headers()
    want = expected(cfg)

    bad = []
    for name, shape in want.items():
        got = hdrs.get(name)
        if got is None:
            bad.append(f"MISSING {name}")
        elif got["shape"] != shape:
            bad.append(f"SHAPE   {name}: want {shape}, got {got['shape']}")

    extra = sorted(k for k in hdrs if k not in want)
    print(f"checked {len(want)} tensors across {len(shards)} shard(s): {len(bad)} mismatches")
    for b in bad[:20]:
        print("  ", b)
    if extra:
        print(f"not loaded by Crane ({len(extra)}): {', '.join(sorted({k.split('.')[0] for k in extra}))}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
