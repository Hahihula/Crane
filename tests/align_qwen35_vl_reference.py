# PyTorch (reference) harness for aligning Crane's Qwen3.5-2B multimodal
# inference against HuggingFace Transformers.
# Produces, under tests/align_out/:
#   pixel_values.bin  - f32 [P, D]  (P patches, D = 3*t*patch*patch)
#   grid_thw.bin      - i32 [1, 3]  (t, h, w patch-grid)
#   input_ids.bin     - i32 [S]     (Crane-style prompt, image-pad expanded)
#   ref_logits.bin    - f32 [V]     (last-position prefill logits)
#   emb_torch.bin     - f32 [M, 2048] (projected vision embeddings, pre-splice)
#   ref_gen_ids.bin   - i32 [G]     (greedy decode, same prompt as Crane)
#   torch_text.txt    - decoded greedy text (thinking stripped)
#   summary.json      - timing numbers
#
# Run: python3 tests/align_qwen35_vl_reference.py

import json
import os
import struct
import time

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration, Qwen3VLProcessor

MODEL = "checkpoints/Qwen3.5-2B"
IMAGE = "data/images/a.jpg"
PROMPT = "Describe this image briefly."
OUT = "tests/align_out"
MAX_NEW = 96
DEVICE = "mps"
DTYPE = torch.float16

os.makedirs(OUT, exist_ok=True)


def write_array(path, arr, dtype):
    """Write a tiny header (rank + dims) then raw little-endian data."""
    arr = np.asarray(arr)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<i", int(d)))
        f.write(np.ascontiguousarray(arr, dtype=dtype).tobytes())


print(f"[torch] loading {MODEL} on {DEVICE} / {DTYPE}", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=DTYPE, device_map={"": DEVICE}
).eval()
proc = Qwen3VLProcessor.from_pretrained(MODEL)
tok = proc.tokenizer

cfg = json.load(open(os.path.join(MODEL, "config.json")))
IMAGE_TOKEN_ID = cfg["image_token_id"]
MERGE = 2  # preprocessor merge_size

img = Image.open(IMAGE).convert("RGB")

# ---- Replicate Crane's render_prompt 1:1, let the processor expand the
#      image pad + emit mm_token_type_ids (required by the HF forward). ----
rendered = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    + PROMPT
    + "<|im_end|>\n<|im_start|>assistant\n"
)
inputs = proc(images=[img], text=[rendered], return_tensors="pt")
pv = inputs["pixel_values"]            # [P, D] float32 on cpu
grid = inputs["image_grid_thw"]        # [1, 3]
P, D = pv.shape
print(f"[torch] image -> {P} patches, grid={tuple(grid[0].tolist())}", flush=True)

# Cross-check: our hand-expanded ids must equal the processor's input_ids,
# otherwise Crane (which hand-renders) and this reference would diverge for a
# reason unrelated to the model math.
n_img_tok = int(grid[0, 0] * (grid[0, 1] // MERGE) * (grid[0, 2] // MERGE))
base_ids = tok.encode(rendered, add_special_tokens=False)
expanded = []
for t in base_ids:
    if t == IMAGE_TOKEN_ID:
        expanded.extend([IMAGE_TOKEN_ID] * n_img_tok)
    else:
        expanded.append(t)
proc_ids = inputs["input_ids"][0].tolist()
print(
    f"[torch] prompt tokens={len(proc_ids)} (image pad expanded x{n_img_tok})",
    flush=True,
)
assert proc_ids.count(IMAGE_TOKEN_ID) == n_img_tok
assert proc_ids == expanded, "processor/ Crane render mismatch!"

eos_ids = []
for name in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
    i = tok.convert_tokens_to_ids(name)
    if isinstance(i, int) and i not in eos_ids:
        eos_ids.append(i)
print(f"[torch] eos ids={eos_ids}", flush=True)

# ---- Vision tower reference (projected embeddings, identical pixels) ----
pv_dev = pv.to(DEVICE)
grid_dev = grid.to(DEVICE)
with torch.no_grad():
    vis = model.model.visual(pv_dev, grid_dev)
    lh = vis.last_hidden_state if hasattr(vis, "last_hidden_state") else vis[0]
    emb = model.model.visual.merger(lh)  # [M, 2048]
emb = emb.detach().float().cpu().numpy()
print(f"[torch] vision emb shape={emb.shape}", flush=True)

# ---- Prefill logits (last position) with identical inputs as Crane ----
input_ids = torch.tensor([expanded], device=DEVICE, dtype=torch.long)
attn = torch.ones_like(input_ids)
mm_tok = inputs["mm_token_type_ids"].to(DEVICE)
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        pixel_values=pv_dev,
        image_grid_thw=grid_dev,
        mm_token_type_ids=mm_tok,
        use_cache=False,
    )
ref_logits = out.logits[0, -1].detach().float().cpu().numpy()
print(f"[torch] prefill logits shape={ref_logits.shape}", flush=True)

# ---- Greedy decode (same prompt) for token-level alignment ----
with torch.no_grad():
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        pixel_values=pv_dev,
        image_grid_thw=grid_dev,
        mm_token_type_ids=mm_tok,
        do_sample=False,
        max_new_tokens=MAX_NEW,
        eos_token_id=eos_ids,
    )
gen_ids = gen[0, input_ids.shape[1] :].tolist()
torch_text = tok.decode(gen_ids, skip_special_tokens=True)

# ---- Timing (warmup + measured) ----
for _ in range(1):
    with torch.no_grad():
        model(
            input_ids=input_ids,
            attention_mask=attn,
            pixel_values=pv_dev,
            image_grid_thw=grid_dev,
            mm_token_type_ids=mm_tok,
            use_cache=False,
        )

torch.cuda.synchronize() if DEVICE == "cuda" else None
t0 = time.perf_counter()
with torch.no_grad():
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        pixel_values=pv_dev,
        image_grid_thw=grid_dev,
        mm_token_type_ids=mm_tok,
        use_cache=False,
    )
prefill_time = time.perf_counter() - t0

torch.cuda.synchronize() if DEVICE == "cuda" else None
t0 = time.perf_counter()
with torch.no_grad():
    gen2 = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        pixel_values=pv_dev,
        image_grid_thw=grid_dev,
        mm_token_type_ids=mm_tok,
        do_sample=False,
        max_new_tokens=MAX_NEW,
        eos_token_id=eos_ids,
    )
gen_time = time.perf_counter() - t0
n_gen = gen2[0, input_ids.shape[1] :].numel()
decode_tps = n_gen / gen_time

# ---- Dump ----
write_array(os.path.join(OUT, "pixel_values.bin"), pv.numpy().astype(np.float32), "<f4")
write_array(os.path.join(OUT, "grid_thw.bin"), grid.cpu().numpy().astype(np.int32), "<i4")
write_array(os.path.join(OUT, "input_ids.bin"), np.array(expanded, np.int32), "<i4")
write_array(os.path.join(OUT, "ref_logits.bin"), ref_logits.astype(np.float32), "<f4")
write_array(os.path.join(OUT, "emb_torch.bin"), emb.astype(np.float32), "<f4")
write_array(os.path.join(OUT, "ref_gen_ids.bin"), np.array(gen_ids, np.int32), "<i4")

with open(os.path.join(OUT, "torch_text.txt"), "w") as f:
    f.write(torch_text)

summary = {
    "device": DEVICE,
    "dtype": str(DTYPE),
    "n_patches": int(P),
    "n_image_tokens": int(n_img_tok),
    "prompt_len": len(expanded),
    "gen_tokens": int(n_gen),
    "prefill_time_s": prefill_time,
    "prefill_tps": len(expanded) / prefill_time,
    "decode_total_s": gen_time,
    "decode_tps": decode_tps,
    "torch_text": torch_text,
}
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n===== PYTORCH REFERENCE =====")
print(f"prompt tokens            : {len(expanded)}")
print(f"generated tokens         : {n_gen}")
print(f"prefill time             : {prefill_time*1000:.1f} ms  ({summary['prefill_tps']:.1f} t/s)")
print(f"decode                   : {decode_tps:.1f} t/s  ({gen_time:.2f} s for {n_gen} tok)")
print(f"vision emb cosine(target): see Rust side")
print(f"\nTEXT:\n{torch_text}")
print(f"\n[dumped to {OUT}/]")
