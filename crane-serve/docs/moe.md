# Mixture-of-Experts (MoE) Models

Qwen3-Coder-30B-A3B (and other `qwen3moe` GGUF checkpoints) is a
30-billion parameter model — normally far too big to fit on a 16 GB or
24 GB GPU. It's built as a "Mixture-of-Experts" (MoE) model, meaning that
generating any one token only ever uses a small slice of those 30 billion
parameters, not all of them at once. Because of that, the model doesn't
need to live in GPU memory in its entirety — Crane can keep part of it in
regular CPU RAM instead, and only pay a speed penalty on the (less common)
tokens that need a part that's on CPU rather than GPU.

At startup, Crane decides once which parts of the model to keep on the
GPU: it loads the whole model to CPU first, then copies over as much as
fits within `--gpu-memory-limit` (see the startup log lines starting with
`Expert placement:` and `Live VRAM ...` to see how much made it onto the
GPU). This split is fixed for as long as the server keeps running. The
more of the model that made it onto the GPU, the faster generation runs
overall; whatever's left on CPU still produces correct output, just more
slowly. If you already know your GPU has no room to spare,
`--offload-experts` skips that startup measurement and keeps everything
on CPU, saving a little startup time.

For long-context tools like opencode or aider that keep feeding a growing
conversation back into the model, two things compete for the same GPU
memory: how much of the model got kept on the GPU at startup (decided
once, and fixed for the life of the server), and how much room is left
for the KV cache — the per-conversation memory that grows as your prompt
and its response do.

**A large `--context` reserves more room for the KV cache up front**,
which can mean less of the model (or none of it) gets kept on the GPU —
the model still works correctly, it just runs more of its computation on
CPU instead, which is slower but not unsafe.

There's also a real limitation worth knowing: once the server is running,
it can only free up VRAM by evicting a request when a *new* request needs
to start and there isn't room — it can't shrink a request that's already
running on its own. If you're running one long-lived session
(`--max-concurrent 1`, matching how most coding assistants use the API),
nothing else is competing for room, so nothing ever gets evicted to make
space. In that case, **`--context` is your only real safety net** against
running out of VRAM mid-session, and it needs to be sized to fit your
actual hardware — not just picked to match how much context you'd like to
have.

## Worked example: Qwen3-Coder-30B-A3B GGUF on a 16 GB card

```bash
crane-serve --model-path /path/to/Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf \
    --format gguf \
    --gpu-memory-limit 12G \
    --max-concurrent 2 \
    --context 128K
```

- `--gpu-memory-limit 12G` leaves 4 GB of the 16 GB card free for your
  desktop and other programs. Don't set this close to the card's full
  capacity — anything else using the same GPU can crash the graphics driver.
- `--max-concurrent 2` lets two requests decode at the same time (e.g. two
  overlapping tool calls from one coding assistant, or two clients). Set
  it to `1` if you only ever run one conversation at a time — that frees
  up a little more room for context.
- `--context 128K` requests the model's full native context window
  (131,072 tokens), which is easier to read and reason about than
  computing a raw token count for `--max-seq-len` by hand. The two flags
  do the same thing; use whichever is more convenient. `128K` = 131,072
  tokens (`K` means x1024, not x1000).

If your GPU has less than 16 GB free, or you need more headroom for other
programs, lower `--context` (e.g. `64K` or `32K`) and/or `--gpu-memory-limit`
first — most of this model's weights already live on CPU, so it's the
context window and concurrency settings that actually compete for the GPU
budget you give it.

Check the startup log's `Expert placement: N/48 layers on <device>, M on
CPU` and `Live VRAM ... available_for_experts=...` lines to see what your
settings actually produced — the more of those 48 layers land on GPU, the
faster generation runs.

If you leave `--context` unset entirely while `--gpu-memory-limit` is set,
Crane computes a safe value for you automatically from measured VRAM
headroom — see [Auto-derived context
length](gpu.md#auto-derived-context-length) — but starting from a
known-working combination like the one above is the simplest path if you
just want Qwen3-Coder running.
