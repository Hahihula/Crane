# Qwen3.5-2B 多模态（VL）：Crane vs PyTorch 对齐性与速度测试

测试环境：**Apple M1 Pro (10 核, 16GB)，macOS Metal**。
模型：`checkpoints/Qwen3.5-2B`（`Qwen3_5ForConditionalGeneration`，含视觉塔）。
双方均使用 **f16**，同一台机器，同一份 checkpoint。
参考实现为 PyTorch + HuggingFace Transformers (`Qwen3_5ForConditionalGeneration`, MPS)。

## 1. 对齐性结论：✅ 完全对齐

把 **完全相同的输入** 同时喂给 PyTorch 与 Crane：

- PyTorch 侧：用 `processor` 算出 `pixel_values`、`image_grid_thw`，并复刻 Crane 的
  `render_prompt` 得到逐 token 对齐的 `input_ids`；把这些 dump 出来。
- Crane 侧：把 PyTorch 算出的 `pixel_values` / `image_grid_thw` / `input_ids` 直接注入
  `Qwen3_5VLModel::forward` / `encode_images`，保证输入逐位一致，只比对模型数学实现。

| 检查项 | 指标 | 结果 |
|---|---|---|
| 视觉塔（同像素输入） | 投影 embedding 平均余弦 | **0.99906** |
| 视觉塔 | 整体余弦 / 最大绝对误差 | 0.99791 / 1.41 |
| Prefill 末位 logits | 余弦相似度 | **0.99876** |
| Prefill 末位 argmax | top-1 是否一致 | ✅ 完全一致 (`248068`) |
| Prefill 末位 | top-5 重叠 | 5/5 |
| 贪心解码 | 最长公共前缀 | **96 / 96 tokens** |
| 贪心解码 | token 级准确率 | **1.0000** |

Crane 跑出的 token 序列与 PyTorch **逐 token 完全相同**（包括模型自发产生的
`<think> ... </think>` 块）。视觉塔 embedding 虽然个别 patch 行余弦最低到 0.886
（单个 patch 的数值边角，最大绝对误差 1.41），但下游 token 输出完全一致，说明
ViT + 2×2 空间合并 + 3D-MRoPE splice + 混合 GDN/softmax 解码器整条多模态链路实现正确。

## 2. 速度对比（同一机器，Metal vs MPS，均 f16）

| 阶段 | Crane (Metal) | PyTorch (MPS) | 比值 |
|---|---|---|---|
| Prefill（642 tok，含视觉塔） | 34.6 t/s | 149.9 t/s | Crane **慢 ~4.3×** |
| └ 其中视觉塔 | 3.7 s（占 prefill 20%）| — | — |
| Decode（96 tok） | 19.2 t/s | 5.9 t/s | Crane **快 3.3×** |

- **Decode：Crane 快 3.3×**。单 token 步进下 Crane 路径更高效，而 PyTorch MPS 的
  GDN decode 非常慢（2B 模型仅 5.9 t/s）。
- **Prefill：Crane 慢 ~4.3×**（仅限 Metal）。根因是 **GDN（混合线性注意力）的融合
  递归 kernel 目前只在 CUDA/ROCm 上启用**：
  `crane-core/src/ops/gdn/backend.rs:239` 仅 `q.device().is_cuda() || q.device().is_rocm()`
  且未设 `CRANE_GDN_PORTABLE` 时走融合 kernel。在 Metal 上回退到 op-by-op 的可移植实现，
  prefill 这种长序列被大量小算子 launch 拖慢；PyTorch MPS 自带较优的 GDN 实现，因此反超。
  这不是数值/实现错误，而是 Metal 上缺少 GDN 融合 kernel 的结构性限制。

## 3. 总论

1. **对齐：完全一致。** 多模态 Qwen3.5-2B 在 Crane 上的视觉塔 embedding（余弦 0.999）、
   prefill logits（余弦 0.999，top-1 精确命中）、贪心解码（96/96 token 逐位相同，
   准确率 1.000）都与 PyTorch / HuggingFace Transformers 对齐。加入多模态
   （ViT + 2×2 空间合并 + 3D-MRoPE splice）后数学实现是正确的。
2. **速度：decode 显著更快，prefill 在 Metal 上更慢。** 这是 Metal 上缺少 GDN 融合
   kernel 导致的；一旦该 kernel 移植到 Metal（或走 CUDA/ROCm），prefill 也会像 README
   中其它模型一样反超 PyTorch。

## 复现方式

- `tests/align_qwen35_vl_reference.py` —— PyTorch 参考：dump 对齐输入 / embedding /
  logits / tokens + 计时。
- `example/src/align_qwen35_vl.rs` —— Crane 对齐 + 计时二进制。
- 构建与运行：
  ```bash
  python3 tests/align_qwen35_vl_reference.py
  cargo build --release --features metal,accelerate --bin align_qwen35_vl -p crane-examples
  ./target/release/align_qwen35_vl checkpoints/Qwen3.5-2B tests/align_out
  ```
- 中间产物位于 `tests/align_out/`。

> 注：对齐测试基于单张图（餐厅小票）+ 一个 prompt、贪心 96 token；token 级完全匹配已是很强的证据。
> 后续 Metal 加速的重点：**为 GDN 在 Metal 上实现融合递归 kernel**（或等价地 fusion 掉
> op-by-op 的小算子），以补齐 prefill 的 4× 差距。
