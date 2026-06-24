# DNN_Aggresvation33 走读报告

## 子项目定位

DNN33 验证用户提出的**双重注意力机制**（multi-head + target-conditioned
head gate）能否缩小 multi 模式与 single 模式的 R² 差距。

---

## 改动

在 DNN32 `model.py` 的 `_confidential_messages` 中：
1. q/k/v 切成 `n_heads` 份，每 head 独立 gated attention
2. `head_weight = softmax(W_head · target_identity / τ)` 加权合并各 head
3. 投影回 hidden_dim

`n_heads=1` 退化为 DNN32 单头行为。

---

## 实验设置

3 个 multi 模型，shuffle split + bipartite + nostaged：
- `baseline_1head`：n_heads=1
- `dual_4head`：n_heads=4
- `dual_8head`：n_heads=8

对照 DNN32 的 12 个 single 模型。

---

## 核心结果

| 指标 | 1head | 4head | 8head | single |
|---|---:|---:|---:|---:|
| best_test_loss | 0.223 | 0.230 | 0.238 | — |
| 8h vs 1h median Δ | — | — | -0.003 | — |
| 8h vs single median Δ | — | — | -0.032 | — |

**双重注意力没有提升**：8head 比 1head 平均降 0.014，仍比 single 低 0.053。
head 越多越差（1head > 4head > 8head）。

---

## 原因分析

1. `head_dim=64/8=8` 太小，单 head 表达能力不足
2. head gate 用 target_identity，但 target_identity 梯度信号弱，gate
   趋向均匀，没形成"专属通道"分化
3. **根本原因**：multi 的瓶颈在 General-GCN backbone 共享表示，不在
   attention 层。head gate 只能在消息传递层做软选择，无法改变 backbone
   已被 12 个 target 平均化的事实

---

## 结论

双重注意力未能缩小 multi-single 差距。**single 模式仍是最优**。
multi 评估用 baseline 1head 即可。
