# DNN_Aggresvation34 走读报告

## 子项目定位

DNN33 的「双重注意力」没提升，并把锅甩给 backbone。但 DNN33 的实现里有两个
缺陷纠缠：**切维度**（每头太小）+ **静态 head gate**（第二级权重与输入无关）。
DNN34 把它们拆开：统一用完整维度修掉缺陷1，然后对照验证缺陷2——
第二级到底该用 GLM 的**静态 gate** 还是用户主张的**动态注意力**。

---

## 改动

`src/model.py` 新增 `head_aggregation` 开关，第二级 head 聚合二选一：

- `static_gate`（GLM）：`head_weight = softmax(W·target_identity/τ)`，静态。
- `dynamic_attention`（用户）：对 head 维度再做注意力——query = target 身份 +
  当前 conf 隐藏态，key = 各 head 的实际输出 `msg_h`，权重随输入动态变化。

`n_heads=1` 时两者等价。

---

## 实验设置

5 个 run（shuffle + bipartite + nostaged + allloss，与 DNN32/33 同口径）：
`baseline_1head` / `static_{4,8}head` / `dynamic_{4,8}head`。

---

## 核心结果

| 变体 | best_test_loss |
|---|---:|
| baseline_1head | 0.2219 |
| static_8head (GLM) | 0.2314 |
| **dynamic_8head (用户)** | **0.2186** ← 全场最优 |

| 对比 | mean Δ R² | 胜出字段 |
|---|---:|---:|
| 8head dynamic − static | **+0.0120** | **12/12 全胜** |
| 8head dynamic − baseline | +0.0030 | 8/12 |
| 4head dynamic − static | +0.0081 | 5/12（混合） |

最难字段 `congestion_price_da`（probe 0.048）：动态 4head 0.376 vs 静态 0.273，
**+0.10**。困难字段收益最大。

---

## 结论

- **用户思路更合理**：8head 上动态注意力 12/12 全胜静态 gate；相对 DNN33
  「head 越多越差」的趋势逆转，dynamic_8head 首次跑赢单头基线。
- **GLM 也对一半**：不切维度、用完整 attention_dim 的判断成立。两人合起来
  才让多头出现正收益。
- **但提升很小**，distance to single 仍在——DNN33「瓶颈在共享 backbone」
  的根本结论没被推翻。本项目把 attention 层内部做到了最优而已。
- 实用：multi 整体评估用 `dynamic_8head`；追极限精度仍选 single。

详见 [CHANGELOG.md](CHANGELOG.md)、对比表 `outputs/dynamic_vs_static_comparison.csv`。
