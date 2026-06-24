# DNN_Aggresvation31 走读报告

## 子项目定位

DNN31 在 DNN30 shuffle split 基础上做 2×2 因子实验，回答：
1. staged training 在 shuffle split 下是否还有必要
2. 加入 net/gross 到 loss 是否损害其它字段（多目标联合推断是否有副作用）

---

## 实验设置

4 个对照，全部 bipartite=True + shuffle split：

| | maskloss | allloss |
|---|---|---|
| staged | staged_maskloss | staged_allloss |
| nostaged | nostaged_maskloss | nostaged_allloss |

---

## 核心结果

### staged vs nostaged（allloss）

| 字段 | staged | nostaged | Δ |
|---|---:|---:|---:|
| net_actual_interchange_mw | 0.65 | **0.74** | **+0.097** |
| gross_actual_interchange_mw | 0.71 | 0.73 | +0.016 |
| 其余 10 字段 | 持平/微升 | 持平/微升 | — |
| **best_test_loss** | 0.2363 | **0.2250** | **-4.8%** |

**staged 应取消**：它为时序 split 的分布漂移设计，shuffle split 下
完全多余，且让 net/gross 少了 100 epoch 训练量，R² 降 0.097。

### maskloss vs allloss（nostaged）

| 指标 | maskloss | allloss | Δ |
|---|---:|---:|---:|
| 10 个稳定字段均值 R² | 0.7920 | 0.7972 | **+0.005** |
| net_actual R² | -0.001 | 0.7427 | — |
| gross_actual R² | -0.000 | 0.7262 | — |

**加入 net/gross 不损害其它字段**：10 个稳定字段均值微升 0.005。
多目标联合推断不会降低单个推断质量。

### 2×2 汇总

| | maskloss | allloss |
|---|---:|---:|
| staged | 0.2208 | 0.2363 |
| nostaged | 0.2208 | **0.2250** |

---

## 结论

1. **取消 staged training** — shuffle split 下多余且对 net/gross 有害
2. **allloss 无副作用** — 多目标联合推断不降低单个推断质量
3. **推荐默认配置**：bipartite + shuffle split + nostaged + allloss
