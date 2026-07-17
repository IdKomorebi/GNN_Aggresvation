# DNN_Aggresvation43 日志

## Modify by Claude: 2026-06-25

## 1. 子项目定位

DNN42 调参（训练/编码/注意力温度）把 GNN 拉近了 probe，但只动了"训练侧"，**图结构
本身（连几条边、共享 alpha 温度、层数）没动**。本项目在 DNN42 的 combo 参数基础上，
专门扫**图结构参数**。因为 single/multi 趋势一致且 multi 对图结构更敏感、一次跑即可
覆盖 12 字段，**全部在 multi 上做**（4 卡并行）。

锚点 = DNN42 multi combo（mean R² 0.8415）；上界 = DNN probe（0.8937）。

## 2. 单因素图结构扫描（11 组，multi）

| 配置 | 改动 | mean R² | vs 锚点 |
|---|---|---:|---:|
| **g05_ealpha20** | edge_alpha_temperature 0.35→2.0 | 0.8486 | **+0.0071** ★ |
| **g08_layers4** | num_layers 3→4 | 0.8485 | **+0.0069** ★ |
| g09_thr015 | threshold 0.25→0.15（更密） | 0.8456 | +0.0041 ★ |
| g03_topk16 | top_k 8→16 | 0.8438 | +0.0022 |
| g00_baseline | （combo 锚点） | 0.8415 | 0 |
| g01_topk4 | top_k 8→4 | 0.8410 | −0.0005 |
| g10_thr035 | threshold→0.35 | 0.8399 | −0.0016 |
| g04_ealpha10 | edge_alpha_temperature→1.0 | 0.8377 | −0.0039 |
| g06_adaptive | 自适应边 min_k4/max_k16 | 0.8352 | −0.0063 |
| g02_topk12 | top_k 8→12 | 0.8343 | −0.0072 |
| g07_layers2 | num_layers→2 | 0.8131 | −0.0284 |

## 3. 关键发现

1. **top_k（边数上限）几乎不敏感**：4/8/12/16 全在 ±0.007 内（top_k16 微升、top_k12
   微降）。**边的数量不是瓶颈**——图已经连够了，加边不帮、自适应边也没用。这是个
   有价值的否定结论。

2. **edge_alpha_temperature 调高有效——且与 single 结论相反**。DNN42 single 上证明
   edge_alpha 不该动；但 **multi 是 12 个字段共享同一个 alpha_general**，低温（0.35）
   让它塌缩到单一相关性度量，对 multi 更有害。调高到 2.0（让多度量融合、不塌缩）
   +0.007。**这是 single/multi 必须分开调的一个参数。**

3. **num_layers 4 有效——也和 single 相反**。single 上 2/4 层都更差；multi 的
   general-general 多跳传播需要更大感受野，4 层 +0.007（2 层 −0.028 大降）。

4. **图组合超叠加（协同）**：g11 = ealpha_T2.0 + layers4 + threshold0.15 组合后
   **0.8663（+0.0247）**，远超三者单因素之和（≈+0.018）——三个图结构改动有正协同。

## 4. 最终结果：multi 缺口累计缩小 67%

| 阶段 | multi mean R² | 说明 |
|---|---:|---|
| DNN40 baseline | 0.8096 | window=1、旧参数 |
| DNN42 combo | 0.8415 | +训练/编码/注意力温度 |
| **DNN43 graph-combo** | **0.8663** | +图结构（ealpha_T2/layers4/thr0.15） |
| DNN probe 上界 | 0.8937 | |

- **multi–probe 缺口：0.0840 → 0.0521 → 0.0274（累计缩小 67%）**。
- DNN43 multi（0.8663）已基本**追平 DNN42 single combo（0.8688）**——multi 一次训练
  12 字段就达到了 single 逐个训练的水平，更实用。
- 全程**推断仍完全经过图**（无残差 MLP 旁路），保住后续"推断路径→敏感度"方案纯粹性。

详见 `outputs/graph_sweep_multi.png`。

## 5. 结论与下一步

- **推荐 multi 新默认**：在 DNN42 combo 之上，再加 edge_alpha_temperature=2.0、
  num_layers=4、threshold=0.15。
- **single/multi 要分开调的参数**：edge_alpha_temperature（single 低/multi 高）、
  num_layers（single 3/multi 4）。其余 combo 参数通用。
- **剩余缺口（0.027）主要在价格类**（congestion/marginal）。纯参数调到此接近天花板，
  要再逼近需图内架构增强（value MLP / 多头）或残差 MLP 旁路（破坏路径纯粹性，慎用）。
- 下一步可选：① 把 graph-combo 推广回 single 全字段、确认 single 也受益；② 转回
  敏感度主线（DNN41 的 GNN-LRP）；③ 架构增强专攻价格类。

## 5.1 推广回 single（A）

把图增益推广回 single 全 12 字段。注意 edge_alpha_temperature/num_layers 是 single/multi
相反的（DNN42 已证 single 上调高/加深都有害），所以 single 版只加通用的 threshold=0.15：

| 指标 | single combo(DNN42, thr0.25) | single +thr0.15(DNN43) | probe |
|---|---:|---:|---:|
| mean R² | 0.8688 | **0.8725 (+0.0037)** | 0.8937 |
| single–probe 缺口 | 0.0249 | **0.0212** | — |

- 价格类大受益：congestion_rt +0.045（0.662，几乎追平 probe 0.670）、congestion_da +0.023、
  total_losses +0.011；但 **gross_actual −0.034**（更密图对它引入噪声）。
- 结论：single 也受益（+0.004），趋势与 multi 一致（价格类最吃图结构）；但 threshold 是
  **per-field 权衡**（价格类爱更密、gross_actual 不爱），如需极致可 per-field 设 threshold。
- 详见 `outputs/single_graphcombo.png`。

**两条线最终状态**：single 0.842→0.873（缺口 0.021）、multi 0.810→0.866（缺口 0.027），
均把与 probe 的缺口砍掉约 65~67%，且推断全程经过图。

## 6. 输出目录结构

```
configs/                              # g00~g11 multi 图结构扫描配置
outputs/
├── tuning/<config>/<timestamp>/      # 各 run
├── _probe_reference/                 # probe 上界参照
├── relationship_cache/               # 相关性缓存（复用）
└── graph_sweep_multi.png             # 扫描对比图
```
