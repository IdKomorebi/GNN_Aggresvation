# DNN_Aggresvation44 日志

## Modify by Claude: 2026-06-25

## 1. 子项目定位

DNN40 系列（敏感度方向）一直用基础版单头 GAT，从没用过多头。DNN34 当年在
window=4 + 旧参数下测过 dynamic head attention（结论：multi 上小幅提升）。本项目
在**当前最优基线**（DNN43 multi graph-combo，单头 mean R² 0.8663）上重测多头，
看 window=1 + combo + graph-combo 之后多头是否变得有用。

做法：把 DNN34 的多头 model（dynamic head attention）移植到 DNN40 的（已修正信息集）
pipeline，扫 n_heads ∈ {1,4,8}，head_aggregation=dynamic_attention，其余沿用 g11
（ealpha_T2、layers4、thr0.15、mlp、attn_T0.25）。全部 multi。

## 2. 结果：多头无帮助，反而拖累

| 配置 | multi mean R² | vs 单头基线 |
|---|---:|---:|
| **单头 graph-combo (DNN43)** | **0.8663** | —（基线） |
| 多头 n=1 dynamic | 0.8546 | **−0.0117** |
| 多头 n=4 dynamic | 0.8621 | −0.0042 |
| 多头 n=8 dynamic | 0.8628 | −0.0035 |
| probe 上界 | 0.8937 | |

详见 `outputs/multihead_comparison.png`。

## 3. 分析

1. **多头框架整体不如纯单头**：即使 n_heads=1（理应退化为单头），DNN34 的多头框架
   也比 DNN43 的纯单头 GAT 低 0.012。原因：DNN34 的 `_confidential_messages` 即使
   单头也走多头逻辑（query 多了 target_identity 项、head 聚合层等额外结构），在数据
   干净 + 充分调参后这些反而是多余/次优的；DNN43 的简洁单头 GAT 已被调到更好。

2. **多头框架内部 n_heads 越多越好**（0.855→0.862→0.863），与 DNN34 当年一致——
   加 head 有小幅帮助，但补不回框架本身相对纯单头的损失（n=8 仍 −0.0035）。

3. **结论：多头不值得用**。窗口=1 + 充分调参后，单头 graph-combo 已经很强（缺口
   0.027），多头框架只会拖累。这与 DNN34 "多头收益有限" 的结论一致，且在新基线上
   更明确——多头这条路在本项目上确认无效。

## 4. 收尾结论（DNN42→44 推断精度提升收官）

- **single**：0.842 → 0.873（缺口 0.021）
- **multi**：0.810 → 0.866（缺口 0.027，= 单头 graph-combo，已追平 single combo）
- 有效手段：mlp 编码 + 充分训练 + attention 低温（combo）；multi 再加 edge_alpha 高温
  + layers4 + thr0.15（graph-combo）。
- 无效手段（已排除）：top_k/边数、自适应边、残差以外的容量加宽、**多头注意力**。
- 推断精度提升到此接近天花板（纯结构/参数层面）。剩余缺口在价格类，若要再逼近只剩
  图内非线性增强或残差 MLP 旁路（破坏路径纯粹性）。**建议转回敏感度主线（DNN41）。**

## 5. 输出
```
configs/mh{1,4,8}_dynamic.yaml
outputs/tuning/mh{1,4,8}_dynamic/<ts>/
outputs/multihead_comparison.png
```
