# DNN_Aggresvation55 日志

## Modify by Claude: 2026-07-01

## 1. 子项目定位

验证一种 GCN/GAT 混合:**分裂设计不变——general→general 用 GCN(+静态先验),
只把 general→confidential 那半边的注意力从 transformer q·k 换成 DNN53 的经典加性 GAT
(K=V=Wh, `e_ij=LeakyReLU(a_l^T z_i + a_r^T z_j)`)**,看是否更好。

2×2:{conf 注意力 = transformer / classic} × {先验混合 = gated 动态 / no_gate 静态}。
G-G 始终 GCN+静态先验、mlp 编码、seed 42。代码:`src/model.py` 加 `conf_gat_variant`,
`_confidential_messages` 支持经典加性 GAT;`scripts/train_split.py`。

## 2. 结果(12-conf 平均 test R²)

| G→C 注意力 | 先验混合 | 平均 R² |
|---|---|---:|
| **transformer** | gated | **0.8663**(复现当前分裂基准) |
| classic | gated | 0.8575 |
| transformer | no_gate | 0.8522 |
| classic | no_gate | 0.8516 |

参照:当前分裂基准 0.8663、最佳 gcn_dynamic 0.8746、DNN probe 上界 0.8937。

## 3. 关键对比

- **同一先验混合下,经典 GAT 在 confidential 侧更差**:
  - gated:classic 0.8575 − transformer 0.8663 = **−0.0088**
  - no_gate:classic 0.8516 − transformer 0.8522 = −0.0006
- **本子项目最佳(transformer_gated 0.8663)仍低于 gcn_dynamic 0.8746(−0.0083)。**
- classic 的劣势又**集中在难字段**:congestion_price_rt −0.048、marginal_loss_price
  −0.038、congestion_price_da −0.024;易字段(R²>0.9)基本打平(+0.0006);
  难字段(R²<0.8)平均 −0.0216。

## 4. 结论

**这个混合(G-G GCN + G→C 经典 GAT)不更好,反而略差。**

- 与 DNN53 一致:在**门控动态先验**这种模式下,transformer 的独立 Q/K/V 给 gate 提供
  更可分解的信号,比经典加性 GAT 好;经典 GAT 的优势(不崩、裸注意力更稳)只在
  "整张图都做注意力"时才显现(DNN53 的 general-general 上),而这里 general-general
  已经交给 GCN,confidential 侧的门控又恰恰是 classic 的弱项。
- 再次印证贯穿 DNN52–55 的主线:**任何注意力变体、任何摆放位置(整图/分裂、经典/
  transformer)都没能超过简单的 gcn_dynamic(动态先验门控、无 q·k 注意力)。**
  confidential 侧的注意力形式不是提升准确率的杠杆。

## 5. 局限
单 seed。−0.0088(gated)幅度明确、且集中在难字段、与 DNN53 同向,基本不受 seed 影响;
no_gate 下 −0.0006 属打平。

## 6. 输出
```
outputs/<conf_variant>_<arch>/{model.pt, per_target_r2.csv, summary.json}   4 变体
```
