# DNN_Aggresvation73_Claude_InterimSummary 日志

## Modify by Claude: 2026-07-22

## 1. 子项目定位

这不是一个新实验，而是一次**全项目复盘 + 口径核查 + 改进方案清单**。

触发：用户在 73 号完成后提出九个问题（图结构、随机失活、校准、微调、二阶协同、
高阶协同、残差解读、最小防护集、整体改进），要求把现有方法与效果讲清楚、
指出可改进处。本子项目：

1. 通读 56–73 号的 CHANGELOG、关键脚本（`oracle.py` / `correlation.py` / `model.py` /
   `protection_greedy.py` / `direct_greedy.py` / `calibration_and_hierarchy.py`）与
   `outputs/` 原始 CSV；
2. 就其中四个**口径存疑**的地方用现成真值当场重算（不重训、不训练、纯 CPU 分析）；
3. 输出一份带术语解释的中期总结 `INTERIM_SUMMARY.md`，含 41 条可执行改进建议
   与优先级排序。

**本子项目不修改任何既往子项目的文件。** 发现的问题以"更正建议"形式记录在
`INTERIM_SUMMARY.md` 第 3 章，是否回写由用户决定。

## 2. 做了什么（四个核查，全部只读现成真值）

| 脚本 | 核查什么 | 一句话结论 |
|---|---|---|
| `verify_synergy_blindness.py` | K=0 oracle 对二阶协同的检出能力（分层 + top-N 排名） | **K=0 对最强协同结构性失明**：真值 top10 的估计排名中位数 7216/11352，估计值多为负 |
| `verify_perconf_vs_average.py` | 逐 conf vs 12-conf 平均，量化差多少 | 阈值 0.1 时平均口径**漏报 383/402 对（95%）**；56% 的强协同只对唯一一个 conf 成立 |
| `verify_apriori_selection_bias.py` | 70 号 Apriori "85%" 的口径与选择偏差 | 85% 是 **recall 不是 Spearman**，且算在估计器选出的 1.49% 池子里；无偏随机池上 recall=0.947 (18/19)、加速 4.8× |
| `verify_residual_ceiling.py` | 70 号"残差 9–16 达峰后回落"是否真实 | **是天花板假象**。逐 conf 天花板归一化后残差单调上升 0.096→0.214，闭合率 0.250→0.913 |

## 3. 四个核查里两个属于"需要更正原结论"

- **70 号第 3 节**"高阶/累积泄露是补足项而非主导项、残差有界不发散" → 应改为
  "高阶泄露随规模持续增强，大集合里九成的低阶-天花板缺口由高阶填补；绝对残差的
  回落是 R² 上限压扁所致"。70 号的核心主张（排序低阶可决定，m3 校准后 Spearman
  0.969）**不受影响**。
- **70 号第 4 节** Apriori 的 85%/2.4× 应换成无偏口径 94.7% [0.75, 0.99] / 4.8×，
  并注明 n=19 太小。

另有两处不是本次核查所得、但通读时发现的表述滞后，一并记在 `INTERIM_SUMMARY.md`：

- **72 号第 3 节**"纯 oracle 贪心必然过拟合代理（Goodhart）"已被
  `DNN_Aggresvation72-71Supplement` 推翻（那是 GNN 代理太差所致），但 72 号
  CHANGELOG 仍是旧版；
- **71/72 的"协同图"实际是"危险对图"**（边定义用的是 `max_c v_c({i,j})>τ` 即泄露，
  不是 `syn_c(i,j)` 即协同），术语需要统一。

## 4. 产出

```
INTERIM_SUMMARY.md                       主文档（术语表 + 全景 + 核查 + 九问 + 路线）
scripts/verify_synergy_blindness.py      核查 A
scripts/verify_perconf_vs_average.py     核查 B
scripts/verify_apriori_selection_bias.py 核查 C
scripts/verify_residual_ceiling.py       核查 D
outputs/q5_synergy_blindness_{bystrength,toprank}.csv + q5_synergy_top10_detail.csv + .txt
outputs/q5b_{perconf_vs_average,nconf_distribution}.csv + .txt
outputs/q6_apriori_selection_bias.csv + .txt
outputs/q7_{residual_ceiling,conf_ceilings}.csv + .txt
```

复现：`python scripts/<脚本名>.py`，无需 GPU，全部秒级。
依赖的上游文件均为只读引用（69/68/70 号的 `outputs/`），路径在脚本头部常量里。

## 5. 下一步

`INTERIM_SUMMARY.md` 第 6 章给了三档优先级共 16 个候选实验。**用户尚未决定从哪里
开始**，本子项目到此为止，不预先启动任何实验。
