# DNN_Aggresvation56 日志

## Modify by Claude: 2026-07-01

## 1. 子项目定位（最小验证：协同泄露是否存在）

单字段敏感度方法都饱和/一致,想看**组合维度**有没有单字段法漏掉的东西:是否存在
**"两个字段单看都几乎不泄露,合起来却能把某个 confidential 推出来"** 的字段对(思路 A)。
oracle = 最佳推断模型 gcn_dynamic(R²=0.8746)。

leakage(S) = 只给攻击者字段集合 S(其余 general 置 0)时的推断 R²。
synergy(g1,g2 ; c) = R²_c(only {g1,g2}) − max(R²_c(only g1), R²_c(only g2))。

## 2. 一个方法论教训（先记下来）

**先按 12-conf 加权平均算,结论是"几乎无协同"**(synergy 中位数 0.0001、最大 0.085、
>0.1 的对数=0)。**但这是被平均冲淡的假象**——协同往往只针对**某一个** confidential,
12 个一平均就被稀释。改成**逐 confidential**看,真实协同立刻显现。教训:组合/协同分析
必须逐 confidential,不能先聚合。

## 3. 结果（逐 confidential，协同真实存在）

946 对里:synergy 中位数 0.002(绝大多数对确实无协同),**最大 0.46;>0.2 的 7 对,
>0.3 的 4 对;两个字段单看都 <0.1、合起来 >0.2 的 3 对。** top 案例:

| confidential | field1 (单) | field2 (单) | pair | synergy |
|---|---|---|---:|---:|
| metered_load_mw | forecast_load_latest (0.30) | forecast_load_day_ahead (0.26) | **0.76** | +0.46 |
| da_as_total_mw_thirty | gen_fuel_nuclear (0.19) | system_energy_price (0.20) | 0.55 | +0.34 |
| **da_as_total_mw_synchronized** | da_as_as_req_syn (**0.00**) | da_as_as_req_thirty (**0.03**) | **0.35** | +0.32 |
| **da_as_total_mw_synchronized** | da_as_as_req_pri (**0.00**) | da_as_as_req_thirty (0.03) | 0.34 | +0.31 |
| da_as_total_mw_primary | da_as_as_req_syn (0.20) | da_as_nsr_primary (0.54) | 0.82 | +0.28 |
| **da_as_total_mw_synchronized** | da_as_as_req_pri (**0.00**) | da_as_as_req_syn (**0.00**) | **0.24** | +0.24 |

## 4. 关键发现（第一个正面结果）

1. **协同泄露真实存在,且单字段方法全会漏掉。** 最典型的是**备用容量总量
   `da_as_total_mw_*`**:它的几个"需求分量"(`da_as_as_req_mw_primary/synchronized/thirty`)
   **单看对总量的推断 R²≈0.00**(所有单字段敏感度法都会判它们"不敏感"),但**任意两个
   分量合起来就能把总量重构到 R²=0.24~0.35**。物理上合理——总量是分量的组合,单个分量
   约束不住,两个就够。`metered_load_mw` 由两个负荷预测协同(0.30/0.26→0.76)同理。

2. **这正是单字段归因的系统性盲区**:mask/single/shapley/ig/lrp 都按字段单独打分,把这些
   分量排在末尾;而它们的真实风险是**成组**的。这是相对现有方法(以及 IGNN)的一个**新维度**。

3. **稀疏但确切**:大多数字段对无协同(中位数 0.002),协同集中在少数有"组合=重构"结构的
   confidential(备用容量总量、负荷)。稀疏是好事——是可点名、可演示的具体发现,不是噪声。

## 5. 意义与下一步
- **思路 A（协同/高阶泄露）被证实有料**,可作为突出贡献:*"推断驱动的敏感度对某些
  confidential 必须是高阶的——一组单看无害的字段联合重构目标,单字段归因系统性低估其风险。"*
- **实现无需 walk-LRP**(你的 BN+残差会破坏守恒):二阶遮蔽/充分性即可,黑箱、对 GCN 适用、
  秒级(946 对已算完)。可再用 **Shapley 交互指数**做成有公理保证的版本。
- 顺接**思路 B（反事实最小防护集）**:要阻断 `da_as_total_mw_synchronized` 的泄露,必须**成组**
  隐藏其需求分量(隐藏单个无效)——直接给出可执行防护处方。

## 6. 输出
```
outputs/synergy_perconf.csv / .png   逐 confidential 协同(主结果)
outputs/pair_synergy.csv             逐对(聚合视角，含必要性超可加)
outputs/summary_perconf.json / summary.json
scripts/synergy_perconf.py (主) / synergy_test.py (聚合)
inference_model/model.pt   复用 gcn_dynamic 作 oracle
```
