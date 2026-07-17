# DNN_Aggresvation68 日志

## Modify by Claude: 2026-07-13

## 1. 子项目定位

Part A（主线）：难度分带专家 oracle——按 |S| 把掩码任务空间切带、每带一专家，检验能否
缓解摊销 K0 误差（用户提出，是唯一直接打在"容量稀释"根因上的思路）。
Part B（应用）：二阶精确协同图（重训真值）+ 三阶增量协同扫描认证。

## 2. Part A：难度分带专家 vs uniform（负结果，且推翻容量稀释假设）

**方法**：5 难度带 E1=1–5/E2=6–9/E3=10–13/E4=14–17/E5=18–44，各训专家（`train_banded.py`，
除掩码采样范围外与 63 号协议逐项一致）。3 容量配置：equal（每专家 h256，5× uniform 参数）、
matched（h104，5×≈162k≈uniform 157k）、tilt（中段大两端小，≈150k）。补 85 个中段密集真值
（`gen_band_truth`）+ 60/67 真值合并评测，按 |S| 路由（`eval_banded.py`）。

**结果**（逐 conf MAE）：

| K | uniform | equal(5×容量) | matched | tilt |
|---|---:|---:|---:|---:|
| 0 | 0.053 | 0.052 | **0.046** | 0.051 |
| 10 | 0.039 | 0.041 | 0.038 | 0.043 |
| 50 | **0.024** | 0.026 | 0.027 | 0.031 |

逐带 K0 差异全部 ≤0.01（噪声底 p90≈0.007 附近），matched 略优（0.046 vs 0.053）但不显著；
到 K50 uniform 反而最好。**关键：equal 给每专家满 256 容量（5× 总容量）仍不赢 uniform**——
若是容量稀释，5× 容量该大胜。

**结论（重要）**：分带无显著增益，且 **equal 不赢直接推翻"容量被 2^44 任务稀释"的根因假设**。
摊销 K0 残差是**映射的内在难度**（对任意 S 一次前向逼近 E[Y|X_S] 本身难），非容量不足。
连同 63(GNN)/65(Reptile)/66(hypernet)，这是第 4 个"让 oracle 更聪明"的负结果，四者共同
锁定：**K0 摊销已到内在天花板，唯一有效杠杆是逐子集微调（67 号已证 K=500→Spearman 0.96、
1s/子集）。** 这条"结构/容量/初始化/条件化皆无捷径"的证据链本身是论文的方法学边界结果。

## 3. Part B 二阶：精确协同图（重训真值）

`build_synergy2.py`：syn_c(i,j)=v_c(ij)−max(v_c(i),v_c(j))，逐 (pair,conf)。

**最强协同物理可解释——"燃料 MW + 燃料占比 → 总发电"**（总发电=MW/占比）：
- 风电 MW(0.06)+风电占比(0.16)→total_gen **0.93**，协同 **+0.765**；
- 太阳能 MW(0.006，几乎无用)+占比→0.70，协同 +0.58；水电/多燃料/其他可再生同构。
另有辅助服务类：da_as_req + da_as_nsr → da_as_total 协同 ~0.37。
这是"单看安全、合体泄露"的物理级实证（非统计巧合）。

**对 56 号置零口径的杀伤性对比**：置零 synergy vs 重训真值 Spearman 仅 **0.351**，
置零均值 0.012 vs 重训 0.041——**置零系统性低估协同 3.4×、排序都对不上**。坐实
"必须用重训忠实真值"（IGNN/置零口径会严重扭曲协同结论）。

## 4. Part B 三阶：真不可约三方协同（估计器扫 13244 + 重训认证）

估计器 K200 扫全部 C(44,3)=13244 三元组（`scan_triples`），三阶增量协同
syn3=v(ijk)−max(三个二元子集)，选 top200+random200 重训认证（`certify_triples`）。

**认证结果（重训真值）确认真三阶协同存在**：
- 燃气MW+燃气占比+负荷预测 → 净交换功率 net_actual_interchange：**syn3=0.455**；
- 燃煤/核电/水电/风电 同构，syn3 0.24–0.42。
物理机制：净交换=发电−负荷；发电需 燃料MW+占比（2 字段），负荷需 负荷预测（1 字段），
三者缺一不可（最好二元子集仅 0.24–0.28）——**不可约三方协同**。

**估计器筛选有效且保守**：估计 vs 认证 syn3 Spearman=**0.795**；估计系统性低估（真 0.455 >
估 0.421），故筛选不漏强项。**信噪分离干净**：top 组 314/2364 条 syn3_true>0.10、最大 0.455；
random 组仅 19/2400、中位 0.018（≈噪声）。即"估计器全扫筛 + 重训认证"协议能在指数级
三元组空间里可靠捞出真三阶针，且随机三元组几乎无三阶协同（协同是稀疏结构）。

## 5. 产出

```
scripts/{gen_band_truth,train_banded,train_sched,eval_banded}.py   Part A
scripts/{build_synergy2,scan_triples,analyze_triples}.py           Part B
outputs/expert_*.pt (15)  banded_eval.csv  banded_curve.png
outputs/synergy2_{top_pairs,perconf}.csv  synergy2_heatmap.png
outputs/triples/*.json (13244)  triples_top.csv  triples_certified.csv
outputs/retrain/{bt*,t*}.json  中段真值 85 + 三元组认证 397
```
