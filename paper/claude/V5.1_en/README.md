# V5.1_en：V5_en 的数据修正版（固定新估计器，下游全部重算）

**《From Set-Level Inferability to Field-Level Risk: Budgeted Marginal Inference and an Amortized Attacker for Grading Power Market Data》**

编译稿：`main.pdf`（43 页，elsarticle 预印本单栏格式）。

## 与 V5_en 的区别（小版本：只改估计器及其下游数字，叙述结构不变）

| 项目 | V5_en | **V5.1_en** |
| --- | --- | --- |
| 估计器 | 只预测目标 × 3 种子（旧口径 E）；重建版本只在估计器一节单独报告 | **重建预训练 ⊕ 未训练网络特征拼接 × 3 种子 + 三道防护**（125 号按事先写定的规则选定） |
| 方法节 | 式 (pretrain) 用 γ 区分两种目标 | γ=1 为本文估计器；读出特征 Z_r = [x_S, x_S², φ_r, φ̃_r]；示意图同步更新 |
| 保真度 | V 0.026、M 0.021、认证 0.934/0.970 | **V 0.018、M 0.015、认证 0.961/0.978** |
| Γ 与背景 | Γ 误差 0.017，召回 47/67/74%，认证前 3 恢复 92.8% | **0.014，55/74/79%，94.8%** |
| 决策（应用节） | 扫描—认证召回 75–88%；只凭估计值扣留残留 1.3–5.8% | **76–91%；0.3–1.6%**（多扣留最多 1.9 个字段） |
| 排序 | R-精确率 0.903 | 0.894（仍最高，与精确 M^(2) 相同） |
| 消融表 | 118 号 | 新增底部一块"预训练目标与特征映射"（125 号），中间块注明针对只预测目标的主干 |
| 成本 | 62–78 ms，191–763 倍 | **48–64 ms，241–906 倍**（126 号统一计时、等价快速读出；消融表新增每集合耗时列；成本段写明各方法耗时） |

数据来源的变化：决策/排序表改读 `DNN_Aggresvation125/outputs/d117/`，估计器保真度图改读 `DNN_Aggresvation125/outputs/final_est/`，
消融表与遮蔽诊断表底部读 `DNN_Aggresvation125/outputs/select/`，成本与消融表耗时列读 `DNN_Aggresvation126/outputs/time_fast_*.csv`、`pretrain_time.csv`。

## 与 V4_en 的区别（按用户 2026-09-25 的三点意见）

| 维度 | V4_en | **V5_en** |
| --- | --- | --- |
| 主线 | 字段风险 → 越来越像"发布优化"（扣留、命中集在正文占重） | **背景依赖的字段风险**：边际增益 Δ_i(T) 是一个分布；M^(K) 取上包络；**背景放大量 Γ^(K)=M^(K)−M^(0)** 与危险背景族是字段画像的输出 |
| 理论 | 安全余量、关键性⇔小 MUS | 保留两条定理；新增 **K=1 时 Γ 的闭式**（=最强正两两交互）；"为什么取最大而非平均"（Shapley/SAGE 平均 vs 安全最坏情况） |
| 机理例子 | 合成数学小例子（正文） | **受控电力机制算例（123 号）**：RTS-GMLC 上机组 313_CC_1 的私有报价加成为目标，价格字段单独为 0、以出力为背景时 0.10–0.15，排序与运行状态依赖（C6 阻塞）完全符合调度物理；合成例子移入附录 C |
| 真实数据 | 组合风险表 | 组合风险表 + **背景放大画像（124 号）**：Γ¹/Γ²/Γ³=0.082/0.110/0.119，平均增益仅为最大值的 1/3，91 对"单字段不显眼、背景下放大" |
| 估计器 | 随机初始化主干 ≈ 预训练主干（只陈述） | **122 号诊断与修复**：只预测目标的预训练使表征塌缩；小集合上随机特征本就够用；M 取最大对读出爆掉极敏感；重建 + 预测截断在全部估计指标上优于 E |
| 处置 / 扣留 | 正文结果 + K=3 用于处置的建议 | 降为 **应用一节（第 12 节）**，并写明阈值敏感（单字段定级漏掉的危险组合在 τ∈[0.5,0.9] 上从 96% 到 38%）与 K 敏感 |
| K 的选择 | "K=2 定级、K=3 处置" | K=2 足以描述和排序；涉及阈值的结论要带上预算说明 |

## 结构

```
V5_en/
├── main.tex
├── sec_abstract.tex
├── sec_intro.tex          # 1 引言（贡献改为：背景感知风险 / 摊销攻击者 + 预训练诊断 / 机制与真实数据）
├── sec_front.tex          # 2 相关工作  3 问题形式化  4 字段级风险（Δ、M、Γ、K=1 闭式、为什么取最大、安全含义、K 的选择）
│                          # 5 摊销攻击者（式 (pretrain) 带 γ：γ=0 为主口径 E，γ=1 为重建；三道数值防护）
├── sec_setup.tex          # 6 实验设置（含受控机制数据集与预训练变体）
├── sec_mech.tex           # 7 Results I：受控电力机制（123 号）
├── sec_amp.tex            # 8 Results II：真实数据中的组合风险与背景放大（111/112/116/124 号）
├── sec_def.tex            # 9 Results III：字段风险打分的排序对比（117 号）
├── sec_res2.tex           # 10 Results IV：摊销攻击者（保真度、Γ 与背景召回、消融、预训练诊断与修复、主干复用、成本）
├── sec_res3.tex           # 11 Results V：稳健性与 K
├── sec_application.tex    # 12 应用：关键字段、扣留、局限
├── sec_discussion.tex     # 13 讨论与局限  14 结论
├── sec_appendix.tex       # 附录：证明、实现细节、字段表、合成机理例子
└── scripts/ tables/ figures/
```

## 图表与数据来源（编号为编译稿中的编号）

| 编译稿 | 文件 | 内容 | 数据来源 |
| --- | --- | --- | --- |
| 图 1 | fig1_framework | 审计框架 | 示意图 |
| 图 2 | fig3_model | 摊销攻击者与扫描—认证 | 示意图 |
| 图 3 / 表 3 | fig_mechanism_narrow / tab:mech（正文手写） | 受控机制算例（±5% 加成） | 123 号 `outputs/narrow/analysis/*.csv` |
| 表 4 / 图 4 | tab_risk / fig4_combination_risk | 单字段定级漏掉的组合风险 | 111、112、116 号 |
| 图 5 | fig5_profile | 字段画像与增益矩阵 | 111、112 号 |
| 图 6 | fig_amplification | 背景放大图与边际增益分布 | 124 号 `profile.csv`、`context_gain_distribution.csv` |
| 表 5 | tab_ranking | 排序对比 | 125 号 `outputs/d117/` |
| 图 7 | fig7_estimator | 估计器保真度 | 125 号 `outputs/final_est/` + 各号真值 |
| 表 6 / 图 8 | tab_ablation / fig8_ablation | 估计器对比与消融 | 118 号 + 125 号 `outputs/select/` |
| 表 7 / 图 9 | tab_masking / fig_masking | 掩码预训练诊断与估计器设计 | 122 号 + 125 号 `variants_by_target.csv`、`error_decomposition.csv`、`rank_truncation.csv`、`large_sets.csv`、`decisions.csv` |
| 表 8 | tab_backbone | 全列主干复用 | 116 号 |
| 图 10 | fig9_robustness | 稳健性与 K | 119 号 |
| 表 9 | tab_decision | 关键字段与扣留（应用节） | 125 号 `outputs/d117/`（新估计器重算） |
| 图 C.11 | fig2_mechanism | 合成机理例子（附录） | 117 号 |
| — | fig_mechanism_wide | ±30% 加成对照（未进正文） | 123 号 `outputs/wide/` |

改动实验数字后：`cd scripts && python make_tables.py [名] && python make_figs.py [名]`，再 `tectonic main.tex`。

## 写作口径说明

- 全文估计器统一为 125 号选定的新估计器；只预测目标的预训练作为对照与诊断对象出现。
- 选择规则在运行最后一个候选前写定（见 125 号 base.yaml），正文如实说明；δ=0.05 下召回略低于旧口径、耗时约 2.5 倍，均已写明。
- IGNN 只陈述其不足，不写"互补"；PJM 负荷预测作为目标副本排除；不涉及时间划分；只与串行重训比较成本。
