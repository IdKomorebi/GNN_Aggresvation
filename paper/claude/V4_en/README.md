# V4_en：英文 SCI 稿（故事线 v3 + 110–119 号实验）

**《From Set-Level Inferability to Field-Level Risk: Budgeted Marginal Inference and an Amortized Attacker for Grading Power Market Data》**

## 与 V3_en 的区别

| 维度 | V3_en | **V4_en** |
| --- | --- | --- |
| 故事 | 发布审查；分段预算是主角 | 三支柱：集合推断能力（真值）→ 字段级风险 M^(K)（定义与安全语义）→ 摊销攻击者 + 扫描—认证（可算、可信） |
| 理论 | 三定理（截断超有效性在正文） | 正文两条：安全余量、关键性⇔小 MUS；截断超有效性与分段预算移入附录 |
| 数据 | PJM/CAISO 41 字段 × 12 目标（多数目标实际公开） | 四类数据、按披露规则定角色：RTS-GMLC、NEM 2024（真实机组出力）、PJM、CAISO（次日披露的实际量为目标，候选补全） |
| 对比 | 单字段、归因 | 相关类、单字段、LOCO/置换/SAGE、仿 IGNN 推断图传播；决策、处置、排序三层评测 + 机理小例子 |
| 估计器 | 三种子拼接（有数值缺陷） | 新主口径 E（数值修正 + 5 折 + 三种子预测平均）；全列主干复用；完整消融 |
| 稳健性 | — | 真值种子、随机划分、τ 扫描、各数据集 K=3 |
| IGNN | — | 按现有技术背景理性叙述其不足，不写"互补" |

## 编译

本机可用 tectonic（首次会自动下载宏包）：

```
tectonic main.tex
```

也可在 Overleaf / TeX Live 上 `pdflatex main → bibtex main → pdflatex main ×2`。

## 结构

```
V4_en/
├── main.tex              # 主文件，\input 以下各节
├── sec_abstract.tex      # 摘要
├── sec_intro.tex         # 1 引言（含图 1）
├── sec_front.tex         # 2 相关工作 / 3 问题形式化 / 4 字段级风险（定义、定理、K 的选择）/ 5 摊销攻击者
├── sec_setup.tex         # 6 实验设置（数据、真值、基线、指标）
├── sec_res1.tex          # 7 组合风险 / 8 定义对比
├── sec_res2.tex          # 9 摊销攻击者（保真度、消融、主干复用、成本）
├── sec_res3.tex          # 10 稳健性（K、种子/划分、τ）
├── sec_discussion.tex    # 11 讨论与局限 / 12 结论
├── sec_appendix.tex      # 附录：其余性质与证明、实现细节、字段表
├── cost_text.tex         # 成本段落（由计时结果生成）
├── ref.bib
├── tables/               # 全部由 scripts/make_tables.py 从实验 CSV 生成（tab_fields 为手写）
├── figures/              # 全部由 scripts/make_figs.py、make_schematics.py 生成（pdf 排版用，png 目检用）
└── scripts/
    ├── style.py          # 统一样式：Helvetica、白底、细线、已验证色板；本文方法固定蓝色
    ├── names.py          # 字段 / 目标英文短名
    ├── make_schematics.py# 图 1 框架、图 3 模型
    ├── make_figs.py      # 图 2、4–10
    └── make_tables.py    # 表 3–7
```

## 图表与数据来源（图号为编译稿中的编号）

| 编译稿 | 文件 | 内容 | 数据来源 |
| --- | --- | --- | --- |
| 图 1 | fig1_framework | 审计框架 | 示意图（make_schematics.py） |
| 图 2 | fig3_model | 摊销攻击者与扫描—认证 | 示意图（make_schematics.py） |
| 图 3 / 表 3 | fig4_combination_risk / tab_risk | 单字段定级漏掉的组合风险 | 111、112、116 号 `summary_targets.csv`、`V_official.npy` |
| 图 4 | fig5_profile | 字段风险画像（RTS C35）与增益矩阵（NEM PPCCGT） | 111、112 号 `field_profile.csv`、`gain_matrix_*.csv` |
| 图 5 | fig2_mechanism | 机理小例子 | 117 号 `toy_scores.csv`、`toy_truth.csv` |
| 图 6 / 表 4、5 | fig6_definition / tab_decision、tab_ranking | 定义对比 | 117 号 `outputs/analysis/*.csv` |
| 图 7 | fig7_estimator | 估计器保真度与认证曲线 | 各号 `est_*.npz` + 真值 |
| 图 8 / 表 6、7 | fig8_ablation / tab_ablation、tab_backbone | 对比、消融、主干复用 | 118 号 `variants_by_target.csv`；116 号 `backbone_compare.csv` |
| 图 9 | fig9_robustness | 稳健性与 K | 119 号 `outputs/analysis/*.csv` |
| 表 1 | （正文） | 要求—方法对照 | 手写 |
| 表 2 | （正文） | 数据集 | 手写（数字来自各号 D.npz） |
| 表 B.8 | tab_fields | 字段角色与规则依据 | 手写 |
| 成本段落 | cost_text.tex | 串行重训 vs 估计器 | 118 号 `outputs/est/*/timeE.npz`、`seq.npz`（make_cost.py） |

改动实验数字后：`cd scripts && python make_tables.py && python make_figs.py`，再编译。
