# 图形源文件及实验来源

所有脚本位于本目录 `scripts/`，只读取现有实验输出。图源格式为 Python，导出 PDF、SVG、PNG 三种文件。PDF 嵌入论文，SVG 用于矢量编辑，PNG 用于预览。

| 正文图号 | 图文件 | 绘制入口 | 来源 |
|---|---|---|---|
| 1 | `fig1_framework` | `make_schematics.py: fig_framework` | 原 V4 方法语义；新绘制的示意电网、字段组合和审计流程 |
| 2 | `fig3_model` | `make_schematics.py: fig_model` | 原 V4 主实验的目标监督、随机掩码、3×256 主干、逐子集 ridge、3-seed 预测平均；绘图矩阵为示意 |
| 3 | `fig4_combination_risk` | `make_figs.py: fig_combination` | 111、112、116 各组的 `analysis/summary_targets.csv`、`V_official.npy`、`D.npz`、`fields.json` |
| 4 | `fig5_profile` | `make_figs.py: fig_profile` | 111 的 `analysis/field_profile.csv`；112 的 `analysis/gain_matrix_机组出力_PPCCGT.csv`（原有顺序前 8 行/列） |
| 5 | `fig2_mechanism` | `make_figs.py: fig_toy` | 117 的 `toy_truth.csv`、`toy_scores.csv`；依赖结构根据原 V4 中的生成关系绘制 |
| 6 | `fig6_definition` | `make_figs.py: fig_definition` | 117 的 `analysis/decision_rules.csv`、`detection.csv`、`withholding_strategies.csv` |
| 7 | `fig7_estimator` | `make_figs.py: fig_estimator` | 111/112 的 `est_variants.npz`，116 各组的 `est_group.npz`；同组 `D.npz`、`V_official.npy`、`fields.json`；复用 111 的 `pipe.py` 数值函数 |
| 8 | `fig8_ablation` | `make_figs.py: fig_ablation` | 118 的 `analysis/variants_by_target.csv` |
| 9 | `fig9_robustness` | `make_figs.py: fig_robust` | 119 的 `analysis/nem_k3.csv`、`k3_critical.csv`、`robust_consistency.csv`、`tau_sweep.csv`，以及各组原汇总表 |

116 的三组路径为 `groups/pjm_load/outputs`、`groups/pjm_gen_ic/outputs`、`groups/caiso_load/outputs`。111 和 112 使用其各自 `outputs` 目录。

`scripts/data.py` 复用 V4 绘图脚本的数据读取和数值整理逻辑，`scripts/names.py` 沿用原稿英文名称映射；`style.py`、方法图和结果图布局为本次重绘。估计器散点使用固定随机种子 0 抽取每组至多 450 个字段集合，字段风险面板保留全部字段和目标；其余图不进行随机抽样。

主表数据从 V4 的 LaTeX 表格沿用，仅调整标题措辞、字号、列宽、表线和出现位置。`tables/cost.csv`、`cost_text.tex` 沿用原 V4 顺序成本统计。`source_manifest.json` 提供原稿来源文件 SHA-256。
