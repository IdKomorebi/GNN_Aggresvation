# 图形、表格与实验来源

三版共享实验事实；期刊差异集中在问题视角、图中文字、组织顺序和解释重点。所有统计图来自既有输出，图中没有新生成的实验结果。示意图的曲线、拓扑与小型画像明确标为 illustrative。

| 资源 | 来源 | 本次处理 |
|---|---|---|
| `fig1_framework` | 本文审计定义；手工矢量示意 | 电网/数据持有方、发布矩阵、已有信息、增量分布、字段画像；各刊调整左侧对象及标注 |
| `fig3_model` | V6 方法、125 号固定配置 | 将共享预训练、两个冻结分支、逐集合 ridge、三种子平均、候选验证分层画出 |
| `fig_dispatch_mechanism` | 123 号受控案例的物理关系 | 新绘局部/远端节点、拥塞关系与报价段；不是 RTS 的真实网络拓扑，也不是实测报价曲线 |
| `fig_mechanism_narrow`、`fig_mechanism_scores` | `DNN_Aggresvation123/outputs/narrow/analysis` | 主文保留增量和拥塞对照；全方法评分矩阵移附录 |
| `fig_amplification` | `DNN_Aggresvation124/outputs/profile.csv`、`context_gain_distribution.csv` | 散点与背景增量分布上下排列，突出单字段/平均/最大增量 |
| `fig4_combination_risk` | 111、112、116 号 `summary_targets.csv` 等 | 两阈值组成比例与最佳集合推断力并列；颜色和标记一致 |
| `fig5_profile` | 111 号 `field_profile.csv`、112 号 PPCCGT gain matrix | 排序字段画像＋背景增量矩阵，避免图例与坐标轴重叠；矩阵使用矢量色块 |
| `fig7_estimator` | 125 号 `final_est`，对应 111、112、116 号 `V_official.npy` | 使用 V6 固定估计器数据；不是旧 V1 的估计器结果 |
| `fig_baselines` | 127 号 `outputs/analysis/summary.csv` 与已保存时间记录 | 精度/时间、认证恢复/时间、共同样本误差分为三个面板 |
| `fig9_robustness` | 119 号分析及各数据组 summary | 原拥挤横排调整为 2×2，统一阈值和方法标记 |
| `fig_scope` | 116 号 `backbone_compare.csv` | 附录保留跨目标表示复用证据 |
| `fig2_mechanism` | 117 号 `toy_scores.csv`、`toy_truth.csv` | 附录保留已存在的合成机制，不改变生成参数 |
| `fig_masking` | 122 号诊断、125 号 `select` | 主文减少配置日志，附录保留维度、集合大小和稳定性对照 |
| `tab_risk`、`tab_risk_sets` | 原 V6 `tab_risk` 的完整数值 | 14 列拆为主文 8 列和附录 7 列（目标列重复），原数值全部保留 |
| `tab_design_main`、`tab_design` | V6 及 122/125 号 | 主文突出 5 个核心设计，完整配置表在附录 |
| `tab_baselines`、`tab_baselines_extra` | V6 及 127 号 | 主文保留风险、认证、召回和时间；额外指标放附录 |
| 其他表格 | 原 V6 对应 LaTeX 表及已有分析输出 | 只调整排版、简称与所在章节 |

重绘由 `scripts/make_figs.py`、`scripts/make_schematics.py` 和 `scripts/style.py` 完成。所有 14 幅图均交付 PDF / SVG / PNG；论文嵌入 PDF。蓝/橙为主色，辅以灰/绿，点形和线型辅助区分。SVG 保留可编辑文字，PDF 嵌入字体，PNG 仅供预览。

数据量口径：4 个数据来源、5 组训练任务、10 个目标、264 个字段—目标对；124,425 个集合为各数据组在既定最大阶数内的穷举总数。`K=2` 的字段背景审计使用至多 3 字段集合；35 字段对应 7,175 个这样的集合。52,360 是恰好 4 字段的额外集合数，59,535 是该组至多 4 字段的累计数。
