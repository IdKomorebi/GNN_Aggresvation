# DNN_Aggresvation104 运行流水

> 补记（本号脚本未接入 runlog 模块，按产物时间戳回填）；实验结论见 `CHANGELOG.md`。

- `2026-09-18 20:04:40` ✔ **[MTABLE] DONE**　`scripts/build_mtable.py`：在 103 号正式真值上生成 M 全表
  - `outputs/analysis/M_table_official.csv`　2,952 行 = 2 数据集 × 12 目标 × 41 字段 × K∈{0,1,2}
  - `outputs/analysis/M_table_summary.csv`　6 行（数据集 × K）汇总
- `2026-09-18 20:06:32` ✔ **[FIG] DONE**　`scripts/make_figs.py`：3 张图
  - `figures/fig1_M_fidelity.png`
  - `figures/fig2_critical_detection.png`
  - `figures/fig3_escalation_official.png`
