# DNN_Aggresvation113 运行流水

- `2026-09-23 21:00` ✔ **[ANALYZE] DONE**　`scripts/analyze113.py`：PJM 3 目标 + CAISO 1 目标 × 口径 A/B；
  读取 103 号正式真值与 L0ensx 估计，不重训 → 113_summary.csv、113_field_profile.csv、113_candidate_roles.csv、增益矩阵
- `2026-09-23 21:05` ✔ **[FIG] DONE**　`scripts/make_figs.py`：3 张图（修正横轴标签重叠；增益矩阵由 PJM 负荷改为 CAISO 负荷，见 WORKLOG）
