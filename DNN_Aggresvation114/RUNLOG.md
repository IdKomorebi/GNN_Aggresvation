# DNN_Aggresvation114 运行流水

- `2026-09-23 23:14` ✔ **[BUILD] DONE**　`scripts/build_groups.py`：三组 D.npz 与 fields.json（规模 ≤4 全枚举）
- `2026-09-23 23:16` 启动四条链（`scripts/chain.sh`，脚本内以 `$$` 自写 PID，并用 ps 核对）：
  - GPU0：caiso_load 单目标 DNN（12,950 集合）→ 23:30 完成
  - GPU1：pjm_gen_ic 多目标 DNN、单目标 DNN，pjm_load 单目标 DNN → 23:26 完成
  - GPU2：三组通用模型（3 种子预训练 + 全部集合估计）→ 23:34 完成
  - CPU：三组梯度提升树（40 进程单线程）→ 23:19 完成
  - GPU3 全程留空
- `2026-09-23 23:28` ✔ **[ANALYZE] DONE**　pjm_load、pjm_gen_ic（`../DNN_Aggresvation111/scripts/analyze_exp.py`）
- `2026-09-23 23:36` ✔ **[ANALYZE] DONE**　caiso_load；`scripts/summarize114.py` 出总览图（修正：图 C 原把不同目标连成折线，改为只画点）
- `2026-09-23 23:40` ✔ **[SENS] DONE**　`scripts/sensitivity_proxies.py`：近似代理字段剔除与否的敏感性
