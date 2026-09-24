# DNN_Aggresvation116 运行流水

- `2026-09-24 01:18` ✔ **[BUILD] DONE**　`scripts/build_groups.py`：三组 D.npz 与 fields.json（候选 22 / 24 / 35，规模 ≤4 全枚举）；新增列与已有列无完全重复（最大 |r| 0.991，total_lmp_da 对日前系统电能价）
- `2026-09-24 01:19` 启动四条链（`scripts/chain.sh`，脚本内以 `$$` 自写 PID）：
  - GPU0：caiso_load 单目标 DNN（59,535 集合）
  - GPU1：pjm_gen_ic 多目标 / 单目标 DNN，pjm_load 单目标 DNN
  - GPU2：三组本组主干 → 两个数据集全列主干 → 三组留目标主干（各 3 种子）
  - CPU：三组梯度提升树（40 进程单线程）
  - GPU3 留空
- `2026-09-24 01:38` ✔ **[BB-TRAIN] DONE**　GPU2：三组本组主干、两数据集全列主干、三组留目标主干（各 3 种子）
- `2026-09-24 02:02` ✔ **[TRUTH] DONE**　GPU1：pjm_gen_ic 多目标 / 单目标 DNN，pjm_load 单目标 DNN
- `2026-09-24 02:03` ✔ **[ANALYZE] DONE**　pjm_load、pjm_gen_ic（`scripts/analyze_group.py`，估计器为新主口径 E）
- `2026-09-24 02:15` ✔ **[EST] DONE**　GPU1/2：三组 × 三种主干的 E 估计（规模 ≤3；PJM 约 100 ms/集合）
- `2026-09-24 02:30` ✔ **[TRUTH] DONE**　GPU0：caiso_load 单目标 DNN（59,535 集合，71 分钟）
- `2026-09-24 02:32` ✔ **[ANALYZE] DONE**　caiso_load；`scripts/backbone_compare.py`（GPU1）；`scripts/summarize116.py`
- 事故记录：同时段 117 号打分脚本曾因 dcor 拉起 numba 线程池，把服务器负载推到约 1800，已立即按 PID 终止并改为强制单线程（见 117 号 RUNLOG）
