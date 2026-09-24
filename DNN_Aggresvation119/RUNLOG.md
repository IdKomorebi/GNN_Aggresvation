# DNN_Aggresvation119 运行流水

- `2026-09-24 01:40` ✔ **[BUILD] DONE**　`scripts/build119.py`：7 个数据组
- `2026-09-24 01:37–02:07` ✔ CPU：全部 7 组的梯度提升树（40 进程单线程）
- `2026-09-24 02:19–04:54` GPU：
  - GPU2：rts_seed1 / rts_seed2（多目标 + 单目标 DNN）、pjm_load_seed1 / 2（单目标 DNN），04:54 完成
  - GPU0：rts_split43（多目标 + 单目标），split44（多目标 + 单目标），nem_k4 多目标，04:38 完成
  - GPU1：nem_k4 单目标（02:51–04:45）
  - 调度：02:50 为提前完成 NEM，停掉原 GPU0 链的外壳（当时正在运行的 split43 single 不受影响、正常完成），把 nem_k4 single 挪到 GPU1
  - GPU3 全程留空
- `2026-09-24 03:20` ✔ τ 扫描（`analyze119.py C`）
- `2026-09-24 04:58` ✔ **[ANALYZE] DONE**　`analyze119.py AB`、`k3_critical.py`、`figs119.py`
