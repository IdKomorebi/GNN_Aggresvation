# DNN_Aggresvation117 运行流水

- `2026-09-24 01:30` ✘ **[SCORES] 中止**　首次运行 `scripts/scores.py`：dcor 在 fork 前拉起 numba 线程池，40 个子进程各 160 线程，服务器负载冲到约 1800；
  按 PID 立即终止（同时终止 toy.py）。修正：脚本开头强制 OMP/OPENBLAS/MKL/NUMBA 线程数 = 1，子进程初始化再调 threadpool_limits(1)；实测子进程 1 线程。
- `2026-09-24 01:36` ✔ **[SCORES] DONE**　RTS-GMLC（76s）、NEM（144s），CPU 36 进程
- `2026-09-24 01:37` ✔ **[TOY] DONE**　`scripts/toy.py`：63 个集合重训 + 全部打分
- `2026-09-24 02:18` ✔ **[SCORES] DONE**　PJM 两组；CAISO 因真值未就绪报错，02:35 补跑（161s）
- `2026-09-24 02:40` ✔ **[ANALYZE] DONE**　`scripts/analyze117.py`（10 个目标）；`scripts/figs117.py` 出 3 张图
