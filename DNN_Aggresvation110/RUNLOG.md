# DNN_Aggresvation110 运行流水

- `2026-09-23 17:45` ✔ **[DATA] DONE**　`scripts/gen_data.py 4000`：DC-OPF 3996/4000 可行，24 s（96 进程）
  - 两轮调参记录：第一轮 L5-7 与 L24-26 同时收紧 → 可行率 58%、阻塞率 86%、电价尖峰 3,519；
    第二轮只收紧 L24-26（78%）→ 阻塞率 71%；第三轮 88% + 负荷上限 1.18 → 阻塞率 56.7%，尖峰 <0.5%（采用）
- `2026-09-23 17:47` ✘ **[TRUTH] KILLED**　`scripts/exact_v.py` 首次运行：120 进程 × 每进程 80 个 OpenMP 线程，
  服务器负载均值升至 4,047，按精确 PID 终止 121 个进程
- `2026-09-23 17:54` ✔ **[TRUTH] DONE**　`OMP_NUM_THREADS=1`、40 进程重跑：2,047 子集 × 2 目标，24 s
- `2026-09-23 17:57` ✔ **[ANALYZE] DONE**　`scripts/analyze.py`：M^(K)、见证、MUS
- `2026-09-23 18:05` ✔ **[FIG] DONE**　`scripts/gain_matrix.py`：增益矩阵 CSV、字段画像 CSV、2 张图
