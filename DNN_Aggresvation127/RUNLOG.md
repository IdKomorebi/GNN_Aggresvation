# RUNLOG — 127 号

隔离环境：TabPFN 装在 `/data1/duhaocun/envs/tabpfn_venv`（基于 Pytorch310_codex 的 venv，`--system-site-packages`，`pip install tabpfn==2.2.1`；导入前须 `SCIPY_ARRAY_API=1`）。
GPU0 下午被他人进程占用（约 16 GB），本号用 GPU1/2/3。

| 时间（09-25） | 步骤 | 命令 | 耗时 / 备注 |
| --- | --- | --- | --- |
| 18:05 | 写定方案 | `base.yaml` | |
| 18:08 | 试跑 | 内联：RTS 30 个集合 | LazyVI 在 GPU0 OOM（他人进程）→ 改 GPU1、雅可比分块 512；单精度 0.9 s/集合但误差明显变大 → 保留双精度 |
| 18:14–19:13 | 全模型 + Dropout + 热启动 | `scripts/run_full.py --gpu 1` | 热启动全表：RTS 至 CAISO 共约 1 小时 |
| 18:14–19:49 | LazyVI（200 集合样本） | `scripts/run_lazy.py --gpu 1` | 1.5–10.4 s/集合 |
| 18:14–21:36 | TabPFN 全表（6 分片） | `scripts/tabpfn_gpu.sh {2,3} <分片>` | 约 3.4 小时 |
| 20:45 | 外部方法计时 | `scripts/time127.py --gpu 1` | → `outputs/time_<组>.csv` |
| 21:40 | TabPFN 计时 | `SCIPY_ARRAY_API=1 tabpfn_venv/bin/python scripts/time127_tabpfn.py --gpu 1` | → `outputs/time_tabpfn.csv` |
| 21:37 | 汇总 | `scripts/eval127.py` | → `outputs/analysis/` |
| 随时 | 图 | `scripts/figs127.py` | → `figures/` |
