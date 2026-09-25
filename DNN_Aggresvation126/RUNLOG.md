# RUNLOG — 126 号

| 时间（09-25） | 步骤 | 命令 | 备注 |
| --- | --- | --- | --- |
| 17:05–17:25 | 统一计时（原实现） | `scripts/bench126.py --gpu 0` | → `outputs/time_<组>.csv`；`logs/bench.log` |
| 17:05–17:12 | 预训练计时 | `scripts/pretrain126.py --gpu 1` | → `outputs/pretrain_time.csv`（权重不保存） |
| 17:30 | 快速实现校验 | 内联脚本：RTS、CAISO 各 200 个集合，本文估计器 | 与原实现、与 125 号估计值逐位相同；快 2.76–2.87 倍 |
| 17:32 | 首次 `--fast` 运行失败 | 替换代码时留下缩进错误（IndentationError） | 修正后重跑 |
| 17:33–17:50 | 统一计时（快速实现） | `scripts/bench126.py --gpu 0 --fast` | → `outputs/time_fast_<组>.csv`；`logs/bench_fast.log` |
| 随时 | 图 | `scripts/figs126.py` | → `figures/` |
