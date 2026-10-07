# RUNLOG（01 号）

环境：`/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python`；`OMP_NUM_THREADS=1`。GPU0 有他人任务，GPU3 留空，本实验只用 GPU1、GPU2。

| 时间（2026-10-07） | 命令 | 输出 | 用时 |
|---|---|---|---|
| 21:17 | `python scripts/run_e1_e2.py` | `outputs/e1_*.csv`、`outputs/e2_*.csv`，日志 `logs/e1_e2.log` | E1 40 s，E2 103 s（纯 CPU） |
| 21:18 | `CUDA_VISIBLE_DEVICES=1 python scripts/train_grpo.py --seed 0 --iters 300`；`CUDA_VISIBLE_DEVICES=2 ... --seed 1` | `outputs/grpo/policy_seed{0,1}.pt`、`trainlog_seed{0,1}.json`，日志 `logs/grpo_seed{0,1}.log` | 各约 49 分钟（单线程采样是瓶颈，GPU 只承担很小的前向/反向） |
| 21:24 | `python scripts/run_e2b.py` | `outputs/e2b_*.csv`，日志 `logs/e2b.log` | 约 80 s |
| 21:41 | `python scripts/eval_e3.py`（CPU，5 进程，每组一个） | `outputs/e3_*.csv`、`e3_verdict.json`，日志 `logs/e3.log` | 约 3 分钟 |
| — | `python scripts/figs.py` | `figures/图1–图3` | 数秒 |

说明：
- 长任务都以 `nohup … &` 启动并把 PID 写入 `logs/*.pid`，用 `kill -0` 轮询。
- E3 第一次评测在训练进行到约 150 轮时用当时的最优检查点做了一遍（检查点备份在 `outputs/grpo/ckpt_at_eval/`）；训练结束后用最终最优检查点重评，`outputs/e3_*` 为最终结果。
