# RUNLOG — 122 号

环境：`/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python`。GPU 四张卡都被他人 vLLM 进程占满（每卡约 23.4 GB），
只能在 GPU1/GPU2 上勉强训练主干，读出（B=4 时 OOM）全部放到 CPU 进程池（单线程，`OMP_NUM_THREADS=1`）。

| 时间（09-25） | 步骤 | 命令 | 耗时 / 备注 |
| --- | --- | --- | --- |
| 01:02–01:55 | 1 训练 + 诊断 | `scripts/run122.py --tag <组> --gpu {1,2,-1}`（`scripts/chain.sh` 串起五组） | 每个变体 35–380 s；PJM-load、RTS 在 CPU 上较慢；日志 `logs/run122_*.log` |
| 01:29–02:53 | 2a 读出（NEM、PJM-gen/ic、CAISO） | `scripts/est122.py --jobs 30 --tags NEM,PJM-gen/ic,CAISO-load` | 875 块，5,049 s；`logs/est_a.log` |
| 02:55–03:00 | 训练 recon 种子 1、2 | `scripts/train_seeds.py`（GPU1/GPU2 + CPU） | 每个 35–142 s；`logs/seeds_*.log` |
| 02:53–03:16 | 2b 读出（RTS、PJM-load） | `scripts/after_est.sh` → `est122.py --tags RTS-GMLC,PJM-load` | 322 块，1,360 s；`logs/est_b.log` |
| 03:16–03:18 | 3 大集合对照 | `scripts/large122.py`（树真值 + 四种主干读出） | 149 s；`logs/large.log` |
| 03:10–03:21 | 6 秩截断 | `scripts/rank122.py 8` | 384 块，659 s；`logs/rank.log` |
| 03:18– | 2c 三种子读出 | `scripts/after_seeds.sh` → `est122.py --variants uniformE,reconE` | 342 块；`logs/est_c.log` |
| 13:38–14:53 | 7 截断版读出（GPU 空出后，三卡并行） | `scripts/est122g.py --gpu {0,1,2} --tags …`（6 个变体 × 5 组，加 clip_y） | 每组每变体 90–960 s；`logs/cy_g*.log` |
| 15:00–15:25 | 种子稳健性 | 生成 random_seed{1,2}.pt；`est122g.py --variants recon+random@1,recon+random@2` | `logs/seed_g*.log` |
| 随时 | 8 决策比较 | `scripts/decide122.py` | → `outputs/analysis/decisions.csv`（uniformE 复现 117 号） |
| 随时 | 4 汇总 | `scripts/analyze122.py` | → `outputs/analysis/variants_by_target.csv`、`variants_mean.csv`、`large_sets.csv` |
| 随时 | 5 误差分解 | `scripts/principle122.py` | → `outputs/analysis/error_decomposition*.csv` |
| 随时 | 图 | `scripts/figs122.py` | → `figures/fig1–fig6` |

## 出错与修正

1. `run122.py` 原先写死 `cuda`，改为 `--gpu -1` 表示 CPU。
2. CPU 上批量求解报 `Intel oneMKL ERROR: Parameter 6 was incorrect on entry to DLASWP`：`torch.eye(d).expand(B,d,d)` 非连续内存，
   `src/readout122.py`（115 号 readout.py 的副本）改为 `.contiguous()`。
3. GPU0 上训练曾 OOM 一次，改用 GPU1/GPU2。
4. PJM-gen/ic 上 reconE 的 M 崩溃：单字段读出爆掉，定位后在 `src/readout122.py` 加 `clip_y`（预测截断）、`winsor`、`alphas` 选项，
   单独调试比较四种防护后选 clip_y（见 WORKLOG）。
5. 后台任务均保存 PID（`logs/*.pid`），用 `kill -0` 轮询。
