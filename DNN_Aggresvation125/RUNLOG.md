# RUNLOG — 125 号

环境：`/data1/duhaocun/miniconda3/envs/Pytorch310_codex/bin/python`；GPU0/1/2（GPU3 留给他人），CPU 单线程。

| 时间（09-25） | 步骤 | 命令 | 耗时 / 备注 |
| --- | --- | --- | --- |
| 16:05 | 写定选择规则 | `base.yaml` | 在 C3 运行前 |
| 16:07 | 自检：批量三种子读出 ≡ 122 号 readout_avg | 内联脚本（RTS 20 个集合） | 最大差 0.0 |
| 16:08–16:44 | 1 C3 读出 | `scripts/est125.py --gpu {0,1,2} --tags …` | RTS 734 s、PJM-gen/ic 710 s、NEM 1,424 s、PJM-load 654 s、CAISO 2,149 s；`logs/c3_g*.log` |
| 16:10–16:20 | 派生脚本自检（旧估计器） | `FINAL125=outputs/check_Eold OUT125=check_d117/check_d124 scripts/d117.py / d124.py` | 与 117/124 号原结果逐数一致 |
| 16:45 | 2 候选比较与选择 | `scripts/select125.py` | → `outputs/select/`；C3 入选 |
| 16:46 | 固定估计器 | `scripts/make_final.py 125 RRE_cy outputs/final_est` | |
| 16:46–17:05 | 3 下游重算 | `scripts/d117.py`、`scripts/d124.py`（CPU） | → `outputs/d117/`、`outputs/d124/` |
| 16:46–16:55 | 4 计时 | `scripts/time125.py --gpu 0`（其余两卡空闲） | → `outputs/time/` |
| 16:50 | 上尾分解 | `scripts/dec125.py` | → `outputs/select/error_decomposition*.csv` |
| 随时 | 图 | `scripts/figs125.py` | → `figures/` |
