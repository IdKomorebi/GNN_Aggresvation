# DNN_Aggresvation76 运行流水

> 本文件由 `src/runlog.py` 自动追加，与 `CHANGELOG.md`（结论）并列。
> 机器可读版本：`outputs/events.jsonl`

## 2026-07-23

### 阶段 0：环境与口径自检

- `03:06:55` ✔ **[SETUP] DONE**　目录骨架 + base.yaml 与 69 号一致 + runlog 自检通过　　GPU -

### 阶段 1：F-1 微调相变曲线

- `03:13:49` ▶ **[F-1] START**　相变扫描 9 job（3 方案 × 3 seed）× 990 子集 × 17 个 K　　进度 0/9　GPU 0,1
- `03:16:40` ✔ **[F-1] DONE**　uniform seed1：990 子集 × 17 个 K　　实耗 2m48s
- `03:16:59` ✔ **[F-1] DONE**　uniform seed0：990 子集 × 17 个 K　　实耗 3m07s
- `03:19:44` ✔ **[F-1] DONE**　uniform seed2：990 子集 × 17 个 K　　实耗 2m57s
- `03:20:00` ✔ **[F-1] DONE**　none seed0：990 子集 × 17 个 K　　实耗 2m58s
- `03:22:45` ✔ **[F-1] DONE**　none seed1：990 子集 × 17 个 K　　实耗 2m53s
- `03:22:59` ✔ **[F-1] DONE**　none seed2：990 子集 × 17 个 K　　实耗 2m52s
- `03:25:52` ✔ **[F-1] DONE**　bern50 seed1：990 子集 × 17 个 K　　实耗 2m50s
- `03:26:09` ✔ **[F-1] DONE**　bern50 seed0：990 子集 × 17 个 K　　实耗 3m17s
- `03:28:47` ✔ **[F-1] DONE**　bern50 seed2：990 子集 × 17 个 K　　实耗 2m50s
- `03:28:50` ✔ **[F-1] DONE**　相变扫描成功 9，失败 0　　实耗 15m00s　进度 9/9　GPU 0,1
- `03:29:32` ◆ **[F-1] DECISION**　uniform 的 K*=25 < none 的 K*=35 → '少步数发现'确属随机失活起点，主张二成立
- `03:30:02` ◆ **[F-1] DECISION**　uniform 的 K*=25 < none 的 K*=35 → '少步数发现'确属随机失活起点，主张二成立

### 阶段 1b：F-2 自适应早停（零 GPU）

- `03:31:40` ◆ **[F-2] DECISION**　推荐规则 eps=0.002, delta=0.02, patience=2：平均步数 22.8（省 54%），召回 93.0%（不低于基线），误报 0.00%
