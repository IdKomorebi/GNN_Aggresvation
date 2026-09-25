# DNN_Aggresvation123 运行流水

- `2026-09-25 01:08` ✔ `scripts/build123.py`（wide，κ~U(±30%)）：8,784 小时 DC-OPF，3 秒（40 进程）
- `2026-09-25 01:12` ✔ `scripts/assemble123.py`：11 个字段，2,047 个子集
- `2026-09-25 01:13` ✘ GPU 上 DNN 真值两次显存不足（他人 vLLM 进程占满 4 张卡，每张只剩约 0.1–1 GB）→ 改为 CPU 分片（`scripts/truth123_cpu.py`，10 进程 × 3 线程）
- `2026-09-25 01:20` ✔ 树真值（wide，15 秒）
- `2026-09-25 01:35` ✔ `scripts/regime123.py`（wide）：发现出力单独已 0.52，"1+1>2"不明显 → 增加 narrow 版本（κ~U(±5%)）
- `2026-09-25 01:37` ✔ narrow：出清、组装、树真值（24 秒）、状态分析；01:37 启动 narrow 的 CPU DNN 真值
- `2026-09-25 02:48 / 02:50` ✔ wide / narrow 的 DNN 真值完成（各约 55 分钟）
- `2026-09-25 02:58` ✔ `scripts/analyze123.py`（两个版本）；图由 paper/claude/V5_en/scripts/make_figs.py 生成后复制到本目录
- 注：一次误把 `run_stage.py --stage tree` 指向本号根目录（旧 wide 数据），结果与已有树真值相同，无影响
