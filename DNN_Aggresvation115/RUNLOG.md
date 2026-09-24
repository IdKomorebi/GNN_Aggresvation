# DNN_Aggresvation115 运行流水

- `2026-09-23 23:20` ✔ **[DIAG] DONE**　估计见证处误差分解（CPU）：高估来自 V̂(T) 崩溃，而非 V̂(T∪i) 偏高
- `2026-09-23 23:22` ✔ **[COLLAPSE] DONE**　`scripts/check_collapse.py`（GPU2）：NEM 124 个崩溃集合上 5 种读出设置对照
- `2026-09-23 23:24` 启动 `scripts/run_variants.py`（变体 C–F）：
  - NEM（4,525 集合）GPU2 → 23:52 完成，375 ms/集合（四个变体合计）
  - RTS-GMLC、114/pjm_load、114/pjm_gen_ic → 改到 GPU1（`scripts/chain115b.sh`）→ 23:37 完成
  - 114/caiso_load → GPU2，等 114 号通用模型训完后开始 → 23:49 完成
  - 中途把最初的串行队列（chain115.sh）拆成两条 GPU 链：先停外层脚本、保留正在运行的 NEM 进程，再启动新链
- `2026-09-23 23:40` ✔ **[G] DONE**　`scripts/run_fullbackbone.py`（GPU1）：全字段主干复用，114 号三组
  - 首次运行写入 est_variants.npz，与 CAISO 组的 run_variants 存在竞写，停止后改写独立文件 est_G.npz 重跑
- `2026-09-23 23:55` ✔ **[ANALYZE] DONE**　`scripts/analyze115.py`（修正：图 1 标题"六种"→"七种"、目标名缩短；图 2 注明 G 只有 4 个目标）
- `2026-09-24 00:40` ✔ **[WHY] DONE**　`scripts/why_full_backbone.py`（GPU1）：两种主干的死神经元、有效维度与误差分解
- GPU3 全程留空
