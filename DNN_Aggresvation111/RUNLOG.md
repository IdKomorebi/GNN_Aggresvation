# DNN_Aggresvation111 运行流水

- `2026-09-23 20:43` ✔ **[DISPATCH] DONE**　`scripts/build_rts.py`：全年 8,784 小时 DC-OPF，8784/8784 可行，40 进程 2 秒
- `2026-09-23 20:48` ✔ **[DATA] DONE**　`scripts/assemble_rts.py`：候选 23、目标 3 → outputs/dataset.csv、fields.json
- `2026-09-23 20:49` ✔ **[PREP] DONE**　`run_stage.py --stage prep`：train 6149 / test 2635，规模 ≤4 集合 10,902 个
- `2026-09-23 20:49 → 20:53` ✔ **[TREE] DONE**　梯度提升树真值，CPU 40 进程单线程，259 s
- `2026-09-23 20:49 → 21:15` ✔ **[MULTI] DONE**　多目标 DNN 真值，GPU2，1,575 s
- `2026-09-23 20:49 → 21:39` ✔ **[SINGLE] DONE**　单目标 DNN 真值（3 个目标各训一遍），GPU2，2,965 s
- `2026-09-23 21:13 → 21:39` ✔ **[ORACLE] DONE**　通用模型 3 种子预训练（52–66 s/种子）+ 全部集合估计，GPU1
- `2026-09-23 21:39 → 22:02` ⚠ **[ORACLE] 重复运行**　旧排队脚本未被正确停止（当初保存的是外层包装进程的 PID），
  在 GPU2 上把通用模型又跑了一遍；结果与第一轮相同（同种子、同 val 损失），est.npz 被同样的结果覆盖。已于 22:10 按实际 PID 终止并删除旧脚本
- `2026-09-23 22:02` ✔ **[ANALYZE] DONE**　`scripts/analyze_exp.py`（首次以机组 223 画增益矩阵）
- `2026-09-23 22:20` ✔ **[ANALYZE] DONE**　改以线路 C35 画增益矩阵后重跑（机组 223 的矩阵被单字段主导，看不出组合结构）
