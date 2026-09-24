# DNN_Aggresvation112 运行流水

- `2026-09-23 20:18 → 20:40` ✔ **[DOWNLOAD] DONE**　`data/external/NEM_2024/download.sh` + `download2.sh`：48 个月度文件，约 300 MB
  （2024-08 起文件名格式变更，前 7 个月一次成功，后 5 个月按新格式补下）
- `2026-09-23 21:02` ✘ **[DATA] 第一版**　SS_*_AVAILABILITY 自 2024-05 才有，缺 2,904 小时，只剩 5,880 小时 → 弃用
- `2026-09-23 21:07` ✔ **[DATA] DONE**　`scripts/assemble_nem.py` 改用 SS_*_UIGF：全年 8,784 小时，候选 30、目标 3，无缺失
- `2026-09-23 21:07` ✔ **[PREP] DONE**　规模 ≤3 集合 4,525 个
- `2026-09-23 21:07 → 21:08` ✔ **[TREE] DONE**　梯度提升树真值，CPU 40 进程单线程，64 s
- `2026-09-23 21:13 → 21:30` ✔ **[ORACLE] DONE**　通用模型，GPU1
- `2026-09-23 21:13 → 21:31` ✔ **[MULTI] DONE**　多目标 DNN 真值，GPU1，1,082 s
- `2026-09-23 21:13 → 21:45` ✔ **[SINGLE] DONE**　单目标 DNN 真值，GPU1，1,923 s
- `2026-09-23 21:39 → 22:11` ⚠ **重复运行**　旧排队脚本在 GPU2 上把 multi / single / oracle 又跑了一遍（原因见 111 号 RUNLOG），
  结果与第一轮相同并覆盖写出；22:10 按实际 PID 终止
- `2026-09-23 22:11` ✔ **[ANALYZE] DONE**　`../DNN_Aggresvation111/scripts/analyze_exp.py`
- `2026-09-23 22:25` ✔ **[CERT] DONE**　`scripts/cert_curve.py`：认证 k 个候选背景的下界比（含 111 号对照）→ 图 5
