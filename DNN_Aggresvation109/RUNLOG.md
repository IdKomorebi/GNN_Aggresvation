# DNN_Aggresvation109 运行流水

> 自动记录；实验结论见 `CHANGELOG.md`。

## 真值队列（scripts/queue109.sh，GPU 0/1/2 三卡分片）

- `2026-09-18 22:14:05 → 22:14:50` ✔ **[TRUTH] DONE**　pjm_f010（n_aux=459）45s
- `2026-09-18 22:14:50 → 22:16:02` ✔ **[TRUTH] DONE**　pjm_f025（n_aux=1147）72s
- `2026-09-18 22:16:02 → 22:18:03` ✔ **[TRUTH] DONE**　pjm_f050（n_aux=2294）121s
- `2026-09-18 22:18:03 → 22:21:41` ✔ **[TRUTH] DONE**　pjm_f100（n_aux=4589）218s
- `2026-09-18 22:21:41 → 22:22:38` ✔ **[TRUTH] DONE**　caiso_f010（n_aux=610）57s
- `2026-09-18 22:22:38 → 22:24:13` ✔ **[TRUTH] DONE**　caiso_f025（n_aux=1526）95s
- `2026-09-18 22:24:13 → 22:26:53` ✔ **[TRUTH] DONE**　caiso_f050（n_aux=3052）160s
- `2026-09-18 22:26:53 → 22:31:40` ✔ **[TRUTH] DONE**　caiso_f100（n_aux=6105）287s
- 合计 17 分 35 秒，24 个分片文件。

## 分析

- `2026-09-18 22:33` ✔ **[ANALYZE] DONE**　`scripts/analyze109.py` → 109_A/B/C/D.csv + report109.md
- `2026-09-18 22:36` ✔ **[CURSE] DONE**　`scripts/winners_curse.py` → 109_E_winners_curse.csv
- `2026-09-18 22:40` ✔ **[FIG] DONE**　`scripts/make_figs.py` 4 张图（修正 fig1-B 的错误标题与 fig4 净偏差折线）
