# DNN_Aggresvation85 运行流水

> 自动记录；实验结论见 `CHANGELOG.md`。

- `2026-07-25 03:50:03` · **[DIAG] SYN3**　13244 三阶 5s
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order4 tau=0.05 n=7122 medRank=478 blindB1000=0.2331
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order4 tau=0.1 n=230 medRank=138 blindB1000=0.1043
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order4 tau=0.15 n=11 medRank=457 blindB1000=0.1818
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order5 tau=0.05 n=7854 medRank=1030 blindB1000=0.5071
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order5 tau=0.1 n=111 medRank=374 blindB1000=0.3243
- `2026-07-25 03:50:05` · **[DIAG] ROW**　order5 tau=0.15 n=6 medRank=301 blindB1000=0.1667
- `2026-07-25 03:52:49` · **[PEEL] PROG**　restart 50/60 已找到 51 个极小集; solves=35572
- `2026-07-25 03:52:54` ✔ **[PEEL] DONE**　restarts=60 usize=14 tau=0.1 solves=42260 用时35s 找到62个
- `2026-07-25 03:54:46` · **[CMP] ROW**　order4 blind=24/230 cov beam=0.8957 peel=0.0261 union=0.8957 blind_peel=0.0
- `2026-07-25 03:56:12` · **[SPLIT] INFO**　train=4589 search=983 audit=984
- `2026-07-25 03:56:12` ✔ **[SPLIT] DONE**　三分割缓存写出，用时 1s
- `2026-07-25 03:56:15` · **[CMP] ROW**　order5 blind=36/111 cov beam=0.6486 peel=0.0631 union=0.6577 blind_peel=0.0278
- `2026-07-25 03:58:36` · **[DIAG2] SPLIT**　强四阶 230: 盲区 24 / 嵌套 206
- `2026-07-25 03:58:37` · **[DIAG2] VMAX**　1s
- `2026-07-25 04:00:44` · **[CLIQUE] PAIRS**　946 二元 + 全集 1.3s
- `2026-07-25 04:00:47` · **[CLIQUE] VERIFY**　order4 tau=0.1: 229/230 落在团内
- `2026-07-25 04:00:47` · **[CLIQUE] VERIFY**　order5 tau=0.1: 106/111 落在团内
- `2026-07-25 04:02:51` ▶ **[CHECK] START**　24 个盲区强四阶
- `2026-07-25 04:06:15` · **[MISS] BEAM**　order4 touched=30967 found=206 18s
- `2026-07-25 04:06:23` · **[MISS] PROG**　order4 20000/104784 k=1 2504/s
- `2026-07-25 04:06:31` · **[MISS] PROG**　order4 40000/104784 k=7 2478/s
- `2026-07-25 04:06:39` · **[MISS] PROG**　order4 60000/104784 k=16 2461/s
- `2026-07-25 04:06:48` · **[MISS] PROG**　order4 80000/104784 k=20 2424/s
- `2026-07-25 04:06:58` · **[MISS] PROG**　order4 100000/104784 k=24 2317/s
- `2026-07-25 04:07:12` · **[MISS] ROW**　order4 found=206 miss_hat=24.0 miss_hi=33.8 true_miss=24 holds=True
- `2026-07-25 04:07:30` · **[MISS] BEAM**　order5 touched=35141 found=72 18s
- `2026-07-25 04:07:39` · **[MISS] PROG**　order5 20000/150000 k=1 2185/s
- `2026-07-25 04:07:48` · **[MISS] PROG**　order5 40000/150000 k=2 2189/s
- `2026-07-25 04:07:57` · **[MISS] PROG**　order5 60000/150000 k=2 2199/s
- `2026-07-25 04:08:06` · **[MISS] PROG**　order5 80000/150000 k=3 2196/s
- `2026-07-25 04:08:16` · **[MISS] PROG**　order5 100000/150000 k=4 2185/s
- `2026-07-25 04:08:25` · **[MISS] PROG**　order5 120000/150000 k=4 2183/s
- `2026-07-25 04:08:34` · **[MISS] PROG**　order5 140000/150000 k=5 2180/s
- `2026-07-25 04:08:41` · **[MISS] ROW**　order5 found=72 miss_hat=35.0 miss_hi=73.7 true_miss=39 holds=True
