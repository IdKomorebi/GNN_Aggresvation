# DNN_Aggresvation85 运行流水

> 自动记录；实验结论见 `CHANGELOG.md`。

- `2026-07-24 23:42:49` ▶ **[CACHE] START**　poly2充分统计量构建与全空间扫描
- `2026-07-24 23:42:51` ▶ **[RESIDUAL] START**　uniform K0 + structured residual ridge
- `2026-07-24 23:43:01` ✔ **[CACHE] DONE**　扫描14190集合，5.04s，maxΔ=1.16e-10
- `2026-07-24 23:43:28` ✔ **[RESIDUAL] DONE**　3143集合×3模型完成
- `2026-07-24 23:46` ✔ **[CACHE-WARM] DONE**　热读取0.035s，全14190集合扫描4.96s
- `2026-07-24 23:50` ✔ **[RESIDUAL-ALL] DONE**　poly2全空间60.85s；arithmetic全空间94.16s
- `2026-07-24 23:54` ✔ **[SEARCH] DONE**　全空间候选曲线、随机/定向真值分层与rank融合完成
- `2026-07-24 23:56` ✔ **[HIGH-ORDER] DONE**　D63 105真值点、稀疏Top50/100/200与枚举成本完成
- `2026-07-24 23:58` ✔ **[VALIDATION] PASS**　缓存等价、K0复现、全量覆盖与tune/test隔离检查通过
- `2026-07-24 23:46:00` ▶ **[CACHE] START**　poly2充分统计量构建与全空间扫描
- `2026-07-24 23:46:10` ✔ **[CACHE] DONE**　扫描14190集合，4.96s，maxΔ=1.16e-10
- `2026-07-24 23:46:36` ▶ **[RESIDUAL] START**　uniform K0 + structured residual ridge；scope=all, models=('arith',)
- `2026-07-24 23:48:10` ✔ **[RESIDUAL] DONE**　14190集合×1残差模型完成
- `2026-07-24 23:48:58` ▶ **[RESIDUAL] START**　uniform K0 + structured residual ridge；scope=all, models=('poly2',)
- `2026-07-24 23:49:59` ✔ **[RESIDUAL] DONE**　14190集合×1残差模型完成
