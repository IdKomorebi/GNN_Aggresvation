# DNN_Aggresvation79_HierarchicalBudget 运行流水

> 自动日志；机器可读版本在 `outputs/events.jsonl`。

## 2026-07-24

- `00:56:25` ▶ **[SCHED] START**　待运行 5 个 K 网格任务，GPU=[0, 1, 2, 3]　　n_jobs=5　gpus=0,1,2,3
- `00:56:27` ▶ **[KGRID] START**　low 同轨迹扫描：990/990 个集合，K=[0, 1, 5, 10, 25, 50]　　kind=low　shard=0　nshard=1　gpu=cuda
- `00:56:28` ▶ **[KGRID] START**　triple 同轨迹扫描：3311/13244 个集合，K=[0, 1, 5, 10, 25, 50]　　kind=triple　shard=2　nshard=4　gpu=cuda
- `00:56:28` ▶ **[KGRID] START**　triple 同轨迹扫描：3311/13244 个集合，K=[0, 1, 5, 10, 25, 50]　　kind=triple　shard=0　nshard=4　gpu=cuda
- `00:56:28` ▶ **[KGRID] START**　triple 同轨迹扫描：3311/13244 个集合，K=[0, 1, 5, 10, 25, 50]　　kind=triple　shard=1　nshard=4　gpu=cuda
- `00:58:18` ✔ **[KGRID] DONE**　low shard 0/1 完成 990 个集合　　实耗 1m50s　kind=low　shard=0　n_queries=990　n_rows=71280
- `00:58:22` ▶ **[KGRID] START**　triple 同轨迹扫描：3311/13244 个集合，K=[0, 1, 5, 10, 25, 50]　　kind=triple　shard=3　nshard=4　gpu=cuda
- `01:02:18` ✔ **[KGRID] DONE**　triple shard 2/4 完成 3311 个集合　　实耗 5m51s　kind=triple　shard=2　n_queries=3311　n_rows=238392
- `01:04:20` ✔ **[KGRID] DONE**　triple shard 3/4 完成 3311 个集合　　实耗 5m58s　kind=triple　shard=3　n_queries=3311　n_rows=238392
- `01:04:26` ✔ **[KGRID] DONE**　triple shard 0/4 完成 3311 个集合　　实耗 7m59s　kind=triple　shard=0　n_queries=3311　n_rows=238392
- `01:04:53` ✔ **[KGRID] DONE**　triple shard 1/4 完成 3311 个集合　　实耗 8m26s　kind=triple　shard=1　n_queries=3311　n_rows=238392
- `01:04:55` ✔ **[SCHED] DONE**　全部 K 网格任务成功　　实耗 8m30s
- `01:05:12` ▶ **[ANALYSIS] START**　合并同保真度 K 网格，按分层总体口径评价并在 tune/test 隔离下选择调度
- `01:05:37` ◆ **[ANALYSIS] DECISION**　K0 top30 recall=82.7%；均衡分层方案 K0@30% -> K1@20% -> K50@10%，隔离 test recall@top10=65.0%。
- `01:07:35` ▶ **[RESCUE] START**　测试 K0/K5/K10 多榜互补与调参集线性组合
- `01:07:53` ◆ **[RESCUE] DECISION**　top10%=tune_weighted_rank_0_5_10, test recall 67.5%；top20%=single_K10, test recall 81.8%；top30%=single_K10, test recall 89.2%
- `01:08:44` ▶ **[RESCUE] START**　测试 K0/K5/K10 多榜互补与调参集线性组合
- `01:09:18` ◆ **[RESCUE] DECISION**　top10%=tune_weighted_rank_0_5_10, test recall 67.5%；top20%=single_K10, test recall 81.8%；top30%=single_K10, test recall 89.2%；top40%=tune_weighted_rank_0_5_10, test recall 90.9%；top50%=single_K10, test recall 91.6%；top60%=best_rank_0_10, test recall 93.4%
- `01:09:24` ▶ **[RESCUE] START**　测试 K0/K5/K10 多榜互补与调参集线性组合
- `01:09:58` ◆ **[RESCUE] DECISION**　top10%=tune_weighted_rank_0_5_10, test recall 67.5%；top20%=single_K10, test recall 81.8%；top30%=single_K10, test recall 89.2%；top40%=tune_weighted_rank_0_5_10, test recall 90.9%；top50%=single_K10, test recall 91.6%；top60%=best_rank_0_10, test recall 93.4%
- `01:11:52` ◆ **[SYNTHESIS] DECISION**　最终采用功能/保真度分层：K5快速、K10均衡、K25多榜高保障；拒绝K1硬筛和把Top10称为高召回池。
- `01:15:13` ✔ **[STAGED-RESCUE] DONE**　完成 K0/K5/K10 宽进 → K25 Top30 精排及多checkpoint救援模拟
- `01:15:52` ▶ **[RESCUE] START**　测试 K0/K5/K10 多榜互补与调参集线性组合
- `01:16:28` ◆ **[RESCUE] DECISION**　top10%=tune_weighted_rank_0_5_10, test recall 67.5%；top20%=single_K10, test recall 81.8%；top30%=single_K10, test recall 89.2%；top40%=tune_weighted_rank_0_5_10, test recall 90.9%；top50%=single_K10, test recall 91.6%；top60%=best_rank_0_10, test recall 93.4%
- `01:16:46` ◆ **[SYNTHESIS] DECISION**　最终采用功能/保真度分层：K5快速、K10均衡、K25多榜高保障；拒绝K1硬筛和把Top10称为高召回池。
- `01:19:47` ✔ **[VALIDATION] DONE**　原始K网格、K0/K25旧实现一致性、分层权重、指标范围与推荐方案反算全部通过
- `01:20:06` ▶ **[ANALYSIS] START**　合并同保真度 K 网格，按分层总体口径评价并在 tune/test 隔离下选择调度
- `01:20:28` ※ **[ANALYSIS] NOTE**　K0 top30 recall=82.7%；Top10单划分调度只作探索，不作为最终推荐，最终方案见 SYNTHESIS/STAGED-RESCUE。
