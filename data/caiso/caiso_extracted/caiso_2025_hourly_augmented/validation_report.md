# CAISO 2025 增补版验证报告

- 行数：8,760
- 总列数：307（2 个时间字段 + 305 个数值字段）
- 本次增加数值字段：142
- UTC 重复时间戳：0
- 数值缺失单元格：10,511
- 插值：否
- 新增 OASIS 响应：108
- Chrome 与命令行控制样本逐字节一致：True

## 分报表新增字段

- PRC_AS：26
- AS_RESULTS（仅 AS_MW）：26
- ENE_SLRS：20
- ENE_EIM_TRANSFER_TIE（仅 CISO）：70

## 新增字段缺失拆分

- PRC_AS：0
- AS_RESULTS（仅 AS_MW）：0
- ENE_SLRS：8,664
- ENE_EIM_TRANSFER_TIE（仅 CISO）：1,750

`ENE_SLRS` 的缺失几乎全部来自
`dam_schedule__mw__import__tac_ncntr`：该维度只在官方文件出现 96 小时，
不是下载失败；保留稀疏列是为了不擅自删减官网维度。EIM 每列只在四个
15 分钟记录齐全时生成小时值。

## 已执行的检查

1. 最终 UTC 轴严格为 8,760 个递增且唯一的小时。
2. 每个输出字段与 `column_dictionary.csv` 一一对应且无重名。
3. 所有下载均校验 ZIP、内部 CSV、响应 SHA-256 和内部 CSV SHA-256。
4. PRC_AS、AS_RESULTS、ENE_SLRS 的原始键在冲突检测后才 pivot。
5. EIM transfer 只在同一 UTC 小时四个 15 分钟点齐全时取算术均值；
   不齐全的组保留为空，逐列覆盖见 `supplemental_interval_coverage.csv`。
6. 原包不重算、不改写；其 ZIP 与主表 SHA-256 记录在 `source_lineage.csv`。
7. 逐列非空数、缺失、范围、均值和标准差见 `column_validation_stats.csv`。
