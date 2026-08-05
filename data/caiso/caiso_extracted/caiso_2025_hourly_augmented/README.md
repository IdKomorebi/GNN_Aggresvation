# CAISO 2025 UTC 小时对齐增补数据包

主数据：`caiso_2025_hourly_aligned_augmented.csv.gz`

这是 GPT Work 原 CAISO 包的可追溯增补版。原表 163 个数值字段保持不变，
并新增 142 个经官方 OASIS 下载和覆盖校验的字段。

## 主要文件

- `caiso_2025_hourly_aligned_augmented.csv.gz`：最终小时宽表
- `column_dictionary.csv`：机器可读逐列字典
- `field_official_mapping_and_explanations.md`：官网对应与中文解释
- `source_manifest.csv`：原包与本次新增的完整响应清单
- `supplemental_source_manifest.csv`：仅本次新增响应
- `source_lineage.csv`：旧包和浏览器控制样本的 SHA-256
- `supplemental_interval_coverage.csv`：EIM 15 分钟到小时覆盖
- `column_validation_stats.csv`、`validation_report.md`、`validation_summary.json`
- `caiso_tls_control_note.md`：本机证书链异常与双通道控制说明
- `build_caiso_2025_augmented.py`：本增补脚本
- `build_caiso_2025_hourly_original.py`：GPT Work 原构建脚本
- `deliverable_checksums.csv`：包内文件 SHA-256

原始补充 ZIP 缓存在本任务目录的 `work/caiso_supplement_raw/`，未重复塞入交付
ZIP；每个原文件的 URL、字节数、ZIP SHA-256 和内部 CSV SHA-256 均在清单中。
