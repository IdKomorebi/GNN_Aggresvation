# DNN_Aggresvation103 工作记录

## 2026-09-18（P0-3 + RQ-A1/A2/A3/A5）

1. 按 102 号诊断结论，补算**全量单目标 DNN 真值**（每数据集 12 目标 × 11,521 集合，三卡约 1.6 小时），与 100 号的多目标 DNN、梯度提升树合成正式攻击器族真值；
2. CAISO 补训 none / bern50 掩码分布 backbone（PJM 复用 75 号同名权重）；
3. 在正式真值上评估全部基线、我们的模型与消融。

## 过程记录

- 100 号的 `run_est.py` 不支持 raw / raw+x² / 随机特征（其特征构造只覆盖带 φ 的模型），首次队列这三项报 `torch.cat(): expected a non-empty list`；在本号新增 `scripts/run_est_base.py` 补跑，未改动 100 号脚本。
- 全部估计仍按 FINAL_PROTOCOL：α 在 train 内部 fit/val 选，test 只报告。
