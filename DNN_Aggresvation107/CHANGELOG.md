# DNN_Aggresvation107　总结与论文规划（P0 收束）

> 本号不跑训练、不占 GPU，只做三件事：汇总 100–106 号成色、钉死遗留问题、给出论文写作方案。
> 详细内容见 `outputs/analysis/` 下三份文档；图见 `figures/`。

## 产物

| 文件 | 内容 |
| --- | --- |
| `outputs/analysis/SUMMARY.md` | 逐号实验成色与全部关键数字（145 行） |
| `outputs/analysis/GAPS.md` | 遗留问题 A/B/C 三档 + 协议偏离登记 |
| `outputs/analysis/PAPER_PLAN.md` | 章节承载、图表清单、**句子级的可主张/不可主张** |
| `outputs/analysis/acceptance.csv` | 三区 10 条 + 二区 7 条验收对照 |
| `outputs/analysis/figure_index.csv` | 论文 Fig1–8 / Table I–V → 现有图文件映射 |
| `outputs/analysis/folder_check.csv` | 100–107 号文档与图完整性核查 |
| `figures/fig1_experiment_matrix.png` | 实验编号 × 10 个 RQ 的覆盖矩阵 |
| `figures/fig2_core_evidence.png` | 三条核心证据（闭式读出 / M 的定位 / 发布审查） |
| `figures/fig3_acceptance_gaps.png` | 验收达成条形 + 遗留问题优先级散点 |

## 结论一：实验成色（10 个 RQ 全部有主证据）

- **最强的三条**：
  1. 共享输出头 → 冻结 φ + 子集闭式读出，V-MAE 降 3.2×（PJM 0.0892→0.0276）/ 4.6×（CAISO 0.0705→0.0153），危险漏判 33.6%/28.7% → 12.6%/3.8%（RQ-A2，103 号）；
  2. τ=0.7、K=2 下 PJM 35/41、CAISO 38/41 个字段「单看安全、加 ≤2 个背景字段即 critical」（RQ-B5，106 号）；
  3. 发布审查中单字段规则放行 19.1%/56.4% 的危险组合，**逐阶预算 U^(2) 零危险放行**且误拒 49.6%/32.8%（RQ-B5，106 号）。
- **两条必须诚实写出的不利结果**：
  1. 单字段精确 M^(0) 是很强的排序基线（PJM τ=0.7 PR-AUC 0.877 vs 估计 M^(1) 0.888，仅差 0.011）⟹ 本文增益要落在**决策层**而非排序层；
  2. p=41、K≤2 时并行专用重训 10.9 分钟就能精确算完 ⟹ 摊销模型**不是唯一解**，只能定位为高查询量/大 p/反复重算场景的加速器。

## 结论二：遗留问题

- **A 档（必须处理）**：A1 PJM 的 M 保真不足（K=2 recall 0.754、认证下界比 0.822）；A2 CAISO 元训练 φ 缺失；A3 K=3 无正式三攻击器口径。
  三条的**推荐处置都不需要重训模型**：A1 改用 [L,U] 区间口径 + top-3 witness 重训认证（约 2 分钟）；A2 降级到附录；A3 正文只主张 K≤2。
- **B 档（提档抓手）**：p≥100 大字段空间（对 Q1 动机最关键）、攻击器能力 sensitivity（成本最低）、XGBoost/LightGBM（要做就现在做，会改变所有真值）。
- **C 档（写局限）**：时间外推未评估（用户决定）、witness 不可跨数据复现（101 号一致率 0.12–0.37）、上界依赖可交换性假设、大背景只有下界、τ 外生。

## 结论三：验收对照

三区 10 条：**8 条达成、1 条已论证偏离（single→multi）、1 条按用户决定取消（时间块）**。
（原「M 保真部分达成」一条已由 108 号的区间口径升级为达成。）
二区 7 条：2 条达成、1 条部分达成、1 条取消、3 条未做。⟹ **三区可投；冲二区还差 B 档的 2–3 项。**

> **2026-09-18 后续**：
> - GAPS 的 A 档三条已在 **108 号**全部处置（零 GPU）。
> - GAPS 的 B2（attacker capability sensitivity）已在 **109 号**完成（17.5 分钟三卡），
>   并额外产出 winner's curse 机制：M^(K) 在弱攻击者下被系统性抬高，选择增益 ∝ 候选背景数。
> - `GAPS.md` / `PAPER_PLAN.md` / `acceptance.csv` / `figure_index.csv` 均已回填。
> - 写作前动作清单 1–5 清空，**论文可以开写**；二区验收 3 条达成。

## 完整性核查（folder_check.csv）

100–107 号全部具备 `base.yaml` / `CHANGELOG.md` / `RUNLOG.md` / `WORKLOG.md` 与 `figures/`。
本轮修复：**104 号原缺 `RUNLOG.md`**，已按产物时间戳补记（该号脚本未接入 runlog 模块）。
