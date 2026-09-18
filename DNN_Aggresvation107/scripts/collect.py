# -*- coding: utf-8 -*-
"""107-1：跨编号完整性核查 + 验收标准对照 + 论文图表索引。不训练、不占 GPU。"""
from pathlib import Path
import csv, sys
ROOT = Path(__file__).resolve().parents[2]
A = Path(__file__).resolve().parents[1] / "outputs/analysis"; A.mkdir(parents=True, exist_ok=True)

NUMS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
ROLE = {100: "全规模精确 M + 独立要素（基线号）", 101: "独立数据认证 C1–C6", 102: "P0-2 single vs multi",
        103: "P0-3 正式真值 + 基线/消融", 104: "P0-4 正式 M 全表", 105: "P0-5 领域 baseline",
        106: "P0-7 分级表 + 发布审查", 107: "总结与论文规划", 108: "A 档收尾：区间口径/元训练定位/K=3 口径", 109: "P1 攻击者辅助标签量 sensitivity"}

rows = []
for n in NUMS:
    d = ROOT / f"DNN_Aggresvation{n}"
    r = {"编号": n, "作用": ROLE[n]}
    for f in ["base.yaml", "CHANGELOG.md", "RUNLOG.md", "WORKLOG.md"]:
        p = d / f
        r[f] = f"{p.read_text(encoding='utf-8').count(chr(10))+1} 行" if p.exists() else "缺失"
    figs = sorted((d / "figures").glob("*.png")) if (d / "figures").exists() else []
    r["figures"] = len(figs)
    r["csv产物"] = len(list((d / "outputs").rglob("*.csv"))) if (d / "outputs").exists() else 0
    r["报告md"] = len(list((d / "outputs").rglob("*.md"))) if (d / "outputs").exists() else 0
    r["齐全"] = "是" if all(r[f] != "缺失" for f in ["base.yaml", "CHANGELOG.md", "RUNLOG.md", "WORKLOG.md"]) and r["figures"] > 0 else "否"
    rows.append(r)
with (A / "folder_check.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---- 验收标准（主线计划 §21）----
ACC = [
    ("三区", 1, "PJM + CAISO 两个独立数据集", "达成", "FINAL_PROTOCOL §1；全部 102–106 双数据集并行"),
    ("三区", 2, "single-target 主模型完成", "偏离（已论证）", "102 号：single 的 V-MAE/M-MAE 全面劣于 multi 且预学习贵 12×；协议 §12 已改判 multi-target 为正文主版本，single 降为消融"),
    ("三区", 3, "exact low-order truth", "达成", "103 号：K≤2 全枚举 + k3 全集合，三攻击器 max + 单调闭包"),
    ("三区", 4, "通用模型对 M 有稳定准确度", "达成", "108 号区间口径：α=0.01 下区间覆盖 98%；top-3 认证下界比 0.908/0.974；保守判定 τ0.7 recall 0.994/0.967（原 104 号点估计 recall 0.75 的弱点已消除）"),
    ("三区", 5, "领域 baseline 完整", "达成", "105 号：Pearson/Spearman/NMI/SAGE/置换重要性/单字段 M^(0)"),
    ("三区", 6, "多攻击器", "达成", "103 号：单目标 DNN ∪ 多目标 DNN ∪ HistGBR，val 选择"),
    ("三区", 7, "时间块独立认证", "已取消", "用户决定（2026-09-18）不做时间划分；改由 101 号随机对半独立数据认证承担"),
    ("三区", 8, "完整字段风险表", "达成", "104 号 M_table_official.csv 2,952 行；106 号 grading_table.csv"),
    ("三区", 9, "至少一个真实发布审查 case", "达成", "106 号：PJM total_gen + CAISO actual_load，各 2,000 个规模 3 集合、6 条规则"),
    ("三区", 10, "理论无明显过度 claim", "达成", "99 号反例库 C1–C5；理论文档 v2 已修正 v1 三处命题"),
    ("二区", 1, "真实大字段空间 p≥100", "未做", "当前 p=41；只有按集合数外推的成本估计，无真实数据"),
    ("二区", 2, "第三个电力市场数据集", "未做", ""),
    ("二区", 3, "辅助标签量 / attacker capability sensitivity", "达成", "109 号：n_aux∈{10%,25%,50%,100%} 双数据集 k3 全枚举；V 单调、★M^(1)/M^(2) 因 winner's curse 小样本反超、10% 档审计危险放行 25.6%/13.2%"),
    ("二区", 4, "很强的时间外泛化", "已取消", "用户决定不做时间划分；论文局限中写明未评估时间外推"),
    ("二区", 5, "更严格的 calibrated risk bound", "达成", "108 号：正式真值上重新校准的 [L,U]，α=0.01 覆盖 98%/98.6%；top-3 认证下界比 0.908/0.974；101 号独立数据覆盖 94–97%"),
    ("二区", 6, "τ-critical 检测显著稳定优于传统方法", "部分达成", "105 号：估计 M 在 6 个 (数据集,τ) 格中 5 个最优；但单字段精确 M^(0) 在 PJM τ=0.7 接近持平（0.877 vs 0.888）"),
    ("二区", 7, "代码与数据处理流程完整可复现", "达成", "协议冻结 + 数据指纹 + 各号 RUNLOG/CHANGELOG"),
]
with (A / "acceptance.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["档位", "序号", "条目", "状态", "证据"]); w.writerows(ACC)

# ---- 论文图表 → 现有产物 ----
FIG = [
    ("Fig. 1", "Overall Framework：S→V̂→M^(K)→分级+witness", "待画（示意图）", "新画，不依赖实验"),
    ("Fig. 2", "Target-specific Universal Model 两阶段", "待画（示意图）", "新画，不依赖实验"),
    ("Fig. 3", "V Fidelity：预测 V vs 专用重训 V", "DNN_Aggresvation100/figures/fig1_*.png + 103 报告 B/C 表", "可直接改用 103 正式真值重绘"),
    ("Fig. 4", "M Fidelity：估计 M vs 精确 M", "DNN_Aggresvation104/figures/fig1_M_fidelity.png", "现成"),
    ("Fig. 4b", "M 的 [L,U] 区间与覆盖率", "DNN_Aggresvation108/figures/fig1_interval_bounds.png + fig2_critical_interval.png", "现成，建议并入 Fig. 4"),
    ("Fig. 5", "K Escalation M^0→M^1→M^2（正文只到 K=2）", "DNN_Aggresvation104/figures/fig3_escalation_official.png", "现成"),
    ("Fig. 5 附录", "K=3 口径对照（攻击器口径差 > 阶数增量）", "DNN_Aggresvation108/figures/fig3_escalation_metatrain.png", "现成，仅附录"),
    ("Fig. 6", "Witness Cases（PJM + CAISO）", "DNN_Aggresvation106/figures/fig2_grading_table.png + 104 top-3 witness 列", "需挑选案例重绘"),
    ("Fig. 7", "Release Audit Case", "DNN_Aggresvation106/figures/fig1_release_audit.png", "现成，正文核心图"),
    ("Fig. 8", "Efficiency / Scalability", "DNN_Aggresvation100/figures/fig7_*.png", "现成；需按'通用模型非必需'的诚实口径改标题"),
    ("Table I", "Dataset and Threat-model Settings", "paper/FINAL_PROTOCOL.md §1–§4", "整理成表"),
    ("Table II", "Universal-model Baselines and Ablations", "DNN_Aggresvation103/outputs/analysis/103_B.csv", "现成"),
    ("Table III", "M Estimation Accuracy（区间口径）", "DNN_Aggresvation104/M_table_summary.csv + DNN_Aggresvation108/108_A1_bounds.csv、108_A1_coverage.csv", "现成"),
    ("Table IV", "Security Metric Comparison", "DNN_Aggresvation105/outputs/analysis/105_critical_detection.csv", "现成"),
    ("Table V", "Field-level Risk and Witness Examples", "DNN_Aggresvation106/outputs/analysis/grading_table.csv", "现成，节选 10–15 行"),
    ("Table VI", "Attacker-capability Sensitivity", "DNN_Aggresvation109/outputs/analysis/109_B_M.csv、109_D_release.csv", "现成"),
    ("Fig. 9", "n_aux sensitivity 与 winner's curse", "DNN_Aggresvation109/figures/fig3_naux_release.png、fig4_winners_curse.png", "现成"),
]
with (A / "figure_index.csv").open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["编号", "内容", "现有产物", "状态"]); w.writerows(FIG)

print("folder_check / acceptance / figure_index 已写出")
for r in rows:
    print(f"  {r['编号']}  齐全={r['齐全']}  figs={r['figures']}  " + " ".join(f"{k}={r[k]}" for k in ["base.yaml","CHANGELOG.md","RUNLOG.md","WORKLOG.md"]))
