# -*- coding: utf-8 -*-
"""RQ-B1：各字段指标在"τ-critical 字段检测"上的表现（正式真值，K=2、τ=0.7；另报 τ=0.5/0.9）。
参与比较：Pearson、Spearman、NMI、单字段 M^(0)、全量模型 SAGE、全量模型置换重要性、估计 M^(1)、估计 M^(2)、精确 M^(2)（上界参考）。
指标：PR-AUC、按同等"高风险字段预算"下的 recall、false-safe rate。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parents[1]; REPO = ROOT.parent; R104 = REPO / "DNN_Aggresvation104"
A = ROOT / "outputs/analysis"; rep = ["# 105 号：领域/归因 baseline 与 τ-critical 检测（RQ-B1）\n"]
T = pd.read_csv(R104 / "outputs/analysis/M_table_official.csv")
rows, prrows = [], []
for ds in ["pjm", "caiso"]:
    B = pd.read_csv(A / f"{ds}_field_baselines.csv")
    t2 = T[(T.数据集 == ds) & (T.K == 2)].set_index(["目标", "字段"])
    t1 = T[(T.数据集 == ds) & (T.K == 1)].set_index(["目标", "字段"])
    t0 = T[(T.数据集 == ds) & (T.K == 0)].set_index(["目标", "字段"])
    B = B.set_index(["目标", "字段"]).loc[t2.index]
    scores = {"Pearson |r|": B.pearson, "Spearman |ρ|": B.spearman, "NMI": B.nmi,
              "单字段 M^(0)（精确）": t0.exact_M, "全量模型 SAGE": B.sage, "全量模型置换重要性": B.perm_importance,
              "估计 M^(1)": t1.est_M_集成, "估计 M^(2)": t2.est_M_集成, "精确 M^(2)（参考上界）": t2.exact_M}
    for tau in [0.5, 0.7, 0.9]:
        y = t2[f"critical_τ{tau}"].values.astype(int)
        if y.sum() == 0 or y.sum() == len(y): continue
        for nm, s in scores.items():
            s = np.nan_to_num(np.asarray(s, float))
            ap = average_precision_score(y, s)
            k = int(y.sum()); top = np.argsort(-s)[:k]
            rec = y[top].sum() / y.sum()
            rows.append(dict(数据集=ds, tau=tau, 指标=nm, PR_AUC=ap, 同预算recall=rec, false_safe=1 - rec, 正例数=int(y.sum()), 总数=len(y)))
        if tau == 0.7:
            for nm, s in scores.items():
                s = np.nan_to_num(np.asarray(s, float)); o = np.argsort(-s); yy = y[o]
                prec = np.cumsum(yy) / np.arange(1, len(yy) + 1); rec = np.cumsum(yy) / yy.sum()
                for q in range(0, len(yy), 5): prrows.append(dict(数据集=ds, 指标=nm, recall=rec[q], precision=prec[q]))
R = pd.DataFrame(rows); R.to_csv(A / "105_critical_detection.csv", index=False)
pd.DataFrame(prrows).to_csv(A / "105_pr_curves.csv", index=False)
piv = R[R.tau == 0.7].pivot_table(index="指标", columns="数据集", values=["PR_AUC", "同预算recall"]).round(4)
rep.append("## τ=0.7、K=2（正式真值）\n\n" + piv.to_markdown())
rep.append("\n## 全部阈值\n\n" + R.pivot_table(index=["指标"], columns=["数据集", "tau"], values="PR_AUC").round(3).to_markdown())
(A / "report105.md").write_text("\n\n".join(rep), encoding="utf-8"); print("\n\n".join(rep))

# ---------- 补充：只在"组合才危险"的字段上评估（真值 K=2 critical 且 K=0 不 critical）
rows2 = []
for ds in ["pjm", "caiso"]:
    B = pd.read_csv(A / f"{ds}_field_baselines.csv")
    t2 = T[(T.数据集 == ds) & (T.K == 2)].set_index(["目标", "字段"]); t1 = T[(T.数据集 == ds) & (T.K == 1)].set_index(["目标", "字段"])
    t0 = T[(T.数据集 == ds) & (T.K == 0)].set_index(["目标", "字段"]); B = B.set_index(["目标", "字段"]).loc[t2.index]
    for tau in [0.5, 0.7]:
        comb = t2[f"critical_τ{tau}"].values & ~t0[f"critical_τ{tau}"].values   # 组合才危险
        safe = ~t2[f"critical_τ{tau}"].values                                    # 任何背景下都不危险
        keep = comb | safe; y = comb[keep].astype(int)
        if y.sum() == 0: continue
        scores = {"Pearson |r|": B.pearson, "Spearman |ρ|": B.spearman, "NMI": B.nmi, "单字段 M^(0)（精确）": t0.exact_M,
                  "全量模型 SAGE": B.sage, "全量模型置换重要性": B.perm_importance, "估计 M^(1)": t1.est_M_集成,
                  "估计 M^(2)": t2.est_M_集成, "精确 M^(2)（参考上界）": t2.exact_M}
        for nm, s in scores.items():
            s = np.nan_to_num(np.asarray(s, float))[keep]
            ap = average_precision_score(y, s); k = int(y.sum()); rec = y[np.argsort(-s)[:k]].sum() / y.sum()
            rows2.append(dict(数据集=ds, tau=tau, 指标=nm, PR_AUC=ap, 同预算recall=rec, 正例数=int(y.sum()), 候选数=int(keep.sum())))
R2 = pd.DataFrame(rows2); R2.to_csv(A / "105_combination_only.csv", index=False)
txt = "\n\n## 只在“组合才危险”的字段上（真值 K=2 critical 且 K=0 不 critical；对照组为任何背景下都不危险的字段）\n\n" + \
      R2[R2.tau == 0.7].pivot_table(index="指标", columns="数据集", values=["PR_AUC", "同预算recall"]).round(4).to_markdown() + \
      "\n\n正例数 / 候选数：" + "；".join(f"{r.数据集} τ={r.tau} {r.正例数}/{r.候选数}" for _, r in R2.drop_duplicates(["数据集", "tau"]).iterrows())
open(A / "report105.md", "a", encoding="utf-8").write(txt); print(txt)
