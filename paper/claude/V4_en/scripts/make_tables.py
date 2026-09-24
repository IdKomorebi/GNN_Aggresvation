# -*- coding: utf-8 -*-
"""V4_en 表格：全部从各编号实验目录的 CSV 生成 LaTeX 片段（tables/*.tex），正文用 \\input 引用，不手抄数字。"""
import os, sys, json
import numpy as np, pandas as pd
import names as N

HERE = os.path.dirname(os.path.abspath(__file__)); TB = os.path.join(HERE, "..", "tables"); os.makedirs(TB, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
R = lambda *p: os.path.join(REPO, *p)
GROUPS = [("RTS-GMLC", "DNN_Aggresvation111/outputs"), ("NEM", "DNN_Aggresvation112/outputs"),
          ("PJM", "DNN_Aggresvation116/groups/pjm_load/outputs"), ("PJM", "DNN_Aggresvation116/groups/pjm_gen_ic/outputs"),
          ("CAISO", "DNN_Aggresvation116/groups/caiso_load/outputs")]


def w(name, s):
    open(os.path.join(TB, name), "w", encoding="utf-8").write(s); print("写入", name)


def tab_risk():
    rows = []
    for ds, rel in GROUPS:
        S = pd.read_csv(R(rel, "analysis", "summary_targets.csv")); spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        z = np.load(R(rel, "D.npz")); keys = list(z["keys"]); V = np.load(R(rel, "V_official.npy")); s3 = np.array([k.count("|") <= 2 for k in keys])
        for c, (_, r) in enumerate(S.iterrows()):
            p = len(spec["cand"])
            rows.append(f"{N.target(r.目标)} & {p} & {r.最强单字段V:.2f} & {V[s3, c].max():.2f} & "
                        f"{int(r['τ0.5_单字段即危险'])} & {int(r['τ0.5_单看安全组合危险'])} & {int(r['τ0.5_危险小组合'])} & {int(r['τ0.5_单字段定级后仍暴露'])} & {int(r['τ0.5_最少扣留'])} & "
                        f"{int(r['τ0.7_单字段即危险'])} & {int(r['τ0.7_单看安全组合危险'])} & {int(r['τ0.7_危险小组合'])} & {int(r['τ0.7_单字段定级后仍暴露'])} & {int(r['τ0.7_最少扣留'])} \\\\")
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Combination risk that single-field grading misses. $V_1$: best single field; $V_{\le3}$: best set of at most three
fields; alone: fields with $V(\{i\})>\tau$; comb.: fields safe alone but $\tau$-critical with at most two background fields;
MUS: minimal unsafe sets of size $\le3$; left: MUS not broken after withholding every field that is unsafe alone; hit: minimum
number of fields whose withholding breaks every MUS.}\label{tab:risk}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{lrrr|rrrrr|rrrrr}
\toprule
 & & & & \multicolumn{5}{c|}{$\tau=0.5$} & \multicolumn{5}{c}{$\tau=0.7$}\\
Target & $p$ & $V_1$ & $V_{\le3}$ & alone & comb. & MUS & left & hit & alone & comb. & MUS & left & hit\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    w("tab_risk.tex", s)


def tab_decision():
    A = R("DNN_Aggresvation117/outputs/analysis"); rules = pd.read_csv(os.path.join(A, "decision_rules.csv"))
    ad = pd.read_csv(os.path.join(A, "withholding_strategies.csv"))

    def rec(rule, dl):
        d = rules[(rules.规则 == rule) & (rules.δ.isna() if dl is None else rules.δ == dl)]
        m = d.groupby("τ")[["召回", "精确"]].mean()
        return " & ".join(f"{m.loc[t, '召回']:.2f} & {m.loc[t, '精确']:.2f}" for t in (0.5, 0.7))

    def wh(lab, dl):
        d = ad[(ad.策略 == lab) & (ad.δ.isna() if dl is None else ad.δ == dl)]; out = []
        for t in (0.5, 0.7):
            e = d[d.τ == t]; out.append(f"{e.扣留数.mean():.1f} ({e.最优扣留数.mean():.1f}) & {100 * e.残余真危险组合.sum() / e.危险组合数.sum():.1f}\\%")
        return " & ".join(out)
    s = r"""\begin{table}[!tbp]
\centering\small
\caption{Decisions over the ten targets. Top: recall and precision of $\tau$-critical fields. Bottom: fields withheld (optimum
in parentheses) and share of true minimal unsafe sets left unbroken. Selection uses estimates only; retrained values are used
for certification and evaluation. $\delta$: conservative margin of the scan.}\label{tab:decision}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{$\tau=0.5$} & \multicolumn{2}{c}{$\tau=0.7$}\\
Flagging rule & recall & precision & recall & precision\\
\midrule
Single-field rule $V(\{i\})>\tau$ & """ + rec("单字段规则 V({i})>τ", None) + r""" \\
Scan on $\hat V$, $\delta=0.05$ & """ + rec("扫描标记（估计）", 0.05) + r""" \\
Scan + certify, $\delta=0$ & """ + rec("扫描—认证标记", 0.0) + r""" \\
Scan + certify, $\delta=0.05$ & """ + rec("扫描—认证标记", 0.05) + r""" \\
\midrule
Withholding strategy & withheld & left & withheld & left\\
\midrule
Single-field grading & """ + wh("单字段定级（扣留单字段越阈者）", None) + r""" \\
Hitting set of MUS on $\hat V$, $\delta=0.05$ & """ + wh("估计MUS最小命中集", 0.05) + r""" \\
Adaptive $\hat M^{(2)}$ greedy, $\delta=0.05$ & """ + wh("自适应M̂2贪心", 0.05) + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    w("tab_decision.tex", s)


def tab_ranking():
    A = R("DNN_Aggresvation117/outputs/analysis"); det = pd.read_csv(os.path.join(A, "detection.csv")); prot = pd.read_csv(os.path.join(A, "protection.csv"))
    order = [("pearson", "Pearson $r^2$"), ("spearman", "Spearman $\\rho^2$"), ("mi", "Mutual information"), ("dcor", "Distance correlation"),
             ("M0", "Single-field $V(\\{i\\})$"), ("loco", "LOCO"), ("perm", "Permutation importance"), ("sage", "SAGE"),
             ("graph", "Inference-graph propagation"), ("M2_est", "$\\hat M^{(2)}$ (amortized, no retraining)"),
             ("M2_cert", "$M^{(2)}$ scan--certify ($k=3$)"), ("M2", "$M^{(2)}$ exact")]
    rp = det.groupby(["τ", "方法"]).R精确率.mean().unstack(0); ex = prot.groupby(["τ", "方法"]).超出最优.mean().unstack(0)
    best = {t: rp[t].max() for t in (0.5, 0.7)}; bex = {t: ex[t].min() for t in (0.5, 0.7)}
    f = lambda v, b, hi=True: (f"\\textbf{{{v:.3f}}}" if abs(v - b) < 1e-9 else f"{v:.3f}") if hi else (f"\\textbf{{{v:.1f}}}" if abs(v - b) < 1e-9 else f"{v:.1f}")
    rows = [f"{lab} & {f(rp.loc[m, 0.5], best[0.5])} & {f(rp.loc[m, 0.7], best[0.7])} & {f(ex.loc[m, 0.5], bex[0.5], False)} & {f(ex.loc[m, 0.7], bex[0.7], False)} \\\\"
            for m, lab in order]
    s = r"""\begin{table}[!tbp]
\centering\small
\caption{Ranking quality over the ten targets. R-precision: share of true critical fields among the top-$c$ fields, $c$ being
the true number of critical fields (a threshold that favours the baselines). Excess: fields withheld in score order until no
MUS remains, minus the optimum.}\label{tab:ranking}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{R-precision} & \multicolumn{2}{c}{excess withheld}\\
Score & $\tau=0.5$ & $\tau=0.7$ & $\tau=0.5$ & $\tau=0.7$\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    w("tab_ranking.tex", s)


def tab_ablation():
    V = pd.read_csv(R("DNN_Aggresvation118/outputs/analysis/variants_by_target.csv"))
    order = [("lin", "Linear readout on $x\\odot m$ (no backbone)"), ("poly", "Readout on $[x, x^2]$ (no backbone)"),
             ("polyS", "In-set quadratic dictionary (no backbone)"), ("head3", "Shared output head (3 seeds)"),
             ("rand1", "Untrained random backbone + readout"), ("masknone1", "Pre-training without masks"),
             ("maskbern1", "Bernoulli(0.5) masks"), ("phionly1", "Readout on $\\phi$ only"), ("nofix1", "No numerical safeguards"),
             ("D1", "\\textbf{Ours}, one seed"), ("E", "\\textbf{Ours}, three seeds (main)")]
    cols = ["V误差", "边际误差", "M2误差", "M2偏差", "M2排序Spearman", "认证前1", "认证前3"]
    m = V.groupby("变体")[cols].mean(); n = V.groupby("变体").size()
    rows = []
    for v, lab in order:
        if v not in m.index:
            continue
        r = m.loc[v]
        rows.append(f"{lab} & {r.V误差:.3f} & {r.边际误差:.3f} & {r.M2误差:.3f} & {r.M2偏差:+.3f} & {r.M2排序Spearman:.3f} & {r.认证前1:.3f} & {r.认证前3:.3f} \\\\")
        if v in ("head3", "nofix1"):
            rows.append("\\midrule")
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Estimator comparison and ablation, averaged over the ten targets and all sets of size $\le 3$. Ablations use one
seed and should be compared with ``ours, one seed''. Cert.@$k$: certified $L^{(2)}$ over exact $M^{(2)}$ when $k$ backgrounds per
field are retrained.}\label{tab:ablation}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccc}
\toprule
Estimator & MAE $V$ & MAE $\Delta$ & MAE $M^{(2)}$ & bias $M^{(2)}$ & Spearman & cert.@1 & cert.@3\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    w("tab_ablation.tex", s)


def tab_backbone():
    B = pd.read_csv(R("DNN_Aggresvation116/outputs/analysis/backbone_compare_mean.csv"), index_col=0)
    lab = {"本组主干": "Per-group (targets supervised)", "全列主干": "All-column (masked reconstruction)", "留目标主干": "All-column, target held out"}
    rows = [f"{lab[b]} & {100 * r.死神经元比例:.1f}\\% & {r.有效维度:.2f} & {r.V绝对误差:.3f} & {100 * r.系统偏差占比:.0f}\\% & {r['边际Δ绝对误差']:.3f} & {r.M2误差:.3f} & {r.M2偏差:+.3f} & {r.认证前1:.3f} & {r.认证前3:.3f} \\\\"
            for b, r in B.iterrows()]
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Backbone scope on the four PJM/CAISO targets (same readout). Inactive: hidden units with zero variance on the training
rows; eff.\ dim: participation ratio of the feature covariance; common-mode: $|\overline{e}|/\overline{|e|}$ of the $V$ error $e$.}\label{tab:backbone}
\setlength{\tabcolsep}{2.8pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccc}
\toprule
Backbone & inactive & eff.\ dim & MAE $V$ & common-mode & MAE $\Delta$ & MAE $M^{(2)}$ & bias $M^{(2)}$ & cert.@1 & cert.@3\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    w("tab_backbone.tex", s)


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["risk", "decision", "ranking", "ablation", "backbone"]):
        globals()[f"tab_{t}"]()
