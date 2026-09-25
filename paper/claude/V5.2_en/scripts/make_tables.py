# -*- coding: utf-8 -*-
"""V5.1_en 表格（估计器固定为 125 号的新估计器；决策、排序取 125 号重算的 d117）：全部从各编号实验目录的 CSV 生成 LaTeX 片段（tables/*.tex），正文用 \\input 引用，不手抄数字。"""
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
    A = R("DNN_Aggresvation125/outputs/d117"); rules = pd.read_csv(os.path.join(A, "decision_rules.csv"))
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
    A = R("DNN_Aggresvation125/outputs/d117"); det = pd.read_csv(os.path.join(A, "detection.csv")); prot = pd.read_csv(os.path.join(A, "protection.csv"))
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
    Sel = pd.read_csv(R("DNN_Aggresvation125/outputs/select/candidates_by_target.csv")).rename(columns={"V误差_3": "V误差", "M2排序": "M2排序Spearman"})
    cols = ["V误差", "边际误差", "M2误差", "M2偏差", "M2排序Spearman", "认证前1", "认证前3"]
    m118 = V.groupby("变体")[cols].mean()
    m125 = Sel.groupby(["候选", "实现"])[cols].mean().groupby("候选").mean()   # C2 为三个种子实现的平均
    import glob   # 每集合耗时：126 号统一计时（同一空闲 GPU、同一批集合、批大小 8、等价快速读出），五组平均
    tm = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation126/outputs/time_fast_*.csv"))]).groupby("方法").每集合ms.mean()
    tkey = {"lin": "lin", "poly": "poly", "polyS": "polyS", "head3": "head3", "rand1": "rand1", "D1": "D1", "E_old": "E", "C1": "C1", "C2": "C2", "C3": "C3"}
    blocks = [[("lin", "Linear readout on $x\\odot m$ (no backbone)"), ("poly", "Readout on $[x, x^2]$ (no backbone)"),
               ("polyS", "In-set quadratic dictionary (no backbone)"), ("head3", "Shared output head (3 seeds)")],
              [("rand1", "Untrained random backbone"), ("masknone1", "Pre-training without masks"), ("maskbern1", "Bernoulli(0.5) masks"),
               ("phionly1", "Readout on $\\phi$ only"), ("nofix1", "No numerical safeguards"), ("D1", "Target-only pre-training, one seed (reference)")],
              [("E_old", "Target-only, three seeds"), ("C1", "Reconstruction, three seeds, clip"),
               ("C2", "Reconstruction $\\oplus$ untrained, one seed, clip"), ("C3", "\\textbf{Ours}: reconstruction $\\oplus$ untrained, three seeds, clip")]]
    rows = []
    for k, blk in enumerate(blocks):
        for v, lab in blk:
            r = m125.loc[v] if k == 2 else m118.loc[v]
            t = tm.get(tkey.get(v, ""), np.nan); ts = "--" if t != t else (f"{t:.1f}" if t < 10 else f"{t:.0f}")
            rows.append(f"{lab} & {r.V误差:.3f} & {r.边际误差:.3f} & {r.M2误差:.3f} & {r.M2偏差:+.3f} & {r.M2排序Spearman:.3f} & {r.认证前1:.3f} & {r.认证前3:.3f} & {ts} \\\\")
        if k < 2:
            rows.append("\\midrule")
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Estimator comparison and ablation, averaged over the ten targets and all sets of size $\le 3$. Middle block: one-seed
ablations of a target-only backbone, to be compared with its reference row. Bottom block: pre-training objective and feature map
(Section~\ref{sec:res-est}); the one-seed row is the mean over three seeds. Cert.@$k$: certified $L^{(2)}$ over exact $M^{(2)}$ when
$k$ backgrounds per field are retrained. ms: time per field set on one idle GPU (same 200 sets per data set, batch 8, mean over data
sets); three-seed rows solve three readouts. Rows marked -- cost the same as their reference.}\label{tab:ablation}
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Estimator & MAE $V$ & MAE $\Delta$ & MAE $M^{(2)}$ & bias $M^{(2)}$ & Spearman & cert.@1 & cert.@3 & ms\\
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


def tab_masking():
    A = R("DNN_Aggresvation122/outputs/analysis"); V = pd.read_csv(os.path.join(A, "variants_by_target.csv"))
    Dd = pd.read_csv(os.path.join(A, "decisions.csv"))
    Sel = pd.read_csv(R("DNN_Aggresvation125/outputs/select/candidates_by_target.csv"))
    m125 = Sel.groupby(["候选", "实现"]).mean(numeric_only=True).groupby("候选").mean()
    diag = V.groupby("变体")["有效维度"].mean()

    def from122(v):
        g = V[V.变体 == v].mean(numeric_only=True)
        rc = Dd[(Dd.估计器 == v) & (Dd.规则 == "扫描—认证标记") & (Dd.δ == 0.0) & (Dd.τ == 0.5)].召回.mean() if v in set(Dd.估计器) else np.nan
        return [g.V误差_3, g.V误差_4, g.M2误差, g.Γ2误差, g.认证前1, g.背景召回前3, rc]

    def from125(c):
        g = m125.loc[c]; return [g.V误差_3, g.V误差_4, g.M2误差, g.Γ2误差, g.认证前1, g.背景召回前3, g["召回_τ0.5_δ0.0"]]
    rows = [("Untrained backbone", diag["random"], from122("random")), ("Untrained backbone, clip", diag["random"], from122("random_cy")),
            ("Target-only, $\\times3$", diag["uniform"], from125("E_old")), ("Target-only, $\\times3$, clip", diag["uniform"], from122("uniformE_cy")),
            ("Reconstruction, $\\times3$", diag["recon"], from122("reconE")), ("Reconstruction, $\\times3$, clip", diag["recon"], from125("C1")),
            ("Reconstruction $\\oplus$ untrained, $\\times1$, clip", None, from125("C2")),
            ("\\textbf{Ours}: reconstruction $\\oplus$ untrained, $\\times3$, clip", None, from125("C3"))]
    body = []
    for lab, ed, v in rows:
        eds = "--" if ed is None else f"{ed:.2f}"; rcs = f"{v[6]:.2f}" if v[6] == v[6] else "--"
        body.append(f"{lab} & {eds} & {v[0]:.4f} & {v[1]:.4f} & {v[2]:.4f} & {v[3]:.4f} & {v[4]:.3f} & {v[5]:.2f} & {rcs} \\\\")
    s = r"""\begin{table}[!tbp]
\centering\small
\caption{Pre-training objective and feature map over the ten targets (mean). Eff.\ dim.: participation ratio of the hidden
representation. $\times3$: predictions of three pre-training seeds averaged ($\times1$: mean over three single-seed estimators).
Clip: predictions clipped to the training range of the target. Top-3 context: share of fields whose worst background is among the
three best-scanned ones. Crit.\ recall: scan--certify recall of $\tau$-critical fields at $\tau=0.5$, $\delta=0$ (precision is
1.00 throughout).}\label{tab:masking}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}
\toprule
Backbone & eff.\ dim. & $V$, $|S|\le3$ & $V$, $|S|=4$ & $M^{(2)}$ & $\Gamma^{(2)}$ & cert.\ top-1 & top-3 context & crit.\ recall\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    w("tab_masking.tex", s)

def tab_design():
    """设计选择：主干 / 特征映射、种子用法、数值防护、预训练掩码。来源：118（只预测目标主干的一种子消融、无主干读出、三种子拼接）、
    122（未训练 + 截断、只预测目标 ×3 + 截断、重建 ×3 未截断）、125（旧 E、C1、C2、C3）；耗时：126 号快速实现（五组平均）。"""
    import glob
    cols = ["V", "M2", "rho", "c1", "c3"]
    V118 = pd.read_csv(R("DNN_Aggresvation118/outputs/analysis/variants_by_target.csv")).rename(
        columns={"V误差": "V", "M2误差": "M2", "M2排序Spearman": "rho", "认证前1": "c1", "认证前3": "c3"})
    V122 = pd.read_csv(R("DNN_Aggresvation122/outputs/analysis/variants_by_target.csv")).rename(
        columns={"V误差_3": "V", "M2误差": "M2", "M2排序": "rho", "认证前1": "c1", "认证前3": "c3"})
    S = pd.read_csv(R("DNN_Aggresvation125/outputs/select/candidates_by_target.csv")).rename(
        columns={"V误差_3": "V", "M2误差": "M2", "M2排序": "rho", "认证前1": "c1", "认证前3": "c3"})
    m118 = V118.groupby("变体")[cols].mean(); m122 = V122.groupby("变体")[cols].mean()
    m125 = S.groupby(["候选", "实现"])[cols].mean().groupby("候选").mean()
    six = V118[V118.变体 == "C3"][["数据", "目标"]]
    e6 = S[S.候选 == "E_old"].merge(six, on=["数据", "目标"])[cols].mean()
    tm = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation126/outputs/time_fast_*.csv"))]).groupby("方法").每集合ms.mean()
    T = lambda k: "--" if k is None else (f"{tm[k]:.1f}" if tm[k] < 10 else f"{tm[k]:.0f}")
    groups = [
        ("Feature map (three seeds averaged, all safeguards)", [
            ("None: readout on $[x_S, x_S^2]$", m118.loc["poly"], "poly"),
            ("Untrained network (one seed)", m122.loc["random_cy"], "rand1"),
            ("Target-only pre-training", m122.loc["uniformE_cy"], "E"),
            ("Reconstruction pre-training", m125.loc["C1"], "C1"),
            ("\\textbf{Reconstruction} $\\oplus$ \\textbf{untrained (ours)}", m125.loc["C3"], "C3")]),
        ("Use of seeds", [
            ("Ours, one seed (mean over three single-seed estimators)", m125.loc["C2"], "C2"),
            ("Target-only, one seed", m118.loc["D1"], "D1"),
            ("Target-only, three seeds, predictions averaged", m125.loc["E_old"], "E"),
            ("Target-only, three seeds, features concatenated in one readout$^{\\dagger}$", m118.loc["C3"], "Ecat")]),
        ("Numerical safeguards", [
            ("Reconstruction, three seeds, without prediction clip", m122.loc["reconE"], "C1"),
            ("Target-only, one seed, without any safeguard", m118.loc["nofix1"], None)]),
        ("Pre-training masks and raw features (target-only, one seed)", [
            ("No masks (always fully visible)", m118.loc["masknone1"], None),
            ("Bernoulli(0.5) masks", m118.loc["maskbern1"], None),
            ("Readout on $\\phi$ only, without $[x_S, x_S^2]$", m118.loc["phionly1"], None)])]
    body = []
    for g, rows in groups:
        body.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{g}}}}}\\\\")
        for lab, r, tk in rows:
            body.append(f"\\quad {lab} & {r.V:.3f} & {r.M2:.3f} & {r.rho:.3f} & {r.c1:.3f} & {r.c3:.3f} & {T(tk)} \\\\")
        body.append("\\addlinespace")
    body = body[:-1]
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Design choices of the estimator, averaged over the ten targets and all sets of size $\le 3$. The untrained network is
paired with the per-set readout. Cert.@$k$: certified $L^{(2)}$ over exact $M^{(2)}$ when $k$ backgrounds per field are retrained.
ms: time per field set on one idle GPU (same 200 sets per data set, batch 8, mean over data sets); -- : same cost as the corresponding
reference row. $^{\dagger}$Evaluated on the six RTS-GMLC and NEM targets only; on the same targets three averaged target-only seeds give
""" + f"{e6.V:.3f}, {e6.M2:.3f}, {e6.rho:.3f}, {e6.c1:.3f} and {e6.c3:.3f}." + r"""}\label{tab:design}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
Configuration & MAE $V$ & MAE $M^{(2)}$ & Spearman & cert.@1 & cert.@3 & ms\\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    w("tab_design.tex", s)


def _baseline_times():
    """每集合耗时（秒），五组的 [最小, 最大]：126 号（本文、线性、二次、代理模型）、127 号（Dropout、热启动、LazyVI、TabPFN）、118 号（串行重训）。"""
    import glob
    t126 = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation126/outputs/time_fast_*.csv"))])
    t127 = pd.concat([pd.read_csv(f) for f in glob.glob(R("DNN_Aggresvation127/outputs/time_[A-Z]*.csv"))] + [pd.read_csv(R("DNN_Aggresvation127/outputs/time_tabpfn.csv"))])
    T = {}
    for k, src, key in [("ours", t126, "C3"), ("lin", t126, "lin"), ("poly", t126, "poly"), ("surrogate", t126, "head3"),
                        ("dropout", t127, "dropout"), ("ws", t127, "ws"), ("lazyvi", t127, "lazyvi"), ("tabpfn", t127, "tabpfn")]:
        v = src[src.方法 == key].每集合ms / 1000; T[k] = (v.min(), v.max())
    seq = [np.load(R("DNN_Aggresvation118", "outputs", "est", t, "seq.npz"))["t"].sum(1).mean() for t in ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]]
    T["retrain"] = (min(seq), max(seq)); return T


def _fmt_t(a, b):
    def f(x):
        if x >= 10:
            return f"{x:.0f}\\,s"
        if x >= 1:
            return f"{x:.1f}\\,s"
        return (f"{x * 1000:.1f}" if x < 0.01 else f"{x * 1000:.0f}") + "\\,ms"
    fa, fb = f(a), f(b)
    if fa == fb:
        return fa
    if fa.endswith("ms") == fb.endswith("ms"):
        return fa.split("\\")[0] + "--" + fb
    return fa + "--" + fb


def tab_baselines():
    A = R("DNN_Aggresvation127/outputs/analysis"); M = pd.read_csv(os.path.join(A, "summary.csv"), index_col=0); T = _baseline_times()
    rows = [("retrain", "Sequential retraining of the attacker family (ground truth)", "--"),
            ("lin", "Per-set linear regression (retrained)", "--"),
            ("poly", "Per-set quadratic regression (retrained)", "--"),
            ("dropout", "Full model, removed fields set to their mean~\\cite{williamson2020spvim,sun2025warmstart}", "3 full models"),
            ("lazyvi", "LazyVI: linearized full model~\\cite{gao2022lazyvi}", "1 full model"),
            ("ws", "Warm start + early stopping~\\cite{sun2025warmstart}", "1 full model"),
            ("surrogate", "Masked surrogate, direct output~\\cite{covert2021explaining,jethani2022fastshap}", "3 masked backbones"),
            ("tabpfn", "TabPFN v2, in-context~\\cite{hollmann2025tabpfn}", "none (pre-trained)"),
            ("ours", "\\textbf{Ours}", "3 reconstr.\\ backbones")]
    f3 = lambda v: "--" if v != v else f"{v:.3f}"
    body = []
    for k, lab, pre in rows:
        if k == "retrain":
            body.append(f"{lab} & 0 & 0 & 0 & 0 & 1 & 1 & 1.00 / 1.00 & {_fmt_t(*T[k])} & {pre} \\\\"); continue
        r = M.loc[k]
        rec = "--" if r.关键召回05 != r.关键召回05 else f"{r.关键召回05:.2f} / {r.关键召回07:.2f}"
        body.append(f"{lab} & {f3(r.V误差)} & {f3(r.样本V误差)} & {f3(r.M2误差)} & {f3(r.Γ2误差)} & {f3(r.M2排序)} & {f3(r.认证前1)} & {rec} & {_fmt_t(*T[k])} & {pre} \\\\")
    s = r"""\begin{table}[!tbp]
\centering\scriptsize
\caption{Comparison with other ways of evaluating many field sets without retraining the attacker family, over the ten targets.
Columns 2 and 4--8 use all sets of size $\le3$; column 3 uses the same 200 random sets per data set for every method (LazyVI was
run only on these, see text). Crit.\ recall: scan--certify recall of $\tau$-critical fields at $\tau=0.5$ / $0.7$ without a margin
(precision 1.00 throughout). Time per field set for all targets of a data set on one idle GPU, range over data sets; last column:
models trained once per data set.}\label{tab:baselines}
\setlength{\tabcolsep}{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccccccl}
\toprule
Method & MAE $V$ & MAE $V$ (200) & MAE $M^{(2)}$ & MAE $\Gamma^{(2)}$ & Spearman & cert.@1 & crit.\ recall & time / set & trained once\\
\midrule
""" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    w("tab_baselines.tex", s)


if __name__ == "__main__":
    for t in (sys.argv[1:] or ["risk", "decision", "ranking", "ablation", "backbone", "masking"]):
        globals()[f"tab_{t}"]()
