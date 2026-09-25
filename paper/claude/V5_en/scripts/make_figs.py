# -*- coding: utf-8 -*-
"""V4_en 实验图：全部从各编号实验目录的 CSV / npy 读数，不手填数字。
用法：make_figs.py [图名 ...]（默认全部）"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S
import names as N

S.setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
GROUPS = [("RTS-GMLC", "DNN_Aggresvation111/outputs"), ("NEM", "DNN_Aggresvation112/outputs"),
          ("PJM", "DNN_Aggresvation116/groups/pjm_load/outputs"), ("PJM", "DNN_Aggresvation116/groups/pjm_gen_ic/outputs"),
          ("CAISO", "DNN_Aggresvation116/groups/caiso_load/outputs")]
DSCOL = {"RTS-GMLC": S.BLUE, "NEM": S.ORANGE, "PJM": S.AQUA, "CAISO": S.VIOLET}
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#dbe9f8", "#9fc3ea", "#3987e5", "#1c5cab", "#0d366b"])


def R(*p):
    return os.path.join(REPO, *p)


def summaries():
    rows = []
    for ds, rel in GROUPS:
        f = R(rel, "analysis", "summary_targets.csv")
        if not os.path.exists(f):
            continue
        spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        for _, r in pd.read_csv(f).iterrows():
            d = r.to_dict(); d.update(ds=ds, rel=rel, p=len(spec["cand"]), name=N.target(r["目标"])); rows.append(d)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 图 2：机理小例子
def fig_toy():
    T = pd.read_csv(R("DNN_Aggresvation117/outputs/toy_scores.csv"))
    V = pd.read_csv(R("DNN_Aggresvation117/outputs/toy_truth.csv")).set_index("集合")["V"]
    cols = [("pearson", "Pearson\n$r^2$"), ("mi", "Mutual\ninform."), ("M0", "Single\nfield"), ("loco", "LOCO\n"),
            ("perm", "Permu-\ntation"), ("sage", "SAGE\n"), ("graph", "Infer.\ngraph"), ("M2", "$M^{(2)}$\n(ours)")]
    fig = plt.figure(figsize=(S.TEXTW, 2.45)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    sets = [("A", "A"), ("B", "B"), ("A + B", "A+B"), ("C1", "C1"), ("C1 + C2", "C1+C2"), ("A + B + C1", "A+B+C1"),
            ("D", "D"), ("A + B + D", "A+B+D")]
    vals = [V[k] for k, _ in sets]; y = np.arange(len(sets))[::-1]
    col = [S.ORANGE if "+" not in k else S.BLUE for k, _ in sets]
    ax.barh(y, vals, 0.62, color=col)
    for yy, v in zip(y, vals):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=6.5, color=S.INK2)
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=":"); ax.axvline(0.7, color=S.MUTED, lw=0.6, ls="--")
    ax.text(0.5, len(sets) - 0.35, "τ=0.5", fontsize=6, color=S.MUTED, ha="center"); ax.text(0.7, len(sets) - 0.35, "0.7", fontsize=6, color=S.MUTED, ha="center")
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in sets]); ax.set_xlim(0, 1.08); ax.set_xlabel("$V_y(S)$ (test $R^2$, retrained)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); S.panel(ax, "a", x=-0.3)
    ax = fig.add_subplot(gs[1])
    M = np.array([[T.loc[T.字段 == f, c].values[0] for c, _ in cols] for f in T.字段])
    Mn = np.clip(M, 0, None) / np.clip(M, 0, None).max(0, keepdims=True)
    ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2, color="white" if Mn[i, j] > 0.6 else S.INK)
    ax.xaxis.tick_top(); ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=6.4)
    ax.set_yticks(range(len(T))); ax.set_yticklabels(T.字段); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((len(cols) - 1.5, -0.5), 1, len(T), fill=False, ec=S.BLUE, lw=1.2))
    lab = {"A": "synergy", "B": "synergy", "C1": "redundant", "C2": "redundant", "D": "indirect", "N": "noise"}
    for i, f in enumerate(T.字段):
        ax.text(len(cols) - 0.35, i, lab[f], va="center", fontsize=6.2, color=S.INK2, style="italic")
    ax.set_xlim(-0.5, len(cols) + 0.55); S.panel(ax, "b", x=-0.08, y=1.16)
    S.save(fig, os.path.join(OUT, "fig2_mechanism"))


# ------------------------------------------------------------------ 图 4：单字段定级漏掉的组合风险
def fig_combination():
    Sm = summaries()
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.85), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.12))
    y = np.arange(len(Sm))[::-1]
    for ax, tau, let in [(axs[0], 0.5, "a"), (axs[1], 0.7, "b")]:
        a = Sm[f"τ{tau}_单字段即危险"] / Sm.p; b = Sm[f"τ{tau}_单看安全组合危险"] / Sm.p
        ax.barh(y, a, 0.66, color=S.ORANGE, label="unsafe alone")
        ax.barh(y, b, 0.66, left=a, color=S.BLUE, label="safe alone, unsafe with ≤2 others")
        ax.barh(y, 1 - a - b, 0.66, left=a + b, color="#e6e6e6", label="never critical")
        for yy, n1, n2, P in zip(y, Sm[f"τ{tau}_单字段即危险"], Sm[f"τ{tau}_单看安全组合危险"], Sm.p):
            ax.text(1.02, yy, f"{n2}/{P}", va="center", fontsize=6.2, color=S.BLUE)
        ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(f"τ = {tau}", fontsize=8); ax.set_xlabel("share of candidate fields")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        S.panel(ax, let, x=-0.02 if let == "b" else -0.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels(Sm.name); axs[1].set_yticks([])
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=3, fontsize=6.6, handlelength=1.2)
    ax = axs[2]; big = "全部规模≤4集合最大V" if "全部规模≤4集合最大V" in Sm else None
    mx3 = [max_v3(r) for _, r in Sm.iterrows()]
    for yy, s1, s3, ds in zip(y, Sm.最强单字段V, mx3, Sm.ds):
        ax.plot([s1, s3], [yy, yy], color="#c9c9c9", lw=1.6, zorder=1)
        ax.scatter([s1], [yy], s=22, facecolor="white", edgecolor=S.ORANGE, lw=1.2, zorder=3)
        ax.scatter([s3], [yy], s=22, color=S.BLUE, zorder=3)
    ax.scatter([], [], s=22, facecolor="white", edgecolor=S.ORANGE, lw=1.2, label="best single field")
    ax.scatter([], [], s=22, color=S.BLUE, label="best set of ≤3 fields")
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=":"); ax.axvline(0.7, color=S.MUTED, lw=0.6, ls="--")
    ax.set_yticks([]); ax.set_xlim(0, 1.02); ax.set_xlabel("$V_y$ (test $R^2$)"); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=6.6); S.panel(ax, "c", x=-0.02)
    S.save(fig, os.path.join(OUT, "fig4_combination_risk"))


def max_v3(r):
    z = np.load(R(r.rel, "D.npz")); keys = [k for k in z["keys"]]; V = np.load(R(r.rel, "V_official.npy"))
    spec = json.load(open(R(r.rel, "fields.json"), encoding="utf-8")); c = [t.replace("Y_", "") for t in spec["targ"]].index(r["目标"])
    s3 = np.array([k.count("|") <= 2 for k in keys]); return float(V[s3, c].max())


# ------------------------------------------------------------------ 图 6：定义对比
def fig_definition():
    A = R("DNN_Aggresvation117/outputs/analysis")
    rules = pd.read_csv(os.path.join(A, "decision_rules.csv")); adapt = pd.read_csv(os.path.join(A, "withholding_strategies.csv"))
    det = pd.read_csv(os.path.join(A, "detection.csv")); curves = pd.read_csv(os.path.join(A, "protection_curves.csv"))
    fig = plt.figure(figsize=(S.TEXTW, 4.8)); gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.45, width_ratios=[1, 1.15, 1.15])
    # (a) 决策规则召回
    ax = fig.add_subplot(gs[0, 0])
    items = [("单字段规则 V({i})>τ", None, "single-field\nrule", S.ORANGE), ("扫描标记（估计）", 0.05, "scan\n(estimate)", LIGHT),
             ("扫描—认证标记", 0.05, "scan + certify", S.BLUE)]
    x = np.arange(2); w = 0.26
    for k, (rule, dl, lab, col) in enumerate(items):
        d = rules[(rules.规则 == rule) & ((rules.δ.isna()) if dl is None else (rules.δ == dl))]
        m = d.groupby("τ").召回.mean().reindex([0.5, 0.7])
        ax.bar(x + (k - 1) * w, m.values, w * 0.9, color=col, label=lab.replace("\n", " "))
        for xi, v in zip(x + (k - 1) * w, m.values):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=5.8, color=S.INK2)
    ax.set_xticks(x); ax.set_xticklabels(["τ = 0.5", "τ = 0.7"]); ax.set_ylim(0, 1.12); ax.set_ylabel("recall of critical fields")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=6.2); S.ygrid(ax); S.panel(ax, "a", x=-0.32)
    # (b) R-precision
    ax = fig.add_subplot(gs[0, 1:])
    order = ["pearson", "spearman", "mi", "dcor", "M0", "loco", "perm", "sage", "graph", "M2_est", "M2_cert", "M2"]
    rp = det.groupby(["τ", "方法"]).R精确率.mean().unstack(0).reindex(order)
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, rp[0.5], w * 0.92, color=[S.METHOD_STYLE[m][1] for m in order], alpha=0.45)
    ax.bar(x + w / 2, rp[0.7], w * 0.92, color=[S.METHOD_STYLE[m][1] for m in order])
    ax.set_xticks(x); ax.set_xticklabels([S.METHOD_STYLE[m][0] for m in order], rotation=40, ha="right", fontsize=6.2)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("R-precision"); S.ygrid(ax)
    ax.set_title("light bars: τ = 0.5;  dark bars: τ = 0.7", fontsize=6.6, color=S.INK2)
    S.panel(ax, "b", x=-0.1)
    # (c) 扣留：残余危险组合比例 vs 扣留数/最优
    ax = fig.add_subplot(gs[1, 0])
    strat = [("单字段定级（扣留单字段越阈者）", None, "single-field grading", S.ORANGE), ("估计MUS最小命中集", 0.05, "hitting set on $\\hat{V}$", S.AQUA),
             ("自适应M̂2贪心", 0.05, "adaptive $\\hat M^{(2)}$ greedy", S.BLUE)]
    for lab, dl, name, col in strat:
        d = adapt[(adapt.策略 == lab) & ((adapt.δ.isna()) if dl is None else (adapt.δ == dl))]
        for tau, mk in [(0.5, "o"), (0.7, "s")]:
            e = d[d.τ == tau]
            ax.scatter((e.扣留数 / e.最优扣留数.clip(lower=1)).mean(), (e.残余真危险组合 / e.危险组合数.clip(lower=1)).mean(),
                       s=34, marker=mk, color=col, edgecolor="white", lw=0.6, zorder=3, label=name if tau == 0.5 else None)
    ax.scatter([], [], marker="o", color=S.MUTED, s=20, label="τ = 0.5"); ax.scatter([], [], marker="s", color=S.MUTED, s=20, label="τ = 0.7")
    ax.set_xlabel("fields withheld / optimum"); ax.set_ylabel("dangerous combinations left"); ax.set_ylim(-0.05, 1.08); ax.set_xlim(0, 1.5)
    ax.axvline(1, color=S.MUTED, lw=0.6, ls=":")
    ax.legend(loc="upper right", fontsize=5.6, ncol=1, handletextpad=0.2, borderaxespad=0.1); S.ygrid(ax); S.panel(ax, "c", x=-0.32)
    # (d)(e) 两个代表目标的静态扣留曲线
    for k, (ds, tgt) in enumerate([("PJM-load", "Y_metered_load_mw"), ("NEM", "Y_机组出力_PPCCGT")]):
        ax = fig.add_subplot(gs[1, 1 + k]); c = curves[(curves.数据 == ds) & (curves.目标 == tgt)]
        for m in ["M0", "loco", "sage", "graph", "M2_est"]:
            e = c[c.方法 == m]; lab, col = S.METHOD_STYLE[m]
            ax.plot(e.k, e.剩余危险组合, color=col, lw=1.3 if m != "M2_est" else 1.8, label=lab, drawstyle="steps-post")
        ax.set_xlabel("fields withheld, by score"); ax.set_ylabel("dangerous combos left" if k == 0 else "")
        ax.set_title(N.target(tgt), fontsize=7.5); ax.set_xlim(0, c.k.max()); S.ygrid(ax); S.panel(ax, "de"[k], x=-0.22)
        if k == 1:
            ax.legend(loc="upper center", bbox_to_anchor=(-0.2, -0.3), ncol=3, fontsize=6.0)
    S.save(fig, os.path.join(OUT, "fig6_definition"))


LIGHT = S.LIGHTBLUE


# ------------------------------------------------------------------ 图 5：字段风险画像与增益矩阵
def fig_profile():
    fig = plt.figure(figsize=(S.TEXTW, 3.35)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.75)
    ax = fig.add_subplot(gs[0])
    P = pd.read_csv(R("DNN_Aggresvation111/outputs/analysis/field_profile.csv")); d = P[P.目标 == "线路潮流_C35"].nlargest(14, "M2").sort_values("M2")
    y = np.arange(len(d))
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.M0, r.M2], [yy, yy], color="#cfcfcf", lw=1.8, zorder=1)
    ax.scatter(d.r2, y, marker="D", s=14, color=S.MUTED, zorder=2, label="Pearson $r^2$")
    ax.scatter(d.M0, y, s=22, facecolor="white", edgecolor=S.ORANGE, lw=1.1, zorder=3, label="$M^{(0)}$ single field")
    ax.scatter(d.M2, y, s=22, color=S.BLUE, zorder=4, label="$M^{(2)}$ worst background")
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([N.field(f) for f in d.字段], fontsize=6.3); ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("inference risk for line C35 flow (test $R^2$)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, fontsize=5.9, handletextpad=0.2, columnspacing=0.8); S.panel(ax, "a", x=-0.62)
    for yy, (_, r) in list(zip(y, d.iterrows()))[-3:]:
        w = " + ".join(N.field(x.strip()) for x in str(r.K2见证).split("+"))
        ax.annotate(f"+ {w}", (r.M2, yy), xytext=(5, -2.2), textcoords="offset points", fontsize=5.2, color=S.BLUE)
    ax = fig.add_subplot(gs[1])
    G = pd.read_csv(R("DNN_Aggresvation112/outputs/analysis/gain_matrix_机组出力_PPCCGT.csv"), index_col=0).iloc[:10, :10]
    Gv = G.values; n = len(G)
    im = ax.imshow(np.clip(Gv, 0, None), cmap=SEQ, vmin=0, vmax=max(0.5, float(np.nanmax(Gv))))
    for i in range(n):
        for j in range(n):
            v = Gv[i, j]
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=S.ORANGE, lw=1.0))
            if v >= 0.10 or i == j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.3, color="white" if v > 0.35 else S.INK)
    lab = [N.field(x) for x in G.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=6.0)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=6.0); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False)
    cb.set_label("$V(\\{i,j\\})-V(\\{j\\})$; diagonal $V(\\{i\\})$", fontsize=6.4)
    ax.set_xlabel("background field $j$ held by the adversary", fontsize=7); ax.set_ylabel("field $i$ to be released", fontsize=7)
    ax.set_title("NEM unit PPCCGT", fontsize=7.5); S.panel(ax, "b", x=-0.62)
    S.save(fig, os.path.join(OUT, "fig5_profile"))


# ------------------------------------------------------------------ 图 7：估计器保真度与扫描—认证
def _est_sets():
    sys.path.insert(0, R("DNN_Aggresvation111", "src")); import pipe
    out = []
    for (ds, rel), (_, _, ef, ek) in zip(GROUPS, [("", "", "est_variants.npz", "E"), ("", "", "est_variants.npz", "E"),
                                                  ("", "", "est_group.npz", "E"), ("", "", "est_group.npz", "E"), ("", "", "est_group.npz", "E")]):
        if not os.path.exists(R(rel, "V_official.npy")) or not os.path.exists(R(rel, ef)):
            continue
        z = np.load(R(rel, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
        e = np.load(R(rel, ef)); s3 = e["sel"]; k3 = [k for k, t in zip(keys, s3) if t]
        V = np.load(R(rel, "V_official.npy"))[s3]; Ve = pipe.closure_max(e[ek], k3); p = z["Xtr"].shape[1]
        _, _, Dt, _, bsz = pipe.m_table(V, k3, list(range(p)), 2); _, _, De, _, _ = pipe.m_table(Ve, k3, list(range(p)), 2)
        spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        out.append(dict(ds=ds, V=V, Ve=Ve, Dt=Dt, De=De, bsz=bsz, targ=spec["targ"], pipe=pipe))
    return out


def fig_estimator():
    E = _est_sets(); fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.35), gridspec_kw=dict(wspace=0.42))
    rng = np.random.RandomState(0); seen = set()
    for e in E:
        col = DSCOL[e["ds"]]; lab = e["ds"] if e["ds"] not in seen else None; seen.add(e["ds"])
        idx = rng.choice(len(e["V"]), min(700, len(e["V"])), replace=False)
        axs[0].scatter(e["V"][idx].ravel(), e["Ve"][idx].ravel(), s=2.5, alpha=0.35, color=col, rasterized=True, label=lab)
        sel = e["bsz"] <= 2
        Mt, Me = e["Dt"][:, sel].max(1), e["De"][:, sel].max(1)
        axs[1].scatter(Mt.ravel(), Me.ravel(), s=9, alpha=0.75, color=col, edgecolor="white", lw=0.3)
    for ax, t in [(axs[0], "$V_y(S)$, all $|S|\\leq 3$"), (axs[1], "$M^{(2)}_{i\\to y}$")]:
        ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.6, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"retrained {t}"); ax.set_ylabel("amortized estimate")
    axs[0].legend(loc="upper left", fontsize=6.0, markerscale=3, handletextpad=0.1); S.panel(axs[0], "a", x=-0.3); S.panel(axs[1], "b", x=-0.3)
    ax = axs[2]; ks = np.arange(1, 11)
    for ds in ["RTS-GMLC", "NEM", "PJM", "CAISO"]:
        rows = []
        for e in [x for x in E if x["ds"] == ds]:
            sel = np.where(e["bsz"] <= 2)[0]
            for c in range(len(e["targ"])):
                dt, de = e["Dt"][:, sel, c], e["De"][:, sel, c]; Mt = dt.max(1); o = np.argsort(-de, 1)
                rows.append([np.take_along_axis(dt, o[:, :k], 1).max(1).sum() / max(Mt.sum(), 1e-9) for k in ks])
        if rows:
            ax.plot(ks, np.mean(rows, 0), "o-", ms=2.8, color=DSCOL[ds], label=ds, lw=1.2)
    ax.axhline(0.95, color=S.MUTED, lw=0.6, ls=":"); ax.set_ylim(0.8, 1.005); ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xlabel("backgrounds certified per field, $k$", fontsize=7); ax.set_ylabel("certified $L^{(2)}$ / exact $M^{(2)}$")
    ax.legend(loc="lower right", fontsize=6.0); S.ygrid(ax); S.panel(ax, "c", x=-0.3)
    S.save(fig, os.path.join(OUT, "fig7_estimator"))


# ------------------------------------------------------------------ 图 8：对比与消融、主干复用
def fig_ablation():
    V = pd.read_csv(R("DNN_Aggresvation118/outputs/analysis/variants_by_target.csv"))
    order = [("lin", "linear readout, no backbone"), ("polyS", "in-set quadratic, no backbone"), ("head3", "shared output head"),
             ("rand1", "untrained random backbone"), ("masknone1", "pre-training without masks"), ("maskbern1", "Bernoulli masks"),
             ("phionly1", "readout on $\\phi$ only"), ("nofix1", "no numerical safeguards"), ("D1", "ours, one seed"), ("E", "ours, three seeds")]
    B = pd.read_csv(R("DNN_Aggresvation116/outputs/analysis/backbone_compare.csv"))
    fig = plt.figure(figsize=(S.TEXTW, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.55, wspace=0.12, left=0.26, right=0.99)
    y = np.arange(len(order))[::-1]
    for k, (col, lab, lim) in enumerate([("V误差", "MAE of $V$", (0, 0.23)), ("M2误差", "MAE of $M^{(2)}$", (0, 0.155)),
                                         ("认证前1", "certified ratio, top-1 background", (0.6, 1.0))]):
        ax = fig.add_subplot(gs[0, k]); m = V.groupby("变体")[col].mean(); per = V.groupby(["变体", "数据"])[col].mean()
        cols = [S.BLUE if v in ("D1", "E") else ("#c8c8c8" if v in ("lin", "polyS", "head3") else "#8c8c8c") for v, _ in order]
        ax.barh(y, [m.get(v, np.nan) for v, _ in order], 0.62, color=cols)
        for yy, (v, _) in zip(y, order):
            ax.scatter(per.loc[v].values, [yy] * len(per.loc[v]), s=4, color=S.INK, zorder=3, alpha=0.65, lw=0)
        ax.set_yticks(y); ax.set_yticklabels([l for _, l in order] if k == 0 else [], fontsize=6.4); ax.tick_params(axis="y", length=0)
        ax.set_xlim(*lim); ax.set_xlabel(lab, fontsize=7); ax.grid(axis="x", color=S.GRID, lw=0.5); ax.set_axisbelow(True)
        S.panel(ax, "abc"[k], x=-0.05 if k else -0.72)
    order_b = ["本组主干", "全列主干", "留目标主干"]; lab_b = ["per-group", "all-column", "target held out"]; cb = ["#8c8c8c", S.BLUE, S.ORANGE]
    m = B.groupby("主干")[["死神经元比例", "有效维度", "V绝对误差", "边际Δ绝对误差", "M2误差", "认证前1", "认证前3"]].mean().reindex(order_b)
    gs2 = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.55, wspace=0.42, left=0.08, right=0.99)
    ax = fig.add_subplot(gs2[1, 0]); x = np.arange(3)
    ax.bar(x, m["死神经元比例"] * 100, 0.6, color=cb)
    for xi, v, dm in zip(x, m["死神经元比例"] * 100, m["有效维度"]):
        ax.text(xi, v + 0.8, f"eff. dim\n{dm:.2f}", ha="center", fontsize=5.6, color=S.INK2)
    ax.set_ylim(0, 36); ax.set_xticks(x); ax.set_xticklabels(lab_b, fontsize=6.2); ax.set_ylabel("inactive units (%)", fontsize=7); S.ygrid(ax); S.panel(ax, "d", x=-0.3)
    ax = fig.add_subplot(gs2[1, 1]); w = 0.26
    for j in range(3):
        ax.bar(x + (j - 1) * w, m.iloc[j][["V绝对误差", "边际Δ绝对误差", "M2误差"]].values, w * 0.9, color=cb[j], label=lab_b[j])
    ax.set_xticks(x); ax.set_xticklabels(["$V$", "marginal $\\Delta$", "$M^{(2)}$"], fontsize=6.6); ax.set_ylabel("MAE", fontsize=7)
    ax.set_ylim(0, 0.085); ax.legend(fontsize=5.6, loc="upper left", ncol=1); S.ygrid(ax); S.panel(ax, "e", x=-0.3)
    ax = fig.add_subplot(gs2[1, 2])
    for j in range(3):
        ax.bar(np.arange(2) + (j - 1) * w, m.iloc[j][["认证前1", "认证前3"]].values, w * 0.9, color=cb[j])
    ax.set_xticks(range(2)); ax.set_xticklabels(["top-1", "top-3"], fontsize=6.6); ax.set_ylim(0.6, 1.0); ax.set_ylabel("certified ratio", fontsize=7)
    S.ygrid(ax); S.panel(ax, "f", x=-0.3)
    S.save(fig, os.path.join(OUT, "fig8_ablation"))


# ------------------------------------------------------------------ 图 9：稳健性（K 饱和、种子/划分、τ）
def fig_robust():
    A = R("DNN_Aggresvation119/outputs/analysis"); Sm = summaries()
    fig, axs = plt.subplots(1, 4, figsize=(S.TEXTW, 2.75), gridspec_kw=dict(wspace=0.75, width_ratios=[1.1, 1.0, 1.0, 1.0]))
    nem = pd.read_csv(os.path.join(A, "nem_k3.csv")); nem = nem[nem.口径.str.startswith("规模 ≤4")].set_index("目标")
    rows = []
    for _, r in Sm.iterrows():
        m23 = float(nem.loc["机组出力_" + r["目标"].split("_", 1)[1], "M2/M3"]) if r.ds == "NEM" else r["K饱和_M2除M3"]
        rows.append((r["name"], r["K饱和_M1除M2"], m23))
    y = np.arange(len(rows))[::-1]; ax = axs[0]
    ax.scatter([r[1] for r in rows], y, s=16, facecolor="white", edgecolor=S.ORANGE, lw=1.1, label="$M^{(1)}/M^{(2)}$", zorder=3)
    ax.scatter([r[2] for r in rows], y, s=16, color=S.BLUE, label="$M^{(2)}/M^{(3)}$", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.0); ax.set_xlim(0.65, 1.02)
    ax.set_xlabel("ratio of mean risk", fontsize=7); ax.legend(loc="lower left", fontsize=5.8, handletextpad=0.1); ax.grid(axis="x", color=S.GRID, lw=0.5)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); S.panel(ax, "a", x=-0.85)
    ax = axs[1]; K = pd.read_csv(os.path.join(A, "k3_critical.csv")); K = K[K.τ == 0.7].reset_index(drop=True)
    yk = np.arange(len(K))[::-1]
    for yy, (_, r) in zip(yk, K.iterrows()):
        ax.plot([r.K2关键 / r.字段数, r.K3关键 / r.字段数], [yy, yy], color="#cfcfcf", lw=1.6, zorder=1)
    ax.scatter(K.K2关键 / K.字段数, yk, s=16, color=S.BLUE, zorder=3, label="$K=2$")
    ax.scatter(K.K3关键 / K.字段数, yk, s=16, facecolor="white", edgecolor=S.VIOLET, lw=1.1, zorder=3, label="$K=3$")
    ax.set_yticks([]); ax.set_xlim(-0.03, 1.03); ax.set_xlabel("critical fields at τ = 0.7", fontsize=7); ax.legend(loc="upper left", fontsize=5.8)
    ax.grid(axis="x", color=S.GRID, lw=0.5); ax.spines["left"].set_visible(False); S.panel(ax, "b", x=-0.1)
    ax = axs[2]; C = pd.read_csv(os.path.join(A, "robust_consistency.csv"))
    mets = [("M2排序Spearman", "Spearman\nof $M^{(2)}$"), ("关键集合Jaccard_τ05", "Jaccard,\nτ = 0.5"), ("关键集合Jaccard_τ07", "Jaccard,\nτ = 0.7")]
    for j, (col, _) in enumerate(mets):
        for kind, mk, colr in [("种子", "o", S.BLUE), ("划分", "s", S.ORANGE)]:
            v = C[C.版本.str.startswith(kind)][col]
            ax.scatter(v, j + (0.12 if kind == "划分" else -0.12) + np.linspace(-0.05, 0.05, len(v)), s=9, marker=mk, color=colr, zorder=3,
                       label=("attacker seed" if kind == "种子" else "data split") if j == 0 else None)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in mets], fontsize=6.0); ax.invert_yaxis(); ax.set_xlim(0.88, 1.01)
    ax.set_xlabel("agreement with main run", fontsize=7); ax.grid(axis="x", color=S.GRID, lw=0.5); ax.tick_params(axis="y", length=0)
    ax.legend(loc="center left", fontsize=5.6, handletextpad=0.1, bbox_to_anchor=(0, 0.72)); S.panel(ax, "c", x=-0.55)
    ax = axs[3]; T = pd.read_csv(os.path.join(A, "tau_sweep.csv"))
    m = T.groupby("τ").apply(lambda e: pd.Series(dict(c=(e.组合危险字段 / e.字段数).mean(), s=(e.单字段即危险 / e.字段数).mean(), l=e.仍暴露比例.mean())))
    ax.plot(m.index, m.c, color=S.BLUE, lw=1.5, label="safe alone, critical")
    ax.plot(m.index, m.s, color=S.ORANGE, lw=1.3, ls="--", label="unsafe alone")
    ax.plot(m.index, m.l, color=S.MUTED, lw=1.2, ls=":", label="MUS left after\nsingle-field grading")
    ax.set_xlabel("threshold τ", fontsize=7); ax.set_ylim(0, 1.02); ax.set_ylabel("share", fontsize=7); S.ygrid(ax)
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.45, -0.28), handlelength=1.6); S.panel(ax, "d", x=-0.4)
    S.save(fig, os.path.join(OUT, "fig9_robustness"))


# ------------------------------------------------------------------ 图 10：成本
def fig_cost():
    rows = []
    for tag in ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]:
        f = R("DNN_Aggresvation118/outputs/est", tag, "seq.npz"); t = R("DNN_Aggresvation118/outputs/est", tag, "timeE.npz")
        if os.path.exists(f) and os.path.exists(t):
            z = np.load(f)["t"]; e = np.load(t)
            rows.append(dict(tag=tag, dnn=z[:, 0].mean(), tree=z[:, 1].mean(), E=float(e["E"]), n3=int(e["n3"]), p=int(e["p"])))
    if not rows:
        print("缺少成本数据"); return
    D = pd.DataFrame(rows); fig, axs = plt.subplots(1, 2, figsize=(S.TEXTW * 0.8, 2.2), gridspec_kw=dict(wspace=0.45))
    x = np.arange(len(D)); ax = axs[0]
    ax.bar(x - 0.2, D.dnn + D.tree, 0.38, color=S.ORANGE, label="sequential retraining")
    ax.bar(x + 0.2, D.E, 0.38, color=S.BLUE, label="amortized estimate")
    ax.set_yscale("log"); ax.set_ylabel("seconds per field set"); ax.set_xticks(x); ax.set_xticklabels(D.tag.str.replace("_", "/"), rotation=25, fontsize=6.3)
    ax.legend(fontsize=6.0); S.ygrid(ax); S.panel(ax, "a", x=-0.3)
    ax = axs[1]
    ax.bar(x - 0.2, (D.dnn + D.tree) * D.n3 / 3600, 0.38, color=S.ORANGE); ax.bar(x + 0.2, D.E * D.n3 / 3600, 0.38, color=S.BLUE)
    ax.set_yscale("log"); ax.set_ylabel("hours for the $K=2$ table"); ax.set_xticks(x); ax.set_xticklabels(D.tag.str.replace("_", "/"), rotation=25, fontsize=6.3)
    S.ygrid(ax); S.panel(ax, "b", x=-0.3)
    S.save(fig, os.path.join(OUT, "fig10_cost"))


# ------------------------------------------------------------------ 背景放大（124 号）
def fig_amplification():
    A = R("DNN_Aggresvation124/outputs"); P = pd.read_csv(os.path.join(A, "profile.csv")); Dd = pd.read_csv(os.path.join(A, "context_gain_distribution.csv"))
    dsmap = {"RTS-GMLC": "RTS-GMLC", "NEM": "NEM", "PJM-load": "PJM", "PJM-gen/ic": "PJM", "CAISO-load": "CAISO"}
    fig = plt.figure(figsize=(S.TEXTW, 2.9)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.55)
    ax = fig.add_subplot(gs[0]); seen = set()
    for _, r in P.iterrows():
        ds = dsmap[r.数据]; lab = ds if ds not in seen else None; seen.add(ds)
        ax.scatter(r.M0, r.G2, s=9, color=DSCOL[ds], alpha=0.8, edgecolor="white", lw=0.3, label=lab)
    ax.axvline(0.3, color=S.MUTED, lw=0.6, ls=":"); ax.axhline(0.1, color=S.MUTED, lw=0.6, ls=":")
    n = int(((P.M0 < 0.3) & (P.G2 > 0.1)).sum())
    ax.text(0.02, 0.44, f"{n} of {len(P)} pairs:\nlow alone, amplified", fontsize=6.2, color=S.INK2, va="top")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.45); ax.set_xlabel("single-field risk $M^{(0)}$"); ax.set_ylabel("amplification $\\Gamma^{(2)}=M^{(2)}-M^{(0)}$")
    ax.legend(loc="center right", fontsize=5.8, handletextpad=0.1, markerscale=1.4); S.ygrid(ax); S.panel(ax, "a", x=-0.28)
    ax = fig.add_subplot(gs[1])
    pick = P[P.数据.isin(Dd.数据.unique())].sort_values("G2", ascending=False).drop_duplicates(["数据", "目标"]).head(6)
    for k, (_, r) in enumerate(pick.iterrows()):
        d = Dd[(Dd.数据 == r.数据) & (Dd.目标 == r.目标) & (Dd.字段 == r.字段)].Δ.values
        ax.scatter(d, np.full(len(d), k) + np.random.RandomState(k).uniform(-.22, .22, len(d)), s=1.6, alpha=0.35, color=DSCOL[dsmap[r.数据]], rasterized=True)
        ax.plot([d.mean()], [k], "|", ms=11, color=S.INK, mew=1.6)
        ax.plot([r.M0], [k], "o", ms=5, mfc="white", mec=S.ORANGE, mew=1.2); ax.plot([d.max()], [k], "o", ms=5, color=S.BLUE)
    ax.set_yticks(range(len(pick))); ax.set_yticklabels([f"{N.field(r.字段)}\n→ {N.target(r.目标)}" for _, r in pick.iterrows()], fontsize=5.9)
    ax.invert_yaxis(); ax.set_xlabel("contextual gain $\\Delta_i(T)$, all $|T|\\leq 2$"); ax.grid(axis="x", color=S.GRID, lw=0.5)
    ax.plot([], [], "o", ms=5, mfc="white", mec=S.ORANGE, mew=1.2, label="alone, $M^{(0)}$"); ax.plot([], [], "|", ms=9, color=S.INK, mew=1.6, label="average")
    ax.plot([], [], "o", ms=5, color=S.BLUE, label="maximum, $M^{(2)}$"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=5.8, handletextpad=0.2)
    ax.tick_params(axis="y", length=0); S.panel(ax, "b", x=-0.5)
    S.save(fig, os.path.join(OUT, "fig_amplification"))


# ------------------------------------------------------------------ 受控机制算例（123 号）
MECH_SHORT = {"P_g (unit 313 output)": "unit output $P_g$", "LMP_g (bus 313)": "own-bus LMP", "LMP_near (bus 306)": "nearby LMP (306)",
              "LMP_far (bus 303, behind C6)": "remote LMP (303, behind C6)", "LMP_area2 (bus 223)": "other-area LMP (223)",
              "System load": "system load", "Area-3 load": "area-3 load", "Wind available": "wind available", "PV available": "PV available",
              "Congestion C6": "congestion C6", "P_comp (unit 316 output)": "competitor output (316)"}


def fig_mechanism(tag="narrow"):
    A = R("DNN_Aggresvation123/outputs", tag, "analysis")
    G = pd.read_csv(os.path.join(A, "contextual_gain_P.csv")); Rg = pd.read_csv(os.path.join(A, "regime.csv")); Sc = pd.read_csv(os.path.join(A, "method_scores.csv"))
    fig = plt.figure(figsize=(S.TEXTW, 3.0)); gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.8, 1.45], wspace=0.55)
    ax = fig.add_subplot(gs[0]); G = G.iloc[::-1].reset_index(drop=True)
    cat = lambda b: S.BLUE if "LMP" in b else (S.AQUA if "P_comp" in b else ("#bdbdbd" if b == "∅" else S.MUTED))
    ax.barh(range(len(G)), G.Δ_P, 0.66, color=[cat(b) for b in G.背景])
    ax.set_yticks(range(len(G))); ax.set_yticklabels(["none" if b == "∅" else MECH_SHORT[b] for b in G.背景], fontsize=6.0)
    ax.set_xlabel("gain of $P_g$ over background $\\{j\\}$", fontsize=7); ax.grid(axis="x", color=S.GRID, lw=0.5); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); S.panel(ax, "a", x=-0.95)
    ax = fig.add_subplot(gs[1]); rows = Rg.set_index("背景")
    labs = [("LMP_g", "own-bus"), ("LMP_far", "remote")]; x = np.arange(2); w = 0.36
    for k, (reg, col) in enumerate([("C6 不阻塞", S.LIGHTBLUE), ("C6 阻塞", S.BLUE)]):
        vals = [rows.loc[b, f"{reg}_V(T+P)"] - rows.loc["∅", f"{reg}_V(T+P)"] for b, _ in labs]
        ax.bar(x + (k - 0.5) * w, vals, w * 0.9, color=col, label="uncongested" if k == 0 else "C6 congested")
    ax.set_xticks(x); ax.set_xticklabels([l + "\nLMP" for _, l in labs], fontsize=6.6); ax.set_ylabel("amplification of $P_g$", fontsize=7)
    ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1); S.ygrid(ax); S.panel(ax, "b", x=-0.45)
    ax = fig.add_subplot(gs[2])
    cols = [("pearson", "Pearson"), ("mi", "MI"), ("M0", "single"), ("loco", "LOCO"), ("perm", "perm."), ("sage", "SAGE"), ("graph", "graph"), ("M2", "$M^{(2)}$")]
    order = ["P_g (unit 313 output)", "LMP_g (bus 313)", "LMP_near (bus 306)", "LMP_area2 (bus 223)", "LMP_far (bus 303, behind C6)",
             "P_comp (unit 316 output)", "System load", "PV available", "Area-3 load", "Wind available", "Congestion C6"]
    Sc = Sc.set_index("字段").reindex(order); M = Sc[[c for c, _ in cols]].values.astype(float)
    Mn = np.clip(M, 0, None) / np.clip(M, 0, None).max(0, keepdims=True)
    ax.imshow(Mn, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            vv = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
            ax.text(j, i, f"{vv:.2f}", ha="center", va="center", fontsize=4.9, color="white" if Mn[i, j] > 0.6 else S.INK)
    ax.xaxis.tick_top(); ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=6.0, rotation=40, ha="left")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([MECH_SHORT[o] for o in order], fontsize=5.8); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(plt.Rectangle((len(cols) - 1.5, -0.5), 1, len(order), fill=False, ec=S.BLUE, lw=1.1)); S.panel(ax, "c", x=-0.62, y=1.1)
    S.save(fig, os.path.join(OUT, f"fig_mechanism_{tag}"))


# ------------------------------------------------------------------ 掩码预训练诊断（122 号）
def fig_masking():
    A = R("DNN_Aggresvation122/outputs/analysis")
    K = pd.read_csv(os.path.join(A, "rank_truncation.csv")); K["秩"] = K["秩"].astype(str)
    V = pd.read_csv(os.path.join(A, "variants_by_target.csv")); L = pd.read_csv(os.path.join(A, "large_sets.csv"))
    Dc = pd.read_csv(os.path.join(A, "error_decomposition.csv"))
    SM = {"recon+random@1_cy": "recon+random_cy", "recon+random@2_cy": "recon+random_cy"}   # 推荐配置取三种子平均
    V = V.replace({"变体": SM}); Dc = Dc.replace({"变体": SM})
    BB = [("random", "untrained", S.GRAYS[0]), ("uniform", "target-only", S.ORANGE), ("recon", "target + reconstruction", S.BLUE)]
    fig = plt.figure(figsize=(S.TEXTW, 2.55)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.42)
    ax = fig.add_subplot(gs[0]); ranks = ["0", "2", "4", "8", "32", "全部"]; xr = {r: k for k, r in enumerate(ranks)}
    g = K.groupby(["主干", "秩"]).V误差.mean()
    for bb, lab, col in BB:
        rr = [r for r in ranks if (bb, r) in g.index]
        ax.plot([xr[r] for r in rr], [g[(bb, r)] for r in rr], "-o", color=col, lw=1.3, ms=3.2, label=lab, mec="white", mew=0.4)
    ax.set_xticks(range(len(ranks))); ax.set_xticklabels(["0", "2", "4", "8", "32", "256"]); ax.set_xlabel("principal components of $\\phi$ kept")
    ax.set_ylabel("error of $\\hat V$ ($|S|\\leq3$)"); S.ygrid(ax); S.panel(ax, "a", x=-0.3)
    ax.legend(fontsize=5.8, loc="upper right", handlelength=1.4)
    ax = fig.add_subplot(gs[1])
    for bb, lab, col in BB:
        y = [V[V.变体 == bb].V误差_3.mean(), V[V.变体 == bb].V误差_4.mean()] + [L[(L.变体 == bb) & (L.规模 == s)].V误差.mean() for s in (6, 10, 16)]
        ax.plot(range(5), y, "-o", color=col, lw=1.3, ms=3.2, mec="white", mew=0.4)
    ax.axvspan(1.5, 4.3, color="#f3f3f3", zorder=0, lw=0)
    ax.set_xticks(range(5)); ax.set_xticklabels(["$\\leq$3", "4", "6", "10", "16"]); ax.set_xlabel("set size $|S|$"); ax.set_ylabel("error of $\\hat V$")
    S.ygrid(ax); S.panel(ax, "b", x=-0.3)
    ax = fig.add_subplot(gs[2]); vs = ["reconE", "reconE_cy", "recon+random_cy"]
    lab = {"reconE": "reconstr. ×3", "reconE_cy": "reconstr. ×3 + clip", "recon+random_cy": "reconstr. ⊕ untrained + clip"}
    col = {"reconE": S.LIGHTBLUE, "reconE_cy": S.BLUE, "recon+random_cy": "#0d366b"}
    mets = [("集合误差", Dc, "$V$\nerror"), ("边际误差", Dc, "$\\Delta$\nerror"), ("字段最大高估", Dc, "max over-\nestimate"), ("M2误差", V, "$M^{(2)}$\nerror")]
    w = 0.8 / len(vs)
    for k, v in enumerate(vs):
        vals = [src[src.变体 == v][c].mean() / src[src.变体 == "uniformE"][c].mean() for c, src, _ in mets]
        ax.bar(np.arange(len(mets)) + (k - len(vs) / 2 + 0.5) * w, vals, w * 0.92, color=col[v], label=lab[v])
    ax.axhline(1, color=S.INK2, lw=0.6); ax.set_xticks(range(len(mets))); ax.set_xticklabels([m for _, _, m in mets], fontsize=6.4)
    ax.set_ylabel("relative to target-only ×3 (E)"); ax.set_ylim(0.5, 1.45); S.ygrid(ax); S.panel(ax, "c", x=-0.2)
    ax.legend(fontsize=5.6, ncol=1, loc="upper left", handlelength=1.0)
    S.save(fig, os.path.join(OUT, "fig_masking"))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["toy", "combination"]
    for t in todo:
        (fig_mechanism("wide") if t == "mechanism_wide" else globals()[f"fig_{t}"]()); print("done", t)
