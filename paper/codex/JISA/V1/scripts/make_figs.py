# -*- coding: utf-8 -*-
"""V5.1_en 实验图（估计器固定为 125 号的新估计器）：全部从各编号实验目录的 CSV / npy 读数，不手填数字。
用法：make_figs.py [图名 ...]（默认全部）"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import style as S
import names as N

S.setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures")
from pathlib import Path
REPO = str(next(p for p in Path(__file__).resolve().parents if (p / "DNN_Aggresvation125").is_dir()))
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
    fig = plt.figure(figsize=(S.TEXTW, 3.1)); gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.9], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    sets = [("A", "A"), ("B", "B"), ("A + B", "A+B"), ("C1", "C1"), ("C1 + C2", "C1+C2"), ("A + B + C1", "A+B+C1"),
            ("D", "D"), ("A + B + D", "A+B+D")]
    vals = [V[k] for k, _ in sets]; y = np.arange(len(sets))[::-1]
    col = [S.ORANGE if "+" not in k else S.BLUE for k, _ in sets]
    ax.barh(y, vals, 0.62, color=col)
    for yy, v in zip(y, vals):
        ax.text(v + 0.02, yy, f"{v:.2f}", va="center", fontsize=7.4, color=S.INK2)
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=":"); ax.axvline(0.7, color=S.MUTED, lw=0.6, ls="--")
    ax.text(0.5, len(sets) - 0.35, "τ=0.5", fontsize=7.4, color=S.MUTED, ha="center"); ax.text(0.7, len(sets) - 0.35, "0.7", fontsize=7.4, color=S.MUTED, ha="center")
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in sets]); ax.set_xlim(0, 1.08); ax.set_xlabel("$V_y(S)$ (test $R^2$, retrained)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0); S.panel(ax, "a", x=-0.3)
    ax = fig.add_subplot(gs[1])
    M = np.array([[T.loc[T.字段 == f, c].values[0] for c, _ in cols] for f in T.字段])
    Mn = np.clip(M, 0, None) / np.clip(M, 0, None).max(0, keepdims=True)
    S.heatmap(ax, Mn, SEQ)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = 0.0 if abs(M[i, j]) < 0.005 else M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4, color="white" if Mn[i, j] > 0.6 else S.INK)
    ax.xaxis.tick_top(); ax.set_xticks(range(len(cols))); ax.set_xticklabels([l for _, l in cols], fontsize=7.4, rotation=25, ha="left")
    ax.set_yticks(range(len(T))); ax.set_yticklabels(T.字段); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((len(cols) - 1.5, -0.5), 1, len(T), fill=False, ec=S.BLUE, lw=1.2))
    lab = {"A": "synergy", "B": "synergy", "C1": "redundant", "C2": "redundant", "D": "indirect", "N": "noise"}
    for i, f in enumerate(T.字段):
        ax.text(len(cols) - 0.35, i, lab[f], va="center", fontsize=7.4, color=S.INK2, style="italic")
    ax.set_xlim(-0.5, len(cols) + 0.55); S.panel(ax, "b", x=-0.08, y=1.16)
    S.save(fig, os.path.join(OUT, "fig2_mechanism"))


# ------------------------------------------------------------------ 图 4：单字段定级漏掉的组合风险
def fig_combination():
    Sm = summaries()
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05], wspace=0.12))
    y = np.arange(len(Sm))[::-1]
    for ax, tau, let in [(axs[0], 0.5, "a"), (axs[1], 0.7, "b")]:
        a = Sm[f"τ{tau}_单字段即危险"] / Sm.p; b = Sm[f"τ{tau}_单看安全组合危险"] / Sm.p
        ax.barh(y, a, 0.66, color=S.ORANGE, label="unsafe alone")
        ax.barh(y, b, 0.66, left=a, color=S.BLUE, label="safe alone, unsafe with ≤2 others")
        ax.barh(y, 1 - a - b, 0.66, left=a + b, color="#e6e6e6", label="never critical")
        for yy, n1, n2, P in zip(y, Sm[f"τ{tau}_单字段即危险"], Sm[f"τ{tau}_单看安全组合危险"], Sm.p):
            ax.text(1.02, yy, f"{n2}/{P}", va="center", fontsize=7.4, color=S.BLUE)
        ax.set_xlim(0, 1.18); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"])
        ax.set_title(f"τ = {tau}", fontsize=8.0); ax.set_xlabel("share of candidate fields")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        S.panel(ax, let, x=-0.02 if let == "b" else -0.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels(Sm.name); axs[1].set_yticks([])
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=3, fontsize=7.4, handlelength=1.2)
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=7.4); S.panel(ax, "c", x=-0.02)
    S.save(fig, os.path.join(OUT, "fig4_combination_risk"))


def max_v3(r):
    z = np.load(R(r.rel, "D.npz")); keys = [k for k in z["keys"]]; V = np.load(R(r.rel, "V_official.npy"))
    spec = json.load(open(R(r.rel, "fields.json"), encoding="utf-8")); c = [t.replace("Y_", "") for t in spec["targ"]].index(r["目标"])
    s3 = np.array([k.count("|") <= 2 for k in keys]); return float(V[s3, c].max())


# ------------------------------------------------------------------ 图 6：定义对比


LIGHT = S.LIGHTBLUE


# ------------------------------------------------------------------ 图 5：字段风险画像与增益矩阵
def fig_profile():
    fig = plt.figure(figsize=(S.TEXTW, 6.3)); gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=.8)
    ax = fig.add_subplot(gs[0])
    P = pd.read_csv(R("DNN_Aggresvation111/outputs/analysis/field_profile.csv")); d = P[P.目标 == "线路潮流_C35"].nlargest(14, "M2").sort_values("M2")
    y = np.arange(len(d))
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.M0, r.M2], [yy, yy], color="#cfcfcf", lw=1.8, zorder=1)
    ax.scatter(d.r2, y, marker="D", s=14, color=S.MUTED, zorder=2, label="Pearson $r^2$")
    ax.scatter(d.M0, y, s=22, facecolor="white", edgecolor=S.ORANGE, lw=1.1, zorder=3, label="$M^{(0)}$ single field")
    ax.scatter(d.M2, y, s=22, color=S.BLUE, zorder=4, label="$M^{(2)}$ worst background")
    ax.axvline(0.5, color=S.MUTED, lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels([N.field(f) for f in d.字段], fontsize=7.4); ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Risk for line C35 flow (test $R^2$)"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.55, 1.16), ncol=3, fontsize=7.4, handletextpad=0.2, columnspacing=0.8); S.panel(ax, "a", x=-0.04)
    for yy, (_, r) in list(zip(y, d.iterrows()))[-3:]:
        w = " + ".join(N.field(x.strip()) for x in str(r.K2见证).split("+"))
        ax.annotate(f"+ {w}", (r.M2, yy), xytext=(5, -2.2), textcoords="offset points", fontsize=7.4, color=S.BLUE)
    ax = fig.add_subplot(gs[1])
    G = pd.read_csv(R("DNN_Aggresvation112/outputs/analysis/gain_matrix_机组出力_PPCCGT.csv"), index_col=0).iloc[:10, :10]
    Gv = G.values; n = len(G)
    im = ax.pcolormesh(np.arange(n+1)-.5, np.arange(n+1)-.5, np.clip(Gv, 0, None), cmap=SEQ, vmin=0, vmax=max(.5,float(np.nanmax(Gv))), shading="flat", rasterized=False); ax.set_ylim(n-.5,-.5)
    for i in range(n):
        for j in range(n):
            v = Gv[i, j]
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=S.ORANGE, lw=1.0))
            if v >= 0.10 or i == j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.4, color="white" if v > 0.35 else S.INK)
    lab = [N.field(x) for x in G.index]
    ax.set_xticks(range(n)); ax.set_xticklabels(lab, rotation=55, ha="right", fontsize=7.4)
    ax.set_yticks(range(n)); ax.set_yticklabels(lab, fontsize=7.4); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.outline.set_visible(False); cb.solids.set_rasterized(False)
    cb.set_label("$V(\\{i,j\\})-V(\\{j\\})$; diagonal $V(\\{i\\})$", fontsize=7.4)
    ax.set_xlabel("background field $j$ held by the adversary", fontsize=7.4); ax.set_ylabel("field $i$ to be released", fontsize=7.4)
    ax.set_title("NEM unit PPCCGT", fontsize=7.5); S.panel(ax, "b", x=-0.04)
    S.save(fig, os.path.join(OUT, "fig5_profile"))


# ------------------------------------------------------------------ 图 7：估计器保真度与扫描—认证
def _est_sets():
    """估计器保真度用 125 号固定的新估计器（outputs/final_est/<组>.npz：sel、E）。"""
    sys.path.insert(0, R("DNN_Aggresvation111", "src")); import pipe
    out = []
    for (ds, rel), T in zip(GROUPS, ["RTS-GMLC", "NEM", "PJM-load", "PJM-gen_ic", "CAISO-load"]):
        ef = R("DNN_Aggresvation125", "outputs", "final_est", f"{T}.npz")
        if not os.path.exists(R(rel, "V_official.npy")) or not os.path.exists(ef):
            continue
        z = np.load(R(rel, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
        e = np.load(ef); s3 = e["sel"]; k3 = [k for k, t in zip(keys, s3) if t]
        V = np.load(R(rel, "V_official.npy"))[s3]; Ve = pipe.closure_max(e["E"], k3); p = z["Xtr"].shape[1]
        _, _, Dt, _, bsz = pipe.m_table(V, k3, list(range(p)), 2); _, _, De, _, _ = pipe.m_table(Ve, k3, list(range(p)), 2)
        spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        out.append(dict(ds=ds, V=V, Ve=Ve, Dt=Dt, De=De, bsz=bsz, targ=spec["targ"], pipe=pipe))
    return out


def fig_estimator():
    E = _est_sets(); fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.85), gridspec_kw=dict(wspace=0.42))
    rng = np.random.RandomState(0); seen = set()
    for e in E:
        col = DSCOL[e["ds"]]; lab = e["ds"] if e["ds"] not in seen else None; seen.add(e["ds"])
        idx = rng.choice(len(e["V"]), min(700, len(e["V"])), replace=False)
        axs[0].scatter(e["V"][idx].ravel(), e["Ve"][idx].ravel(), s=2.5, alpha=0.35, color=col, rasterized=False, label=lab)
        sel = e["bsz"] <= 2
        Mt, Me = e["Dt"][:, sel].max(1), e["De"][:, sel].max(1)
        axs[1].scatter(Mt.ravel(), Me.ravel(), s=9, alpha=0.75, color=col, edgecolor="white", lw=0.3)
    for ax, t in [(axs[0], "$V_y(S)$, all $|S|\\leq 3$"), (axs[1], "$M^{(2)}_{i\\to y}$")]:
        ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.6, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel(f"retrained {t}"); ax.set_ylabel("amortized estimate")
    axs[0].legend(loc="upper left", fontsize=7.4, markerscale=3, handletextpad=0.1); S.panel(axs[0], "a", x=-0.3); S.panel(axs[1], "b", x=-0.3)
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
    ax.set_xlabel("backgrounds certified, $k$", fontsize=7.4); ax.set_ylabel("certified $L^{(2)}$ / exact $M^{(2)}$")
    ax.legend(loc="lower right", fontsize=7.4); S.ygrid(ax); S.panel(ax, "c", x=-0.3)
    S.save(fig, os.path.join(OUT, "fig7_estimator"))


# ------------------------------------------------------------------ 图 8：对比与消融、主干复用


# ------------------------------------------------------------------ 图 9：稳健性（K 饱和、种子/划分、τ）
def fig_robust():
    A = R("DNN_Aggresvation119/outputs/analysis"); Sm = summaries()
    fig, axs = plt.subplots(2, 2, figsize=(S.TEXTW, 6.2), gridspec_kw=dict(wspace=0.85, hspace=0.65)); axs = axs.ravel()
    nem = pd.read_csv(os.path.join(A, "nem_k3.csv")); nem = nem[nem.口径.str.startswith("规模 ≤4")].set_index("目标")
    rows = []
    for _, r in Sm.iterrows():
        m23 = float(nem.loc["机组出力_" + r["目标"].split("_", 1)[1], "M2/M3"]) if r.ds == "NEM" else r["K饱和_M2除M3"]
        rows.append((r["name"], r["K饱和_M1除M2"], m23))
    y = np.arange(len(rows))[::-1]; ax = axs[0]
    ax.scatter([r[1] for r in rows], y, s=16, facecolor="white", edgecolor=S.ORANGE, lw=1.1, label="$M^{(1)}/M^{(2)}$", zorder=3)
    ax.scatter([r[2] for r in rows], y, s=16, color=S.BLUE, label="$M^{(2)}/M^{(3)}$", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.4); ax.set_xlim(0.65, 1.02)
    ax.set_xlabel("ratio of mean risk", fontsize=7.4); ax.legend(loc="lower left", fontsize=7.4, handletextpad=0.1); ax.grid(axis="x", color=S.GRID, lw=0.5)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); S.panel(ax, "a", x=-0.85)
    ax = axs[1]; K = pd.read_csv(os.path.join(A, "k3_critical.csv")); K = K[K.τ == 0.7].reset_index(drop=True)
    yk = np.arange(len(K))[::-1]
    for yy, (_, r) in zip(yk, K.iterrows()):
        ax.plot([r.K2关键 / r.字段数, r.K3关键 / r.字段数], [yy, yy], color="#cfcfcf", lw=1.6, zorder=1)
    ax.scatter(K.K2关键 / K.字段数, yk, s=16, color=S.BLUE, zorder=3, label="$K=2$")
    ax.scatter(K.K3关键 / K.字段数, yk, s=16, facecolor="white", edgecolor=S.VIOLET, lw=1.1, zorder=3, label="$K=3$")
    ax.set_yticks(yk); ax.set_yticklabels([N.target(t) for t in K.目标], fontsize=7.4); ax.set_xlim(-0.03, 1.03); ax.set_xlabel("critical fields at τ = 0.7", fontsize=7.4); ax.legend(loc="upper left", fontsize=7.4)
    ax.grid(axis="x", color=S.GRID, lw=0.5); ax.spines["left"].set_visible(False); S.panel(ax, "b", x=-0.1)
    ax = axs[2]; C = pd.read_csv(os.path.join(A, "robust_consistency.csv"))
    mets = [("M2排序Spearman", "Spearman\nof $M^{(2)}$"), ("关键集合Jaccard_τ05", "Jaccard,\nτ = 0.5"), ("关键集合Jaccard_τ07", "Jaccard,\nτ = 0.7")]
    for j, (col, _) in enumerate(mets):
        for kind, mk, colr in [("种子", "o", S.BLUE), ("划分", "s", S.ORANGE)]:
            v = C[C.版本.str.startswith(kind)][col]
            ax.scatter(v, j + (0.12 if kind == "划分" else -0.12) + np.linspace(-0.05, 0.05, len(v)), s=9, marker=mk, color=colr, zorder=3,
                       label=("attacker seed" if kind == "种子" else "data split") if j == 0 else None)
    ax.set_yticks(range(3)); ax.set_yticklabels([l for _, l in mets], fontsize=7.4); ax.invert_yaxis(); ax.set_xlim(0.88, 1.01)
    ax.set_xlabel("agreement with main run", fontsize=7.4); ax.grid(axis="x", color=S.GRID, lw=0.5); ax.tick_params(axis="y", length=0)
    ax.legend(loc="center left", fontsize=7.4, handletextpad=0.1, bbox_to_anchor=(0, 0.72)); S.panel(ax, "c", x=-0.55)
    ax = axs[3]; T = pd.read_csv(os.path.join(A, "tau_sweep.csv"))
    m = T.groupby("τ").apply(lambda e: pd.Series(dict(c=(e.组合危险字段 / e.字段数).mean(), s=(e.单字段即危险 / e.字段数).mean(), l=e.仍暴露比例.mean())))
    ax.plot(m.index, m.c, color=S.BLUE, lw=1.5, label="safe alone, critical")
    ax.plot(m.index, m.s, color=S.ORANGE, lw=1.3, ls="--", label="unsafe alone")
    ax.plot(m.index, m.l, color=S.MUTED, lw=1.2, ls=":", label="MUS left after\nsingle-field grading")
    ax.set_xlabel("threshold τ", fontsize=7.4); ax.set_ylim(0, 1.02); ax.set_ylabel("share", fontsize=7.4); S.ygrid(ax)
    ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.45, -0.28), handlelength=1.6); S.panel(ax, "d", x=-0.4)
    S.save(fig, os.path.join(OUT, "fig9_robustness"))


# ------------------------------------------------------------------ 图 10：成本


# ------------------------------------------------------------------ 背景放大（124 号）
def fig_amplification():
    A=R('DNN_Aggresvation124/outputs');P=pd.read_csv(os.path.join(A,'profile.csv'));Dg=pd.read_csv(os.path.join(A,'context_gain_distribution.csv'))
    dsmap={'RTS-GMLC':'RTS-GMLC','NEM':'NEM','PJM-load':'PJM','PJM-gen/ic':'PJM','CAISO-load':'CAISO'}
    fig,axs=plt.subplots(2,1,figsize=(S.TEXTW,5.25),gridspec_kw={'height_ratios':[1,1.25],'hspace':.55})
    ax=axs[0]
    for ds in ['RTS-GMLC','NEM','PJM','CAISO']:
        q=P[P.数据.map(dsmap)==ds];ax.scatter(q.M0,q.G2,s=18,color=DSCOL[ds],alpha=.8,edgecolor='white',lw=.35,label=ds)
    ax.axvline(.3,color=S.MUTED,lw=.8,ls=':');ax.axhline(.1,color=S.MUTED,lw=.8,ls=':')
    ax.set(xlim=(0,1),ylim=(0,.46),xlabel=r'Single-field risk $M^{(0)}$',ylabel=r'Amplification $\Gamma^{(2)}$')
    ax.text(.04,.96,'91 / 264 pairs: low alone, amplified',transform=ax.transAxes,va='top',fontsize=8)
    ax.legend(loc='upper right',ncol=2,fontsize=8);S.ygrid(ax);S.panel(ax,'a','Amplification across all fields',x=-.02)
    ax=axs[1]
    pick=P[P.数据.isin(Dg.数据.unique())].sort_values('G2',ascending=False).drop_duplicates(['数据','目标']).head(6)
    for k,(_,r) in enumerate(pick.iterrows()):
        vals=Dg[(Dg.数据==r.数据)&(Dg.目标==r.目标)&(Dg.字段==r.字段)].Δ.values
        jitter=np.random.RandomState(k).uniform(-.23,.23,len(vals))
        ax.scatter(vals,k+jitter,s=2,alpha=.28,color=DSCOL[dsmap[r.数据]],lw=0)
        ax.plot(vals.mean(),k,'|',ms=13,color=S.INK,mew=1.6)
        ax.plot(r.M0,k,'o',ms=5.5,mfc='white',mec=S.ORANGE,mew=1.2)
        ax.plot(vals.max(),k,'o',ms=5.5,color=S.BLUE)
    ax.set_yticks(range(len(pick)));ax.set_yticklabels([N.field(r.字段)+'\n→ '+N.target(r.目标) for _,r in pick.iterrows()],fontsize=8)
    ax.invert_yaxis();ax.set_xlabel(r'Contextual gain $\Delta_i(T)$, all $|T|\leq2$');ax.grid(axis='x');ax.tick_params(axis='y',length=0)
    ax.plot([],[],'o',ms=5.5,mfc='white',mec=S.ORANGE,label='Alone');ax.plot([],[],'|',ms=10,color=S.INK,label='Average');ax.plot([],[],'o',ms=5.5,color=S.BLUE,label='Maximum')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8)
    S.panel(ax,'b','Different backgrounds, different gains',x=-.02)
    S.save(fig,os.path.join(OUT,'fig_amplification'))


# ------------------------------------------------------------------ 受控机制算例（123 号）
MECH_SHORT = {"P_g (unit 313 output)": "unit output $P_g$", "LMP_g (bus 313)": "own-bus LMP", "LMP_near (bus 306)": "nearby LMP (306)",
              "LMP_far (bus 303, behind C6)": "remote LMP (303, behind C6)", "LMP_area2 (bus 223)": "other-area LMP (223)",
              "System load": "system load", "Area-3 load": "area-3 load", "Wind available": "wind available", "PV available": "PV available",
              "Congestion C6": "congestion C6", "P_comp (unit 316 output)": "competitor output (316)"}


def fig_mechanism(tag='narrow'):
    A=R('DNN_Aggresvation123/outputs',tag,'analysis')
    G=pd.read_csv(os.path.join(A,'contextual_gain_P.csv')).iloc[::-1].reset_index(drop=True)
    regimes=pd.read_csv(os.path.join(A,'regime.csv')).set_index('背景')
    fig,axs=plt.subplots(1,2,figsize=(S.TEXTW,3.25),gridspec_kw={'width_ratios':[1.25,1],'wspace':.62})
    ax=axs[0]
    labels=[('none' if b=='∅' else MECH_SHORT[b].replace(' (303, behind C6)',' (303)')) for b in G.背景]
    colors=[S.BLUE if 'LMP' in b else S.AQUA if 'P_comp' in b else S.MUTED for b in G.背景]
    ax.barh(range(len(G)),G.Δ_P,.65,color=colors)
    ax.set_yticks(range(len(G)));ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel(r'Gain of $P_g$ over background');ax.set_xlim(0,.25);ax.tick_params(axis='y',length=0);ax.grid(axis='x');ax.set_axisbelow(True)
    S.panel(ax,'a','Context changes the gain',x=-.55)
    ax=axs[1];xs=np.arange(2);w=.35
    for k,(reg,color,label) in enumerate([('C6 不阻塞',S.LIGHTBLUE,'Uncongested'),('C6 阻塞',S.BLUE,'C6 congested')]):
        vals=[regimes.loc[b,f'{reg}_V(T+P)']-regimes.loc['∅',f'{reg}_V(T+P)'] for b in ['LMP_g','LMP_far']]
        ax.bar(xs+(k-.5)*w,vals,w*.9,color=color,label=label)
    ax.set_xticks(xs);ax.set_xticklabels(['Own-bus price','Remote price'],fontsize=8)
    ax.set_ylabel(r'Amplification of $P_g$');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=8)
    S.ygrid(ax);S.panel(ax,'b','Congestion matters',x=-.17)
    S.save(fig,os.path.join(OUT,f'fig_mechanism_{tag}'))
    Sc=pd.read_csv(os.path.join(A,'method_scores.csv'))
    cols=[('pearson','Pearson'),('mi','MI'),('M0','Single field'),('loco','LOCO'),('perm','Permutation'),('sage','SAGE'),('graph','Graph'),('M2',r'$M^{(2)}$')]
    order=['P_g (unit 313 output)','LMP_g (bus 313)','LMP_near (bus 306)','LMP_area2 (bus 223)','LMP_far (bus 303, behind C6)','P_comp (unit 316 output)','System load','PV available','Area-3 load','Wind available','Congestion C6']
    M=Sc.set_index('字段').loc[order,[c for c,_ in cols]].values.astype(float)
    Mpos=np.clip(M,0,None);Mn=Mpos/np.maximum(Mpos.max(axis=0,keepdims=True),1e-9)
    fig,ax=plt.subplots(figsize=(S.TEXTW,3.9));S.heatmap(ax,Mn,SEQ)
    for i in range(len(order)):
        for j in range(len(cols)):
            val=0 if abs(M[i,j])<.005 else M[i,j]
            ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=8,color='white' if Mn[i,j]>.6 else S.INK)
    ax.set_xticks(range(len(cols)));ax.set_xticklabels([l for _,l in cols],rotation=30,ha='left',fontsize=8);ax.xaxis.tick_top()
    ax.set_yticks(range(len(order)));ax.set_yticklabels([MECH_SHORT[o].replace(' (303, behind C6)',' (303)') for o in order],fontsize=8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.add_patch(plt.Rectangle((6.5,-.5),1,len(order),fill=False,ec=S.BLUE,lw=1.3))
    S.save(fig,os.path.join(OUT,'fig_mechanism_scores'))


# ------------------------------------------------------------------ 掩码预训练诊断（122 号）
def fig_masking():
    A = R("DNN_Aggresvation122/outputs/analysis")
    K = pd.read_csv(os.path.join(A, "rank_truncation.csv")); K["秩"] = K["秩"].astype(str)
    V = pd.read_csv(os.path.join(A, "variants_by_target.csv")); L = pd.read_csv(os.path.join(A, "large_sets.csv"))
    Dc = pd.read_csv(os.path.join(A, "error_decomposition.csv"))
    SM = {"recon+random@1_cy": "recon+random_cy", "recon+random@2_cy": "recon+random_cy"}   # 单种子拼接取三种子平均
    V = V.replace({"变体": SM}); Dc = Dc.replace({"变体": SM})
    D5 = pd.read_csv(R("DNN_Aggresvation125/outputs/select/error_decomposition.csv"))            # 本文估计器（125 号 RRE_cy）
    S5 = pd.read_csv(R("DNN_Aggresvation125/outputs/select/candidates_by_target.csv"))
    Dc = pd.concat([Dc, D5[D5.变体 == "RRE_cy"]], ignore_index=True)
    V = pd.concat([V, S5[S5.候选 == "C3"].assign(变体="RRE_cy")], ignore_index=True)
    BB = [("random", "untrained", S.GRAYS[0]), ("uniform", "target-only", S.ORANGE), ("recon", "target + reconstruction", S.BLUE)]
    fig = plt.figure(figsize=(S.TEXTW, 5.0)); grid = fig.add_gridspec(2, 2, wspace=.38, hspace=.7); gs = [grid[0,0],grid[0,1],grid[1,:]]
    ax = fig.add_subplot(gs[0]); ranks = ["0", "2", "4", "8", "32", "全部"]; xr = {r: k for k, r in enumerate(ranks)}
    g = K.groupby(["主干", "秩"]).V误差.mean()
    for bb, lab, col in BB:
        rr = [r for r in ranks if (bb, r) in g.index]
        ax.plot([xr[r] for r in rr], [g[(bb, r)] for r in rr], "-o", color=col, lw=1.3, ms=3.2, label=lab, mec="white", mew=0.4)
    ax.set_xticks(range(len(ranks))); ax.set_xticklabels(["0", "2", "4", "8", "32", "256"]); ax.set_xlabel("principal components of $\\phi$ kept")
    ax.set_ylabel("error of $\\hat V$ ($|S|\\leq3$)"); S.ygrid(ax); S.panel(ax, "a", x=-0.3)
    ax.legend(fontsize=7.4, loc="upper right", handlelength=1.4)
    ax = fig.add_subplot(gs[1])
    for bb, lab, col in BB:
        y = [V[V.变体 == bb].V误差_3.mean(), V[V.变体 == bb].V误差_4.mean()] + [L[(L.变体 == bb) & (L.规模 == s)].V误差.mean() for s in (6, 10, 16)]
        ax.plot(range(5), y, "-o", color=col, lw=1.3, ms=3.2, mec="white", mew=0.4)
    ax.axvspan(1.5, 4.3, color="#f3f3f3", zorder=0, lw=0)
    ax.set_xticks(range(5)); ax.set_xticklabels(["$\\leq$3", "4", "6", "10", "16"]); ax.set_xlabel("set size $|S|$"); ax.set_ylabel("error of $\\hat V$")
    S.ygrid(ax); S.panel(ax, "b", x=-0.3)
    ax = fig.add_subplot(gs[2]); vs = ["reconE", "reconE_cy", "recon+random_cy", "RRE_cy"]
    lab = {"reconE": "reconstr. ×3", "reconE_cy": "reconstr. ×3 + clip", "recon+random_cy": "reconstr. + random ×1 + clip",
           "RRE_cy": "proposed: reconstr. + random ×3 + clip"}
    col = {"reconE": "#dbe9f8", "reconE_cy": S.LIGHTBLUE, "recon+random_cy": "#3987e5", "RRE_cy": "#0d366b"}
    mets = [("集合误差", Dc, "$V$\nerror"), ("边际误差", Dc, "$\\Delta$\nerror"), ("字段最大高估", Dc, "max over-\nestimate"), ("M2误差", V, "$M^{(2)}$\nerror")]
    w = 0.8 / len(vs)
    for k, v in enumerate(vs):
        vals = [src[src.变体 == v][c].mean() / src[src.变体 == "uniformE"][c].mean() for c, src, _ in mets]
        ax.bar(np.arange(len(mets)) + (k - len(vs) / 2 + 0.5) * w, vals, w * 0.92, color=col[v], label=lab[v])
    ax.axhline(1, color=S.INK2, lw=0.6); ax.set_xticks(range(len(mets))); ax.set_xticklabels([m for _, _, m in mets], fontsize=7.4)
    ax.set_ylabel("relative to target-only ×3"); ax.set_ylim(0.5, 1.9); S.ygrid(ax); S.panel(ax, "c", x=-0.2)
    ax.legend(fontsize=7.4, ncol=1, loc="upper left", handlelength=1.0)
    S.save(fig, os.path.join(OUT, "fig_masking"))


# ------------------------------------------------------------------ 主干复用范围（原图 8 的 d–f）
def fig_scope():
    B = pd.read_csv(R("DNN_Aggresvation116/outputs/analysis/backbone_compare.csv"))
    order_b = ["本组主干", "全列主干", "留目标主干"]; lab_b = ["per-group", "all-column", "target held out"]; cb = ["#8c8c8c", S.BLUE, S.ORANGE]
    m = B.groupby("主干")[["死神经元比例", "有效维度", "V绝对误差", "边际Δ绝对误差", "M2误差", "认证前1", "认证前3"]].mean().reindex(order_b)
    fig, axs = plt.subplots(1, 3, figsize=(S.TEXTW, 2.9), gridspec_kw=dict(wspace=0.45)); x = np.arange(3); w = 0.26
    ax = axs[0]; ax.bar(x, m["死神经元比例"] * 100, 0.6, color=cb)
    for xi, v, dm in zip(x, m["死神经元比例"] * 100, m["有效维度"]):
        ax.text(xi, v + 0.8, f"eff. dim\n{dm:.2f}", ha="center", fontsize=7.4, color=S.INK2)
    ax.set_ylim(0, 36); ax.set_xticks(x); ax.set_xticklabels(["per-\ngroup", "all-\ncolumn", "target\nheld out"], fontsize=7.4); ax.set_ylabel("inactive units (%)", fontsize=7.4); S.ygrid(ax); S.panel(ax, "a", x=-0.3)
    ax = axs[1]
    for j in range(3):
        ax.bar(x + (j - 1) * w, m.iloc[j][["V绝对误差", "边际Δ绝对误差", "M2误差"]].values, w * 0.9, color=cb[j], label=lab_b[j])
    ax.set_xticks(x); ax.set_xticklabels(["$V$", "marginal $\\Delta$", "$M^{(2)}$"], fontsize=7.4); ax.set_ylabel("MAE", fontsize=7.4)
    ax.set_ylim(0, 0.085); ax.legend(fontsize=7.4, loc="upper left", ncol=1); S.ygrid(ax); S.panel(ax, "b", x=-0.3)
    ax = axs[2]
    for j in range(3):
        ax.bar(np.arange(2) + (j - 1) * w, m.iloc[j][["认证前1", "认证前3"]].values, w * 0.9, color=cb[j])
    ax.set_xticks(range(2)); ax.set_xticklabels(["top-1", "top-3"], fontsize=7.4); ax.set_ylim(0.6, 1.0); ax.set_ylabel("certified ratio", fontsize=7.4)
    S.ygrid(ax); S.panel(ax, "c", x=-0.3)
    S.save(fig, os.path.join(OUT, "fig_scope"))


# ------------------------------------------------------------------ 与外部方法对比（127 号）
def fig_baselines():
    from data import _baseline_times
    T=_baseline_times();M=pd.read_csv(R('DNN_Aggresvation127/outputs/analysis/summary.csv'),index_col=0)
    names={'lin':'Linear','poly':'Quadratic','dropout':'Mean substitution','lazyvi':'LazyVI','ws':'Warm start','surrogate':'Masked surrogate','tabpfn':'TabPFN v2','ours':'Proposed','retrain':'Retraining'}
    sty={'lin':(S.MUTED,'v'),'poly':(S.MUTED,'^'),'dropout':(S.MUTED,'X'),'ws':(S.MUTED,'D'),'surrogate':(S.MUTED,'P'),'tabpfn':(S.ORANGE,'o'),'ours':(S.BLUE,'o'),'retrain':(S.INK,'s'),'lazyvi':(S.MUTED,'*')}
    tm={k:np.sqrt(T[k][0]*T[k][1]) for k in T}
    fig=plt.figure(figsize=(S.TEXTW,5.15));gs=fig.add_gridspec(2,2,height_ratios=[1,1.1],wspace=.38,hspace=.95)
    for idx,col,lab,lim in [(0,'M2误差',r'MAE of $M^{(2)}$',(-.005,.26)),(1,'认证前1','Certified ratio (top-1)',(.25,1.04))]:
        ax=fig.add_subplot(gs[0,idx])
        for k in ['lin','poly','dropout','ws','surrogate','tabpfn','ours','retrain']:
            val=(0 if idx==0 else 1) if k=='retrain' else M.loc[k,col];cl,mk=sty[k]
            ax.errorbar(tm[k],val,xerr=[[tm[k]-T[k][0]],[T[k][1]-tm[k]]],fmt=mk,ms=5.5,color=cl,elinewidth=1,label=names[k])
        ax.set(xscale='log',xlim=(5e-5,100),ylim=lim,xlabel='Time per field set (s)',ylabel=lab)
        ax.set_xticks([.0001,.01,1,100]);S.ygrid(ax);S.panel(ax,'ab'[idx],['Risk error','Recovered risk'][idx],x=-.04)
        if idx==0:handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.5,.49),ncol=4,fontsize=7.8,handletextpad=.35,columnspacing=.8)
    ax=fig.add_subplot(gs[1,:]);ks=sorted(['dropout','lin','poly','surrogate','ws','lazyvi','ours','tabpfn'],key=lambda k:-M.loc[k,'样本V误差'])
    vals=[M.loc[k,'样本V误差'] for k in ks];yy=np.arange(len(ks));ax.barh(yy,vals,.64,color=[sty[k][0] if k in ['ours','tabpfn'] else '#B4BFC6' for k in ks])
    for y,v in zip(yy,vals):ax.text(v+.006,y,f'{v:.3f}',va='center',fontsize=8)
    ax.set_yticks(yy);ax.set_yticklabels([names[k] for k in ks],fontsize=8.5);ax.invert_yaxis();ax.set_xlim(0,.52)
    ax.set_xlabel(r'MAE of $V$ (same 200 sets per data set)');ax.grid(axis='x');ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    S.panel(ax,'c','Matched evaluation including LazyVI',x=-.04)
    S.save(fig,os.path.join(OUT,'fig_baselines'))

if __name__ == "__main__":
    todo = sys.argv[1:] or ["toy", "combination"]
    for t in todo:
        (fig_mechanism("wide") if t == "mechanism_wide" else globals()[f"fig_{t}"]()); print("done", t)
