# -*- coding: utf-8 -*-
"""V3_en figures. Reads experiment artifacts directly from DNN_Aggresvation100-109.

Usage:  python scripts/make_figs.py
Outputs: figures/fig3..fig9 (English labels, serif, print-ready).
"""
from pathlib import Path
import glob, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]                       # .../GNN_Aggresvation
F = HERE / "figures"; F.mkdir(exist_ok=True)
R = {n: REPO / f"DNN_Aggresvation{n}" for n in (100, 103, 104, 105, 106, 108, 109)}
sys.path.insert(0, str(R[100] / "src")); sys.path.insert(0, str(R[100] / "scripts"))

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e3e0",
    "grid.linewidth": .6, "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE, ORANGE, GREEN, RED, GRAY, INK = "#2a6fb5", "#d9622b", "#1b9e77", "#c0392b", "#9a9a95", "#222220"
MK = {"pjm": ("PJM", BLUE, "o"), "caiso": ("CAISO", ORANGE, "s")}


def _shards(pat, n, key, ncol=12):
    fs = sorted(glob.glob(pat))
    if not fs or len(fs) != int(fs[0].split("of")[-1].split(".")[0]):
        return None
    out = np.zeros((n, ncol), np.float32)
    for f in fs:
        z = np.load(f); out[z["idx"]] = z[key]
    return out


# ---------------------------------------------------------------- Fig 3: V fidelity
def fig3():
    from mkfull import closure
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, ds in enumerate(["pjm", "caiso"]):
        meta = pickle.load(open(R[100] / f"outputs/sets/{ds}_meta.pkl", "rb"))
        keys = meta["keys_k3"]; n = len(keys)
        Vt = np.load(R[103] / f"outputs/analysis/{ds}_truth_official.npz")["V"]
        E = _shards(str(R[103] / f"outputs/est/{ds}_k3_L0ensx_s*of3.npz"), n, "v")
        Eb = closure(np.clip(E, 0, 1), keys)
        sz = np.array([len(x) for x in keys])
        rs = np.random.RandomState(0)
        idx = rs.choice(n, min(2500, n), replace=False)
        name, col, _ = MK[ds]
        for s, a in [(1, .9), (2, .55), (3, .3)]:
            m = idx[sz[idx] == s]
            ax[k].scatter(Vt[m].ravel()[::3], Eb[m].ravel()[::3], s=1.4, alpha=a,
                          color=col, linewidths=0, rasterized=True)
        ax[k].plot([0, 1], [0, 1], color=INK, lw=.8, ls="--")
        ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1)
        ax[k].set_xlabel("dedicated retraining $V_y(S)$")
        ax[k].set_title(f"({chr(97+k)}) {name}", loc="left")
        mae = np.abs(Eb - Vt).mean()
        ax[k].text(.05, .93, f"MAE = {mae:.4f}", fontsize=8, color=INK)
    ax[0].set_ylabel("amortized estimate")
    # panel c: MAE by set size, main vs shared head
    d = pd.read_csv(R[103] / "outputs/analysis/103_C.csv") if (R[103] / "outputs/analysis/103_C.csv").exists() else None
    if d is not None:
        sub = {"shared output head": "共享输出头（direct）", "frozen $\\phi$ + closed-form": "φ+x,x²（三种子集成）"}
        w = .2; xs = np.arange(3)
        for j, (lab, key) in enumerate(sub.items()):
            for k, ds in enumerate(["pjm", "caiso"]):
                e = d[(d.数据集 == ds) & (d.估计器 == key)].set_index("规模")
                if len(e) == 0: continue
                col = [GRAY, MK[ds][1]][j]
                ax[2].bar(xs + (2 * k + j - 1.5) * w, e.loc[[1, 2, 3], "V_MAE"], w * .9,
                          color=col, alpha=1 if j else .75,
                          label=f"{MK[ds][0]}, {lab}" if True else None)
        ax[2].set_xticks(xs); ax[2].set_xticklabels(["$|S|=1$", "$|S|=2$", "$|S|=3$"])
        ax[2].set_ylabel("$V_y$ MAE"); ax[2].grid(axis="x", visible=False)
        ax[2].legend(frameon=False, fontsize=5.8, loc="upper left", handlelength=1.1)
        ax[2].set_title("(c) error by set size", loc="left")
    fig.tight_layout(); fig.savefig(F / "fig3_v_fidelity.pdf"); fig.savefig(F / "fig3_v_fidelity.png", dpi=170); plt.close(fig)
    print("fig3 done")


# ---------------------------------------------------------------- Fig 4: M interval
def fig4():
    B = pd.read_csv(R[108] / "outputs/analysis/108_A1_bounds.csv")
    C = pd.read_csv(R[108] / "outputs/analysis/108_A1_coverage.csv")
    Kc = pd.read_csv(R[108] / "outputs/analysis/108_A1_critical_interval.csv")
    T = pd.read_csv(R[104] / "outputs/analysis/M_table_official.csv")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) estimated vs exact M, K=2
    d = T[T.K == 2]
    for ds in ["pjm", "caiso"]:
        e = d[d.数据集 == ds]; name, col, _ = MK[ds]
        ax[0].scatter(e.exact_M, e["est_M_集成"], s=3, alpha=.45, color=col, linewidths=0,
                      label=name, rasterized=True)
    ax[0].plot([0, 1], [0, 1], color=INK, lw=.8, ls="--")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("exact $M^{(2)}$"); ax[0].set_ylabel("estimated $M^{(2)}$")
    ax[0].legend(frameon=False, loc="upper left", markerscale=2.5)
    ax[0].set_title("(a) point estimate", loc="left")
    # (b) certified lower bound top1 vs top3
    x = np.arange(3); w = .18
    for k, ds in enumerate(["pjm", "caiso"]):
        e = B[B.数据集 == ds].set_index("K"); name, col, _ = MK[ds]
        ax[1].bar(x + (2 * k - 1.5) * w, e.认证下界比_top1, w * .9, color=col, alpha=.45,
                  label=f"{name}, top-1")
        ax[1].bar(x + (2 * k - .5) * w, e.认证下界比_top3, w * .9, color=col,
                  label=f"{name}, top-3")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"$K={i}$" for i in x])
    ax[1].set_ylim(.78, 1.12); ax[1].set_ylabel(r"certified ratio $\sum L/\sum M$")
    ax[1].grid(axis="x", visible=False)
    ax[1].legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center", columnspacing=.8, handlelength=1.2)
    ax[1].set_title("(b) witness certification", loc="left")
    # (c) critical recall: point vs interval
    taus = [0.5, 0.7, 0.9]; xs = np.arange(3); w = .2
    for k, ds in enumerate(["pjm", "caiso"]):
        e = Kc[(Kc.数据集 == ds) & (Kc.alpha == .05) & (Kc.K == 2)].set_index("tau")
        name, col, _ = MK[ds]
        ax[2].bar(xs + (2 * k - 1.5) * w, e.loc[taus, "点估计recall"], w * .9, color=GRAY,
                  alpha=1 - .35 * k, label=f"{name}, point")
        ax[2].bar(xs + (2 * k - .5) * w, e.loc[taus, "区间保守recall"], w * .9, color=col,
                  label=f"{name}, interval")
    ax[2].set_xticks(xs); ax[2].set_xticklabels([rf"$\tau={t}$" for t in taus])
    ax[2].set_ylim(0, 1.42); ax[2].set_ylabel(r"$\tau$-critical recall")
    ax[2].grid(axis="x", visible=False)
    ax[2].legend(frameon=False, fontsize=6.0, ncol=2, loc="upper center", columnspacing=.8, handlelength=1.2)
    ax[2].set_title("(c) $K=2$ detection", loc="left")
    fig.tight_layout(); fig.savefig(F / "fig4_m_interval.pdf"); fig.savefig(F / "fig4_m_interval.png", dpi=170); plt.close(fig)
    print("fig4 done")


# ---------------------------------------------------------------- Fig 5: escalation
def fig5():
    T = pd.read_csv(R[104] / "outputs/analysis/M_table_official.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, ds in enumerate(["pjm", "caiso"]):
        d = T[T.数据集 == ds]; name, col, _ = MK[ds]
        piv = d.pivot_table(index=["目标", "字段"], columns="K", values="exact_M")
        piv = piv.dropna().sort_values(2, ascending=False)
        for _, row in piv.iloc[::17].iterrows():
            ax[k].plot([0, 1, 2], row.values, color=col, alpha=.22, lw=.7)
        ax[k].plot([0, 1, 2], piv.mean().values, color=INK, lw=2.2, marker="o", ms=4,
                   label="mean")
        q = piv.quantile([.25, .75])
        ax[k].fill_between([0, 1, 2], q.iloc[0].values, q.iloc[1].values, color=col,
                           alpha=.18, lw=0, label="IQR")
        ax[k].set_xticks([0, 1, 2]); ax[k].set_xticklabels(["$M^{(0)}$", "$M^{(1)}$", "$M^{(2)}$"])
        ax[k].set_ylim(0, 1); ax[k].grid(axis="x", visible=False)
        ax[k].set_title(f"({chr(97+k)}) {name}", loc="left")
        ax[k].legend(frameon=False, loc="upper left")
        m = piv.mean().values
        ax[k].text(1.35, .06, f"{m[0]:.3f} $\\to$ {m[1]:.3f} $\\to$ {m[2]:.3f}", fontsize=7.5, color=INK)
    ax[0].set_ylabel("budgeted marginal capability")
    fig.tight_layout(); fig.savefig(F / "fig5_escalation.pdf"); fig.savefig(F / "fig5_escalation.png", dpi=170); plt.close(fig)
    print("fig5 done")


# ---------------------------------------------------------------- Fig 7: release audit
def fig7():
    S = pd.read_csv(R[106] / "outputs/analysis/release_audit_summary.csv")
    order = ["单字段放行", "相关性放行", "直接估计放行", "逐阶预算放行", "预算放行_估计", "预算放行"]
    lab = ["single-field", "correlation", r"direct $\hat V(A)$",
           r"staged $U^{(2)}$", r"additive $\sum\hat M$", r"additive $\sum M$"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(order)); w = .36
    for k, (col, ttl) in enumerate([("危险放行率", "(a) dangerous release rate"),
                                    ("误拒率", "(b) false rejection rate")]):
        for j, ds in enumerate(["pjm", "caiso"]):
            d = S[S.数据集 == ds].set_index("审查规则").reindex(order)
            name, c, _ = MK[ds]
            ax[k].bar(x + (j - .5) * w, d[col], w * .9, color=c, label=name)
            for xi, v in zip(x + (j - .5) * w, d[col]):
                if v > .004:
                    txt = f"{v:.0%}" if v >= .1 else f"{v:.1%}"
                    dx, ha = ((-.04, "right") if j == 0 else (.04, "left")) if v < .1 else (0, "center")
                    ax[k].text(xi + dx, v + .015, txt, ha=ha, fontsize=6.3)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(lab, fontsize=6.6, rotation=28, ha="right")
        ax[k].grid(axis="x", visible=False); ax[k].set_title(ttl, loc="left")
        ax[k].set_ylim(0, [.68, 1.05][k])
    ax[0].legend(frameon=False); ax[0].set_ylabel("fraction")
    ax[0].annotate("current practice", (0.18, .564), xytext=(1.3, .60), fontsize=6.8,
                   color=RED, ha="left",
                   arrowprops=dict(arrowstyle="->", color=RED, lw=.8))
    fig.tight_layout(); fig.savefig(F / "fig7_release_audit.pdf"); fig.savefig(F / "fig7_release_audit.png", dpi=170); plt.close(fig)
    print("fig7 done")


# ---------------------------------------------------------------- Fig 8: efficiency
def fig8():
    q = np.array([1e3, 1e4, 1e5])
    # main estimator = 3-seed ensemble (L0ensx); light variant = single seed (L0)
    seq_ms = {"pjm": 21800., "caiso": 29600.}
    bat_ms = {"pjm": 56.6, "caiso": 74.3}
    amo_ms = {"pjm": 28.0, "caiso": 35.2}          # 3-seed ensemble query cost
    tr_s = {"pjm": 27. * 3, "caiso": 40. * 3}      # 3 backbones
    lite_ms = {"pjm": 2.4, "caiso": 3.1}           # single seed
    lite_s = {"pjm": 27., "caiso": 40.}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for k, ds in enumerate(["pjm", "caiso"]):
        name, col, _ = MK[ds]
        ax[k].loglog(q, q * seq_ms[ds] / 1000 / 3600, "o-", color=GRAY, ms=4, label="sequential retraining")
        ax[k].loglog(q, q * bat_ms[ds] / 1000 / 3600, "s-", color=col, ms=4, label="batched retraining")
        ax[k].loglog(q, (tr_s[ds] + q * amo_ms[ds] / 1000) / 3600, "^-", color=GREEN, ms=4,
                     label="amortized, 3-seed (main)")
        ax[k].loglog(q, (lite_s[ds] + q * lite_ms[ds] / 1000) / 3600, "v--", color=GREEN, ms=3.5,
                     alpha=.55, label="amortized, 1-seed")
        be = {"pjm": 2816, "caiso": 3082}[ds]        # measured, see Table 5
        ax[k].axvline(be, color=RED, lw=.9, ls=":")
        ax[k].text(be * 1.15, 6.5e-3, f"break-even\n$\\approx${be:,}", fontsize=6.3, color=RED)
        ax[k].axvline(11521, color=INK, lw=.9, ls="--")
        ax[k].text(11521 * 1.25, 2.2e3, "full\n$K{=}2$", fontsize=6.3, color=INK, va="top")
        ax[k].set_ylim(4e-3, 6e3)
        ax[k].set_xlabel("number of subset queries")
        ax[k].set_title(f"({chr(97+k)}) {name}", loc="left")
        ax[k].grid(True, which="both", alpha=.4)
    ax[0].set_ylabel("wall-clock time (hours)")
    ax[0].legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.4,
                 borderpad=.2, labelspacing=.3)
    fig.tight_layout(); fig.savefig(F / "fig8_efficiency.pdf"); fig.savefig(F / "fig8_efficiency.png", dpi=170); plt.close(fig)
    print("fig8 done")


# ---------------------------------------------------------------- Fig 9: attacker capability
def fig9():
    M = pd.read_csv(R[109] / "outputs/analysis/109_B_M.csv")
    Rl = pd.read_csv(R[109] / "outputs/analysis/109_D_release.csv")
    W = pd.read_csv(R[109] / "outputs/analysis/109_E_winners_curse.csv")
    FR = ["10%", "25%", "50%", "100%"]; x = np.arange(4)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) M ratio
    for ds in ["pjm", "caiso"]:
        name, col, mk = MK[ds]
        for K, ls in [(0, ":"), (2, "-")]:
            e = M[(M.数据集 == ds) & (M.K == K)].set_index("比例").reindex(FR)
            ax[0].plot(x, e.M相对full, ls, marker=mk, color=col, ms=3.5, lw=1.5,
                       label=f"{name}, $M^{{({K})}}$")
    ax[0].axhline(1, color=INK, lw=.8, ls="--")
    ax[0].set_xticks(x); ax[0].set_xticklabels(FR); ax[0].set_xlabel("adversary's auxiliary labels")
    ax[0].set_ylabel("$M$ relative to full data"); ax[0].grid(axis="x", visible=False)
    ax[0].legend(frameon=False, fontsize=6.4, loc="lower right")
    ax[0].set_title("(a) $M^{(2)}$ is inflated", loc="left")
    # (b) selection gain vs candidate count
    for ds in ["pjm", "caiso"]:
        name, col, mk = MK[ds]
        e = W[(W.数据集 == ds) & (W.比例 == "10%")].set_index("K")
        ax[1].plot(e.候选背景数, e.选择增益, marker=mk, color=col, ms=4, lw=1.5, label=name)
        for K in [0, 1, 2]:
            ax[1].annotate(f"$K={K}$", (e.loc[K, "候选背景数"], e.loc[K, "选择增益"]),
                           textcoords="offset points", xytext=(5, 4), fontsize=6.2, color=col)
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of candidate backgrounds")
    ax[1].set_ylabel("selection gain"); ax[1].set_ylim(-.006, .095)
    ax[1].legend(frameon=False, loc="upper left")
    ax[1].set_title("(b) winner's curse mechanism", loc="left")
    # (c) dangerous release
    w = .2
    for j, tau in enumerate([0.5, 0.7]):
        for k, ds in enumerate(["pjm", "caiso"]):
            e = Rl[(Rl.数据集 == ds) & (Rl.tau == tau)].set_index("比例").reindex(FR)
            name, col, _ = MK[ds]
            ax[2].bar(x + (2 * j + k - 1.5) * w, e.危险放行率, w * .9,
                      color=col, alpha=[.5, 1][j], label=rf"{name}, $\tau={tau}$")
    ax[2].set_xticks(x); ax[2].set_xticklabels(FR)
    ax[2].set_xlabel("auxiliary labels assumed by auditor")
    ax[2].set_ylabel("dangerous release rate"); ax[2].grid(axis="x", visible=False)
    ax[2].set_ylim(0, .38)
    ax[2].legend(frameon=False, fontsize=5.9, ncol=2, loc="upper center", columnspacing=.8,
                 handlelength=1.1)
    ax[2].set_title("(c) cost of a weak assumption", loc="left")
    fig.tight_layout(); fig.savefig(F / "fig9_attacker_capability.pdf"); fig.savefig(F / "fig9_attacker_capability.png", dpi=170); plt.close(fig)
    print("fig9 done")


if __name__ == "__main__":
    for fn in [fig3, fig4, fig5, fig7, fig8, fig9]:
        try:
            fn()
        except Exception as e:
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
