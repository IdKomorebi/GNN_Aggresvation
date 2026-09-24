# -*- coding: utf-8 -*-
"""112 号补充：扫描—认证需要认证多少个候选背景？三种子 vs 单种子估计器。
对每个字段，按估计器给出的边际从大到小取前 k 个背景，用真值边际认证，报告 Σ认证下界 / Σ精确 M^(2)。
同时给出 111 号（RTS-GMLC）作对照。不重训：两号的全部规模 ≤3 真值都已在 V_official.npy 中。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
KS = [1, 2, 3, 5, 10, 20, 30, 50, 100]
rows = []
for exp, tag in [("DNN_Aggresvation112", "NEM 真实市场"), ("DNN_Aggresvation111", "RTS-GMLC")]:
    O = os.path.join(REPO, exp, "outputs"); spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    V = np.load(os.path.join(O, "V_official.npy")); E = np.load(os.path.join(O, "est.npz")); p = len(spec["cand"])
    _, _, Dt, _, bsz = pipe.m_table(V, keys, list(range(p)), 2); sel = np.where(bsz <= 2)[0]
    for nm, est in [("三种子", E["est"]), ("单种子", E["est_1seed"])]:
        _, _, De, _, _ = pipe.m_table(pipe.closure_max(est, keys), keys, list(range(p)), 2)
        for c, y in enumerate(spec["targ"]):
            dt, de = Dt[:, sel, c], De[:, sel, c]; Mt, Me = dt.max(1), de.max(1); order = np.argsort(-de, 1)
            for k in KS:
                L = np.take_along_axis(dt, order[:, :k], 1).max(1)
                rows.append(dict(数据集=tag, 估计器=nm, 目标=y.replace("Y_", ""), k=k, 背景总数=len(sel),
                                 认证下界比=float(L.sum() / Mt.sum()), 见证召回=float((L >= Mt - 0.02).mean()),
                                 M2偏差=float((Me - Mt).mean()), M2误差=float(np.abs(Me - Mt).mean())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "cert_curve.csv"), index=False)

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#e6e5e0"
fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
for ax, tag in zip(axs, ["NEM 真实市场", "RTS-GMLC"]):
    d = R[R.数据集 == tag]
    for (tg, col) in zip(d.目标.unique(), [BLUE, ORANGE, AQUA]):
        for nm, ls, mk in [("三种子", "-", "o"), ("单种子", "--", "s")]:
            e = d[(d.目标 == tg) & (d.估计器 == nm)]
            ax.plot(e.k, e.认证下界比, ls, marker=mk, ms=4, lw=1.8, color=col, label=f"{tg}｜{nm}")
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_ylim(0.3, 1.02)
    ax.axhline(0.95, color="#8a8984", lw=.8, ls=":"); ax.grid(color=GRID); ax.set_axisbelow(True)
    ax.set_xlabel(f"每个字段认证的候选背景数 k（共 {d.背景总数.iloc[0]} 个背景）")
    ax.set_title(f"{'A' if tag.startswith('NEM') else 'B'}. {tag}", loc="left", fontsize=10.2, color=INK)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right", ncol=1)
axs[0].set_ylabel("认证下界 / 精确 M^(2)")
fig.suptitle("图5  扫描—认证：按估计器排序认证前 k 个背景，下界能恢复精确风险的多少（实线三种子、虚线单种子；点线为 95%）",
             x=.01, ha="left", fontsize=11.2)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(ROOT, "figures", "fig5_certification_curve.png"), bbox_inches="tight")
print(R[R.k.isin([1, 3, 10])].pivot_table(index=["数据集", "目标", "估计器"], columns="k", values="认证下界比").round(3).to_string())
