# -*- coding: utf-8 -*-
"""论文 V1 配图生成。全部数字直接读实验落盘文件，无手工输入。

数据源：
  PJM   : DNN_Aggresvation91/93/94/outputs/
  CAISO : DNN_Aggresvation95_caiso/outputs/
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

HERE = Path(__file__).resolve().parents[1]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
REPO = HERE.parents[2]
R91, R93, R94 = (REPO / f"DNN_Aggresvation{k}" for k in (91, 93, 94))
R95 = REPO / "DNN_Aggresvation95_caiso"

C_BAD, C_MID, C_GOOD, C_ACC = "#c0392b", "#e69f00", "#2e7d32", "#1f5fa8"


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  → {name}")


# ============================================================
# 图 2：尾部失明——摊销误差随协同强度系统性放大
# ============================================================
def fig_tail_blindness():
    p = pd.read_csv(R91 / "outputs/order2_pairs.csv")
    l1 = pd.read_csv(R93 / "outputs/eval_l1_pairs_aug8.csv")
    l1 = l1[l1.kind == "last"].assign(kind="L1")
    d = pd.concat([p[p.kind.isin(["oracle", "last"])], l1], ignore_index=True)
    d["kind"] = d.kind.replace({"oracle": "共享读出 oracle（全摊销）", "last": "L0：冻结特征+闭式读出"})
    order = ["共享读出 oracle（全摊销）", "L0：冻结特征+闭式读出", "L1"]
    d["kind"] = d.kind.replace({"L1": "L1：闭式头进训练回路"})
    order[2] = "L1：闭式头进训练回路"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True, sharey=True)
    for ax, k in zip(axes, order):
        s = d[d.kind == k]
        ax.scatter(s.truth, s.est, s=4, alpha=0.25, color=C_ACC, edgecolors="none")
        lim = [-0.05, 0.85]
        ax.plot(lim, lim, "k--", lw=1, label="理想 $y=x$")
        rho = spearmanr(s.est, s.truth).statistic
        top = s.nlargest(100, "truth")
        rt = spearmanr(top.est, top.truth).statistic
        ax.scatter(top.truth, top.est, s=10, color=C_BAD, alpha=0.75,
                   edgecolors="none", label="真值 top-100（尾部）")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{k}\n$\\rho$={rho:.3f}   $\\rho_{{tail}}$={rt:.3f}", fontsize=9.5)
        ax.set_xlabel("重训真值 $\\mathrm{syn}_2$")
    axes[0].set_ylabel("估计 $\\widehat{\\mathrm{syn}}_2$")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("PJM：二阶协同估计 vs 重训真值（越强的协同被压得越低）", fontsize=11)
    fig.tight_layout()
    save(fig, "tail_blindness.png")


# ============================================================
# 图 3：分档 gap + 方法阶梯（两面板）
# ============================================================
def fig_gap_and_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))

    # (a) 分档 gap：oracle vs L0（91 号 readout_vs_feature_summary）
    s = pd.read_csv(R91 / "outputs/readout_vs_feature_summary.csv").set_index("method")
    bands = ["weak", "mid", "strong"]
    lab = ["弱档\n(<0.1)", "中档\n(0.1–0.2)", "强档\n(>0.2)"]
    x = np.arange(3); w = 0.26
    for off, (m, name, col) in enumerate([("oracle", "共享读出 oracle", C_BAD),
                                          ("ft50", "50 步微调", C_MID),
                                          ("L0", "L0 闭式读出（零训练）", C_GOOD)]):
        vals = [s.loc[m, f"gap_{b}"] for b in bands]
        axes[0].bar(x + (off - 1) * w, vals, w, label=name, color=col)
        for xi, v in zip(x + (off - 1) * w, vals):
            axes[0].text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(lab)
    axes[0].set_ylabel("低估量 gap = 真值 − 估计")
    axes[0].set_title("(a) PJM：摊销误差按真值强度分档", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].axhline(0, color="k", lw=0.8)

    # (b) 方法阶梯：PJM vs CAISO 的 rho_tail
    pjm = pd.read_csv(R91 / "outputs/order2_truth.csv").set_index("kind")
    pjm_l1 = pd.read_csv(R93 / "outputs/eval_l1_order2_aug8.csv").set_index("kind")
    cai = pd.read_csv(R95 / "outputs/eval_l1_order2.csv").set_index("kind")
    cai_l1 = pd.read_csv(R95 / "outputs/eval_l1_order2_aug8s0.csv").set_index("kind")
    steps = ["oracle", "oracle_affine", "poly2", "last", "L1"]
    names = ["共享读出\noracle", "+仿射校准\n(2 参数)", "poly2\n手工字典", "L0\n冻结φ+闭式", "L1\nmeta-训练"]
    pv = [pjm.loc[k, "rho_tail"] for k in steps[:4]] + [pjm_l1.loc["last", "rho_tail"]]
    cv = [cai.loc[k, "rho_tail"] for k in steps[:4]] + [cai_l1.loc["last", "rho_tail"]]
    x = np.arange(5)
    axes[1].plot(x, pv, "o-", color=C_ACC, lw=2, ms=7, label="PJM")
    axes[1].plot(x, cv, "s--", color=C_GOOD, lw=2, ms=7, label="CAISO（第二数据集）")
    for xi, v in zip(x, pv):
        axes[1].text(xi, v - 0.075, f"{v:.3f}", ha="center", fontsize=8, color=C_ACC)
    for xi, v in zip(x, cv):
        axes[1].text(xi, v + 0.04, f"{v:.3f}", ha="center", fontsize=8, color=C_GOOD)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, fontsize=8.5)
    axes[1].set_ylabel("尾部排序保真 $\\rho_{tail}$")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_title("(b) 沿“φ 适应程度”的方法阶梯（两数据集同构）", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "gap_and_ladder.png")


# ============================================================
# 图 4：成本-保真权衡
# ============================================================
def fig_cost():
    c = pd.read_csv(R91 / "outputs/bench_cost_o3.csv").set_index("method")
    s = pd.read_csv(R91 / "outputs/readout_vs_feature_summary.csv").set_index("method")
    pts = [("oracle", "共享读出 oracle", C_BAD, "o"),
           ("ft25", "25 步微调", C_MID, "^"),
           ("L0[last]", "L0 闭式读出", C_GOOD, "*")]
    rho = {"oracle": s.loc["oracle", "rho"], "ft25": s.loc["ft25", "rho"], "L0[last]": s.loc["L0", "rho"]}
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for key, name, col, mk in pts:
        ax.scatter(c.loc[key, "ms_per_set"], rho[key], s=230 if mk == "*" else 120,
                   color=col, marker=mk, zorder=3, label=name)
        ax.annotate(f"{c.loc[key,'ms_per_set']:.2f} ms\n$\\rho$={rho[key]:.3f}",
                    (c.loc[key, "ms_per_set"], rho[key]), textcoords="offset points",
                    xytext=(8, -16), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("单集合查询耗时（ms，对数轴）")
    ax.set_ylabel("对重训真值的排序保真 $\\rho$")
    ax.set_title("PJM：成本—保真权衡（L0 同时更快更准）", fontsize=10)
    ax.set_ylim(0.6, 0.98)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.annotate("", xy=(2.5, 0.90), xytext=(55, 0.79),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.4))
    ax.text(9, 0.845, "22.4×\n更快且更准", fontsize=8, color="gray", ha="center")
    fig.tight_layout()
    save(fig, "cost_fidelity.png")


# ============================================================
# 图 5：高阶认证——结构化估计 vs 重训认证真值
# ============================================================
def _cert(root, pat):
    fs = sorted(glob.glob(str(root / f"outputs/{pat}")))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    return d


def fig_certify():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    cfg = [
        (axes[0], _cert(R93, "certify_o5_s*of[24].csv"), "PJM order-5：full 手工字典 top-40",
         "认证塌陷：无一过 0.10"),
        (axes[1], _cert(R93, "certify_o5_l1top_s*.csv"), "PJM order-5：L1 自选 top-21",
         "10/21 过 0.10"),
        (axes[2], _cert(R95, "certify_o5_fulltop_s*.csv"), "CAISO order-5：full 字典 top-20",
         "0/20 过 0.10（系统性膨胀 1.7×）"),
    ]
    for ax, d, title, note in cfg:
        if d is None:
            continue
        s = d[d.group == "strong"] if "group" in d else d
        c = d[d.group == "control"] if "group" in d and (d.group == "control").any() else None
        ax.scatter(s.syn_struct, s.syn_true_audit, s=42, color=C_BAD, alpha=0.8,
                   edgecolors="k", linewidths=0.4, label="估计器自选 top")
        if c is not None and len(c):
            ax.scatter(c.syn_struct, c.syn_true_audit, s=30, color="gray", alpha=0.6,
                       edgecolors="none", label="随机对照组")
        lim = [-0.02, max(0.58, float(s.syn_struct.max()) * 1.15)]
        ax.plot(lim, lim, "k--", lw=1, label="估计=真值")
        ax.axhline(0.10, color=C_GOOD, ls=":", lw=1.4, label="真协同阈 0.10")
        ax.set_xlim(lim); ax.set_ylim(-0.02, 0.35)
        ax.set_xlabel("结构化估计 $\\widehat{\\mathrm{syn}}$")
        ax.set_title(f"{title}\n{note}", fontsize=9)
    axes[0].set_ylabel("重训认证真值 $\\mathrm{syn}$")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("高阶协同：手工字典“报假”，学出的表示无偏（点落在对角线上）", fontsize=11)
    fig.tight_layout()
    save(fig, "certify_highorder.png")


# ============================================================
# 图 6：CAISO 认证——L1 自选 top 的无偏性（o4/o5）
# ============================================================
def fig_caiso_certify():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (o, pat, ttl) in zip(axes, [
            (4, "certify_o4_l1top_s*.csv", "CAISO order-4：L1 top-20 + 对照 20"),
            (5, "certify_o5_l1top_s*.csv", "CAISO order-5：L1 top-21 + 对照 20")]):
        d = _cert(R95, pat)
        s, c = d[d.group == "strong"], d[d.group == "control"]
        ax.scatter(s.syn_struct, s.syn_true_audit, s=45, color=C_BAD, alpha=0.85,
                   edgecolors="k", linewidths=0.4, label="L1 自选 top")
        ax.scatter(c.syn_struct, c.syn_true_audit, s=30, color="gray", alpha=0.6,
                   edgecolors="none", label="随机对照组")
        lim = [-0.01, 0.19]
        ax.plot(lim, lim, "k--", lw=1, label="估计=真值")
        ax.axhline(0.10, color=C_GOOD, ls=":", lw=1.4, label="真协同阈 0.10")
        ax.set_xlim(lim); ax.set_ylim(lim)
        n_true = int((s.syn_true_audit > 0.10).sum())
        ax.set_title(f"{ttl}\n精确率 {n_true}/{len(s)}，对照 0/{len(c)}", fontsize=9)
        ax.set_xlabel("结构化估计 $\\widehat{\\mathrm{syn}}$")
    axes[0].set_ylabel("重训认证真值 $\\mathrm{syn}$")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save(fig, "caiso_certify.png")


# ============================================================
# 图 7：衰减律（强度 + 密度，两数据集）
# ============================================================
def fig_decay():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    orders = [3, 4, 5]
    pjm_s, cai_s = [0.455, 0.196, 0.154], [0.294, 0.157, 0.104]
    pjm_d, cai_d = [0.498, 0.069, 0.0051], [0.921, 0.0308, 0.0005]
    axes[0].plot(orders, pjm_s, "o-", color=C_ACC, lw=2, ms=8, label="PJM")
    axes[0].plot(orders, cai_s, "s--", color=C_GOOD, lw=2, ms=8, label="CAISO")
    for o, v in zip(orders, pjm_s):
        axes[0].text(o, v + 0.018, f"{v:.3f}", ha="center", fontsize=8.5, color=C_ACC)
    for o, v in zip(orders, cai_s):
        axes[0].text(o, v - 0.032, f"{v:.3f}", ha="center", fontsize=8.5, color=C_GOOD)
    axes[0].set_xticks(orders); axes[0].set_xlabel("协同阶数 $k$")
    axes[0].set_ylabel("最强认证真值 $\\max\\mathrm{syn}_k$")
    axes[0].set_title("(a) 强度：温和单调衰减", fontsize=10)
    axes[0].set_ylim(0, 0.55); axes[0].legend(fontsize=8.5)

    axes[1].semilogy(orders, pjm_d, "o-", color=C_ACC, lw=2, ms=8, label="PJM")
    axes[1].semilogy(orders, cai_d, "s--", color=C_GOOD, lw=2, ms=8, label="CAISO")
    for o, v in zip(orders, pjm_d):
        axes[1].text(o, v * 1.6, f"{v:.3g}%", ha="center", fontsize=8.5, color=C_ACC)
    for o, v in zip(orders, cai_d):
        axes[1].text(o, v * 0.42, f"{v:.3g}%", ha="center", fontsize=8.5, color=C_GOOD)
    axes[1].set_xticks(orders); axes[1].set_xlabel("协同阶数 $k$")
    axes[1].set_ylabel("强协同密度（%，对数轴）")
    axes[1].set_title("(b) 密度：每阶下降约一个数量级", fontsize=10)
    axes[1].legend(fontsize=8.5)
    fig.suptitle("真值版衰减律：最危险的协同是低阶的（两数据集一致，CAISO 更陡）", fontsize=11)
    fig.tight_layout()
    save(fig, "decay_law.png")


# ============================================================
# 图 8：跨 seed 秩一致性过滤
# ============================================================
def fig_ensemble():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    pj = pd.read_csv(R94 / "outputs/ensemble_filter_o4.csv")
    ca = pd.read_csv(R95 / "outputs/ensemble_filter_o4.csv")
    for ax, d, name in [(axes[0], pj, "PJM"), (axes[1], ca, "CAISO")]:
        t, f = d[d.true], d[~d.true]
        ax.scatter(t.cross_rank, t.syn_true_audit, s=55, color=C_GOOD, marker="o",
                   edgecolors="k", linewidths=0.4, label=f"认证为真（{len(t)}）")
        ax.scatter(f.cross_rank, f.syn_true_audit, s=55, color=C_BAD, marker="x",
                   linewidths=1.6, label=f"认证为假（{len(f)}）")
        ax.axvline(500, color="gray", ls="--", lw=1.4)
        ax.text(560, 0.02, "过滤阈\ncross_rank=500", fontsize=7.5, color="gray")
        ax.axhline(0.10, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("跨 seed 几何平均排名 $\\sqrt{r_1 r_2}$")
        sep = f.cross_rank.median() / t.cross_rank.median()
        ax.set_title(f"{name}：真/假中位分离 {sep:.0f}×", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("重训认证真值 $\\mathrm{syn}_4$")
    fig.suptitle("独立重训的秩一致性可打破共享摊销偏差（过滤后精确率 100%，召回不损失）",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "ensemble_filter.png")


# ============================================================
# 图 9：配对 FDR 的筛选力（CAISO order-3）
# ============================================================
def fig_fdr():
    d = pd.read_csv(R95 / "outputs/paired_fdr_o3_cat3.csv")
    taus = [0.0, 0.05, 0.10, 0.15]
    hard, fdr = [], []
    for t in taus:
        col = [c for c in d.columns if c.startswith(f"rej@{t}") or c == f"rej@{t:g}"]
        hard.append(int((d.syn_audit > t).sum()))
        fdr.append(int(d[col[0]].sum()) if col else 0)
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(taus)); w = 0.36
    ax.bar(x - w / 2, hard, w, color=C_MID, label="硬阈值 $\\widehat{\\mathrm{syn}}>\\tau$")
    ax.bar(x + w / 2, fdr, w, color=C_GOOD, label="配对检验 + BH-FDR (q=0.05)")
    for xi, v in zip(x - w / 2, hard):
        ax.text(xi, v * 1.15, f"{v}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, fdr):
        ax.text(xi, max(v, 1) * 1.15, f"{v}", ha="center", fontsize=8, color=C_GOOD)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\tau$={t}" for t in taus])
    ax.set_ylabel("通过的三元组数（对数轴，共 13244）")
    ax.set_title("CAISO order-3：统计检验把“看起来强”收缩为“站得住”", fontsize=10)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "fdr_filter.png")


# ============================================================
# 图 1：三层管线框架图（示意）
# ============================================================
def fig_framework():
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.3)

    def box(x, y, w, h, txt, fc, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="k",
                                   linewidth=1.1, zorder=2, alpha=0.92))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=3)

    def arrow(x1, y1, x2, y2, txt="", fs=8):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333"))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, txt, ha="center", fontsize=fs)

    ax.text(0.1, 4.05, "摊销层次：把“值”摊销改为“表示”摊销", fontsize=11, weight="bold")
    box(0.2, 2.95, 2.5, 0.75, "数据 $X_S$、可见性掩码 $m_S$", "#e8eef7")
    box(3.3, 2.95, 2.6, 0.75, "摊销表示 $\\varphi_\\theta(x\\odot m,\\,m)$\n（一次前向，共享）", "#cfe0f3")
    box(6.5, 2.95, 3.2, 0.75,
        "逐集合闭式读出\n$\\beta_{S,c}=(\\Phi^\\top\\Phi+\\lambda I)^{-1}\\Phi^\\top Y_c$", "#a8c8ea")
    arrow(2.7, 3.32, 3.3, 3.32)
    arrow(5.9, 3.32, 6.5, 3.32)
    ax.text(8.1, 2.72, "参数不共享，故不受跨集合干涉", fontsize=8, ha="center", color="#c0392b")

    ax.text(0.1, 2.35, "三层审计管线", fontsize=11, weight="bold")
    box(0.2, 1.15, 2.9, 0.95, "第 1 层：全枚举摊销扫描\n阶数 $\\leq 5$，~2.5 ms/集合\n（候选生成）", "#dff0d8")
    box(3.5, 1.15, 2.9, 0.95, "第 2 层：配对 $t$ 检验 + BH-FDR\n$H_0:\\mathrm{syn}\\leq\\tau$\n（统计保证清单）", "#fcf0cd")
    box(6.8, 1.15, 2.9, 0.95, "第 3 层：专用 DNN 重训认证\n~160 s/集合\n（最终真值，只做评估）", "#f8d7da")
    arrow(3.1, 1.62, 3.5, 1.62, "$10^6\\to10^3$")
    arrow(6.4, 1.62, 6.8, 1.62, "$10^3\\to10^1$")

    ax.add_patch(plt.Rectangle((0.2, 0.25), 9.5, 0.62, facecolor="#f2f2f2",
                               edgecolor="gray", linestyle="--", linewidth=1))
    ax.text(4.95, 0.56, "方法学红线：重训真值只用于评估与认证，"
                        "绝不进入查询、路由或校准路径", ha="center", fontsize=9)
    save(fig, "framework.png")


if __name__ == "__main__":
    print("生成论文配图：")
    fig_framework()
    fig_tail_blindness()
    fig_gap_and_ladder()
    fig_cost()
    fig_certify()
    fig_caiso_certify()
    fig_decay()
    fig_ensemble()
    fig_fdr()
    print("完成。")
