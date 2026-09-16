# -*- coding: utf-8 -*-
"""100 号图表（只读 outputs/analysis 下的结果，不重新计算）。"""
import glob, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, YEL, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#2b2b2a"
DSN = {"pjm": "PJM", "caiso": "CAISO"}


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight"); plt.close(fig); print("->", name)


def short(f):
    rep = [("gen_fuel_", ""), ("forecast_load_mw_", "负荷预测_"), ("rt15_lmp_hourly_mean__", "实时"), ("dam_lmp__", "日前"),
           ("_usd_per_mwh__th_", "_"), ("_gen_apnd", ""), ("dam_renewable_forecast__mw__", "可再生预测_"), ("dam_schedule__mw__", "计划_"),
           ("actual_load__mw__", "实际负荷_"), ("__", "_")]
    for a, b in rep: f = f.replace(a, b)
    return f[:34]


# 图1 升级曲线与饱和 ----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
for ds, color in [("pjm", BLUE), ("caiso", ORANGE)]:
    z = np.load(A / f"{ds}_mci_lower.npz"); M, low = z["M"], z["low"]; ok = low > 0.05
    ax[0].plot(range(4), [M[:, K].mean() for K in range(4)], color=color, lw=2.2, marker="o", label=f"{DSN[ds]} 平均 M^(K)")
    ax[0].axhline(low.mean(), color=color, ls=":", lw=1.3)
    r = np.stack([M[:, K][ok] / low[ok] for K in range(4)])
    ax[1].plot(range(4), np.median(r, 1), color=color, lw=2.2, marker="o", label=f"{DSN[ds]} 中位数")
    ax[1].fill_between(range(4), np.quantile(r, 0.1, 1), np.quantile(r, 0.9, 1), color=color, alpha=0.15, label=f"{DSN[ds]} 10–90% 分位")
ax[0].text(3.05, 0.285, "点线：已认证 MCI 下界均值", fontsize=8.5, color="#52514e", ha="right", va="bottom")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["K=0\n单字段", "K=1", "K=2", "K=3"]); ax[0].set_ylabel("字段×目标平均（R² 口径）")
ax[0].legend(frameon=False, loc="lower right"); ax[0].set_title("A. 推断升级曲线（41 字段 × 12 目标，专用重训真值）", loc="left", fontsize=10.5)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["K=0", "K=1", "K=2", "K=3"]); ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("M^(K) / 已认证 MCI 下界"); ax[1].legend(frameon=False, loc="lower right", fontsize=8.5)
ax[1].set_title("B. 饱和比：K=2 中位 PJM 0.93 / CAISO 0.97；K=3 基本饱和", loc="left", fontsize=10.5)
fig.suptitle("图1  完整字段集上的 K 敏感性（M^(K) 精确计算；MCI 由通用模型搜索大背景 + 专用重训认证给出下界）", x=0.01, ha="left", fontsize=12)
save(fig, "fig1_escalation_saturation.png")

# 图2 典型升级曲线 -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
cases = {"pjm": [("total_gen", "gen_fuel_wind_mw"), ("total_gen", "gen_fuel_solar_mw"), ("metered_load_mw", "gen_fuel_multiple_fuels_pct"),
                 ("total_lmp_da", "system_energy_price_da"), ("congestion_price_da", "gen_fuel_oil_mw")],
         "caiso": [("rt15_lmp_hourly_mean__congestion_usd_per_mwh__th_sp15_gen_apnd", "rt15_lmp_hourly_mean__lmp_usd_per_mwh__th_np15_gen_apnd"),
                   ("actual_load__mw__ca_iso_tac", "dam_renewable_forecast__mw__sp15__solar"),
                   ("dam_schedule__mw__generation__caiso_totals", "dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd"),
                   ("dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd", "dam_lmp__lmp_usd_per_mwh__th_zp26_gen_apnd"),
                   ("dam_as_total_procured__mw__sr__as_caiso", "actual_load__mw__pace")]}
cols = [BLUE, ORANGE, AQUA, YEL, GRAY]
for k, ds in enumerate(["pjm", "caiso"]):
    P = pd.read_csv(A / f"{ds}_profiles.csv"); used = []
    for (c, f), color in zip(cases[ds], cols):
        r = P[(P.conf == c) & (P.field == f)]
        if r.empty: continue
        r = r.iloc[0]; y = [r.M0, r.M1, r.M2, r.M3]
        ax[k].plot(range(4), y, color=color, lw=2, marker="o", label=f"{short(f)} → {short(c)}")
        ty = y[-1]
        while any(abs(ty - u) < 0.04 for u in used): ty -= 0.04
        used.append(ty)
        ax[k].text(3.08, ty, f"{y[0]:.2f}→{y[1]:.2f}→{y[2]:.2f}→{y[3]:.2f}", fontsize=7.8, va="center", color=INK)
    ax[k].set_xticks(range(4)); ax[k].set_xticklabels(["K=0", "K=1", "K=2", "K=3"]); ax[k].set_xlim(-0.2, 4.3)
    ax[k].legend(frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1)
    ax[k].set_title(f"{DSN[ds]}", loc="left", fontsize=10.5)
ax[0].set_ylabel("M^(K)")
fig.suptitle("图2  典型推断升级曲线：单字段弱、一个辅助字段即跃升（出力+占比、节点价差），与单字段即强的字段对照", x=0.01, ha="left", fontsize=12)
save(fig, "fig2_profile_cases.png")

# 图3 攻击器鲁棒性 -----------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
for k, ds in enumerate(["pjm", "caiso"]):
    Md, Ma = np.load(A / f"{ds}_M_dnn.npz")["M"], np.load(A / f"{ds}_M_attack.npz")["M"]
    for K, color in [(0, GRAY), (1, BLUE), (2, ORANGE)]:
        ax[k].scatter(Md[:, K].ravel(), Ma[:, K].ravel(), s=9, alpha=0.55, color=color, label=f"K={K}", edgecolor="none")
    ax[k].plot([0, 1], [0, 1], color="#8a8984", lw=1, ls="--"); ax[k].set_xlim(0, 1); ax[k].set_ylim(0, 1)
    ax[k].set_xlabel("M^(K)：仅 DNN 攻击器"); ax[k].set_title(DSN[ds], loc="left", fontsize=10.5)
    ax[k].legend(frameon=False, loc="lower right", markerscale=2)
ax[0].set_ylabel("M^(K)：DNN ∪ 梯度提升树（val 选择）")
fig.suptitle("图3  攻击器鲁棒性：树模型只让 M 小幅上移（排序 Kendall PJM 0.82–0.88 / CAISO 0.91–0.94），但 PJM 阈值跨越数显著增加", x=0.01, ha="left", fontsize=11.5)
save(fig, "fig3_attacker_robustness.png")

# 图4 边际误差与见证 ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
for ds, mk in [("pjm", "o"), ("caiso", "s")]:
    E3 = pd.read_csv(sorted(glob.glob(str(A / f"{ds}_E3_*.csv")))[0])
    for est, color in [("L0ensx", ORANGE), ("L0", BLUE), ("L1x", AQUA)]:
        d = E3[E3.est == est]
        if d.empty: continue
        ax[0].plot(d.K, d.估计偏差, color=color, marker=mk, lw=1.8, label=f"{DSN[ds]} {est}")
        ax[1].plot(d.K, d["L除以M"], color=color, marker=mk, lw=1.8)
        ax[2].plot(d.K, d["召回_L大于M减002"], color=color, marker=mk, lw=1.8)
for a_, t in zip(ax, ["A. 估计 M_hat − 真 M（max 选择偏差）", "B. 认证下界 L=Δ(T_hat) / 真 M", "C. 见证召回（L ≥ M − 0.02）"]):
    a_.set_xticks(range(4)); a_.set_xlabel("背景预算 K"); a_.set_title(t, loc="left", fontsize=10.5)
ax[0].axhline(0, color="#8a8984", lw=0.8); ax[0].legend(frameon=False, fontsize=8, ncol=1)
fig.suptitle("图4  通用模型找最坏背景：K≤1 可靠；K 增大时召回下降、高维集成特征的上偏更明显（同表认证，独立认证见 101 号）", x=0.01, ha="left", fontsize=11.5)
save(fig, "fig4_witness_quality.png")

# 图5 上界覆盖 --------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ds, mk in [("pjm", "o"), ("caiso", "s")]:
    E4 = pd.read_csv(sorted(glob.glob(str(A / f"{ds}_E4_*.csv")))[0])
    for (est, alpha), color, ls in [(("L0ensx", 0.05), ORANGE, "-"), (("L0ensx", 0.01), ORANGE, "--"), (("L0", 0.05), BLUE, "-"), (("L0", 0.01), BLUE, "--")]:
        d = E4[(E4.est == est) & (E4.alpha == alpha)]
        ax[0].plot(d.K, d.分层上界覆盖率, color=color, ls=ls, marker=mk, lw=1.8, label=f"{DSN[ds]} {est} α={alpha}")
        ax[1].plot(d.K, d.分层上界紧度, color=color, ls=ls, marker=mk, lw=1.8)
ax[0].axhline(0.95, color=GRAY, lw=1); ax[0].axhline(0.99, color=GRAY, lw=1, ls="--")
ax[0].set_xticks([1, 2, 3]); ax[1].set_xticks([1, 2, 3]); ax[0].set_ylim(0.8, 1.005)
ax[0].set_title("A. 上界覆盖率 P(M ≤ U)（灰线为名义 95% / 99%）", loc="left", fontsize=10.5); ax[1].set_title("B. 紧度 U − M", loc="left", fontsize=10.5)
ax[0].legend(frameon=False, fontsize=7.8, loc="lower left"); ax[0].set_xlabel("K"); ax[1].set_xlabel("K")
fig.suptitle("图5  U = max_T[Δ_hat(T)+q(层)]（按 Δ_hat 分 10 层、留一字段交叉校准）：K≤2 接近名义覆盖，K=3 覆盖不足", x=0.01, ha="left", fontsize=11.5)
save(fig, "fig5_upper_bound.png")

# 图6 基底场景 --------------------------------------------------------------------
meta = pickle.load(open(ROOT / "outputs/sets/pjm_meta.pkl", "rb")); conf = meta["conf"]
zb = np.load(A / "pjm_M_base.npz"); Mb, VB, actb = zb["M"], zb["VB"], list(zb["active"])
Md = np.load(A / "pjm_M_dnn.npz"); M0, act = Md["M"], list(Md["active"]); pb = [act.index(a) for a in actb]
fig, ax = plt.subplots(figsize=(12, 4.3)); x = np.arange(12); w = 0.28
ax.bar(x - w, VB, w - 0.03, color=GRAY, label="V(B)：基底本身的推断能力")
ax.bar(x, M0[pb, 2].max(0), w - 0.03, color=BLUE, label="max_i M_i^(2)，B=∅")
ax.bar(x + w, Mb[:, 2].max(0), w - 0.03, color=ORANGE, label="max_i M_i^(2)，给定 B")
ax.axhline(0.7, color="#8a8984", ls="--", lw=1); ax.text(11.6, 0.71, "τ=0.7", fontsize=8.5, ha="right")
for c in range(12):
    if VB[c] > 0.7: ax.text(c - w, VB[c] + 0.02, "基底泄露", fontsize=8, ha="center", color=INK)
ax.set_xticks(x); ax.set_xticklabels([t.replace("da_as_total_mw_", "备用_").replace("_reserve", "")[:22] for t in conf], rotation=25, ha="right", fontsize=8.5)
ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.12)
ax.set_title("图6  PJM 公开基底 B = {最新负荷预测, 日前负荷预测}：两个目标被基底直接泄露，其余目标的字段风险与排序明显改变", loc="left", fontsize=11.5)
save(fig, "fig6_public_base.png")

# 图7 效率 ------------------------------------------------------------------------
from math import comb
ef = json.load(open(A / "efficiency.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for ds, mk in [("pjm", "o"), ("caiso", "s")]:
    o = ef[ds]; Q = np.logspace(0, 6, 40)
    small = {int(k): v for k, v in o["parallel_small"].items()}
    bx = np.log([1, 64, 256, 1024]); by = np.log([small[1], small[64], small[256], o["parallel_B1024"]])
    pu = np.exp(np.interp(np.log(np.minimum(Q, 1024)), bx, by))   # 批大小 = min(Q,1024)，实测点之间对数插值
    for y, lab, color in [(Q * o["sequential"], "逐个专用重训", GRAY), (Q * pu, "GPU 并行专用重训", BLUE),
                          (o["uniform_train_sec_per_seed"] + Q * o["est"]["L0"], "通用模型 L0（含训练）", AQUA),
                          (3 * o["uniform_train_sec_per_seed"] + Q * o["est"]["L0ensx"], "通用模型 L0ensx（含训练）", ORANGE)]:
        ax[0].plot(Q, y / 60, color=color, lw=2 if ds == "pjm" else 1.2, ls="-" if ds == "pjm" else "--", label=f"{lab}" if ds == "pjm" else None)
for q, t in [(861, "完整K=1"), (11521, "完整K=2"), (112791, "完整K=3")]:
    ax[0].axvline(q, color="#d6d5cf", lw=1); ax[0].text(q, 3e5, t, fontsize=8.5, ha="center", color=INK)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("集合价值查询次数"); ax[0].set_ylabel("单 GPU 墙钟（分钟）")
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left"); ax[0].set_title("A. 查询规模 vs 墙钟（实线 PJM，虚线 CAISO；实测单价）", loc="left", fontsize=10.5)
o = ef["pjm"]; ps = np.arange(20, 301, 10); nK2 = np.array([sum(comb(p, k) for k in range(1, 4)) for p in ps])
ax[1].plot(ps, nK2 * o["parallel_B1024"] / 3600, color=BLUE, lw=2, label="GPU 并行专用重训")
ax[1].plot(ps, (o["uniform_train_sec_per_seed"] + nK2 * o["est"]["L0"]) / 3600, color=AQUA, lw=2, label="通用模型 L0")
ax[1].plot(ps, (3 * o["uniform_train_sec_per_seed"] + nK2 * o["est"]["L0ensx"]) / 3600, color=ORANGE, lw=2, label="通用模型 L0ensx")
ax[1].axvline(41, color="#d6d5cf", lw=1); ax[1].text(43, ax[1].get_ylim()[1] if False else 20, "本文 41 字段", fontsize=8.5)
ax[1].set_yscale("log"); ax[1].set_xlabel("候选字段数 p"); ax[1].set_ylabel("完整 K=2 所需小时（单 GPU，外推）")
ax[1].legend(frameon=False, fontsize=8.5); ax[1].set_title("B. 完整 K=2 随字段数外推（按 PJM 单价，未计入估计器随 p 变慢）", loc="left", fontsize=10.5)
fig.suptitle(f"图7  效率对最强基线：41 字段完整 K=2 并行重训约 11 分钟即可；L0 盈亏平衡约 {ef['pjm']['盈亏平衡查询数_L0_相对并行重训']:.0f} 次查询，L0ensx 约 {ef['pjm']['盈亏平衡查询数_L0ensx_相对并行重训']:.0f} 次",
             x=0.01, ha="left", fontsize=11.5)
save(fig, "fig7_efficiency.png")

# 图8 CAISO 副本/备用通道对照 --------------------------------------------------------------
G = pd.read_csv(A / "caiso_gameC_fields.csv"); G["SAGE"] = G[["sage_s0", "sage_s1", "sage_s2"]].mean(1)
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=False)
for k, (tgt, title) in enumerate([("dam_lmp__lmp_usd_per_mwh__th_sp15_gen_apnd", "目标：SP15 日前 LMP（节点电价互为替代通道）"),
                                  ("actual_load__mw__ca_iso_tac", "目标：CA ISO 实际负荷（分区负荷预测/计划近重复）")]):
    d = G[G.conf == tgt].sort_values("M1"); y = np.arange(len(d))
    for col, lab, color, mk in [("a", "单字段 a", GRAY, "o"), ("shapley", "总体 Shapley", BLUE, "s"), ("SAGE", "全量模型 SAGE（3 种子均值）", AQUA, "^"),
                                ("loo", "全集删除边际", YEL, "v"), ("M1", "M^(1)", ORANGE, "D")]:
        if col == "M1":
            ax[k].scatter(d[col], y, s=70, facecolors="none", edgecolors=color, marker=mk, label=lab, zorder=2, linewidth=1.8)
        else:
            ax[k].scatter(d[col], y, s=42, color=color, marker=mk, label=lab, zorder=3, edgecolor="#fcfcfb", linewidth=1.2)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax[k].plot([min(r.a, r.shapley, r.SAGE, r.loo, r.M1), max(r.a, r.shapley, r.SAGE, r.loo, r.M1)], [yi, yi], color="#e0dfda", lw=1, zorder=1)
    ax[k].set_yticks(y); ax[k].set_yticklabels(d.field, fontsize=8.5); ax[k].grid(axis="y", visible=False); ax[k].set_xlim(-0.1, 1.02)
    ax[k].set_title(title, loc="left", fontsize=10.5); ax[k].set_xlabel("R² 口径")
ax[1].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.suptitle("图8  CAISO 对照博弈（14 字段全枚举）：强字段 Shapley 仅为单字段能力的 12–23%，删除边际≈0；全量模型 SAGE 集中到被模型选中的一两个替代字段", x=0.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "fig8_caiso_redundancy_control.png")
