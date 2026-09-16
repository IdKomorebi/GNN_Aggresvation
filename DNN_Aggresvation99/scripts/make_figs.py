# -*- coding: utf-8 -*-
"""99 号图表：读 outputs/*.csv，不重新计算。"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]; O = ROOT / "outputs"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160,
                     "mathtext.fontset": "dejavusans"})
BLUE, ORANGE, AQUA, YEL, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a3a29c", "#2b2b2a"


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight"); plt.close(fig); print("->", name)


# 图1 概念关系 ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec="#8a8984", lw=1))
    ax.text(x + 0.15, y + h - 0.2, title, fontsize=11, va="top", color=INK, weight="bold")
    ax.text(x + 0.15, y + h - 0.62, body, fontsize=9.2, va="top", color=INK, linespacing=1.55)


def arrow(x1, y1, x2, y2, txt="", dx=0, dy=0.12):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#6b6a66", lw=1.2))
    if txt: ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, txt, fontsize=8.6, color="#52514e", ha="center")


box(0.2, 5.3, 3.6, 1.7, "集合推断能力 v(S)", "攻击者族 F 的最优测试 R²\n单调包络 v̄(S)=max_{T⊆S} v(T)\nv(∅)=0，0≤v≤1", "#eef4fc")
box(4.7, 5.3, 3.9, 1.7, "条件边际 Δ_i(T)", "Δ_i(T)=v(T∪i)−v(T)\n= Σ_{U⊆T} m(U∪i)   (Harsanyi)\n“已知 T 时再公开 i 增加多少”", "#eef4fc")
box(9.4, 5.3, 3.4, 1.7, "Möbius 红利 m(U)", "v(S)=Σ_{U⊆S} m(U)\nm({i})=a_i，m({i,j})=I_ij\n高阶 m 在噪声下大量正负抵消", "#f3f2ee")
box(0.2, 2.35, 3.0, 2.2, "单字段能力 a_i", "a_i=v({i})=M_i^(0)\n\n= 背景为空时的边际\n次模(无协同)时\nM_i^(K)=a_i 对所有 K", "#fdf0ea")
box(3.55, 2.35, 3.2, 2.2, "背景受限最大边际 M_i^(K)", "max_{|T|≤K, i∉T} Δ_i(T)\n\nM^(1)=a_i+max(0, max_j I_ij)\n增强 Γ^(K)=M^(K)−a_i\nK=n−1 即 MCI", "#fdf0ea")
box(7.1, 2.35, 2.75, 2.2, "集合协同 syn(S)", "v(S)−max_{T⊊S} v(T)\n=min_{j∈S} Δ_j(S\\j)\n\n⇒ syn(S) ≤ min_{j∈S}\n   M_j^(|S|−1)", "#fdf0ea")
box(10.2, 2.35, 2.6, 2.2, "Shapley φ_i", "E_π[Δ_i(前驱)]\n= Σ_{U∋i} m(U)/|U|\n\n满足 Σφ=v(N)\n副本/遮挡下被稀释", "#f3f2ee")
box(0.2, 0.15, 5.9, 1.75, "安全语义（M 的操作含义）", "① 余量：M_i^(K)=最小 m，使“∀|T|≤K: v(T)≤τ−m ⇒ v(T∪i)≤τ”对所有 τ 成立\n② 关键性：∃|T|≤K 使 i 越过 τ ⇔ i 属于某个 ≤K+1 的最小不安全集合（v̄ 上）\n③ 预算：|A|≤K+1 ⇒ v(A) ≤ Σ_{i∈A} M_i^(K)，逐阶预算更紧", "#eaf6f0")
box(6.5, 0.15, 6.3, 1.75, "与 Shapley 的公理差异", "M：复制不变、字段全集单调、单字段下界 M≥a_i、不要求守恒\nShapley：效率 Σφ=v(N) 强制分摊 → 副本 1/k 稀释；备用通道被主通道遮挡\n两者序关系：a_i/n ≤ φ_i ≤ M_i^(n−1)（单调时）", "#f3f2ee")
arrow(3.8, 6.15, 4.7, 6.15, "取差分")
arrow(8.6, 6.15, 9.4, 6.15, "Möbius")
arrow(5.6, 5.3, 1.7, 4.55, "T=∅", dx=-0.4)
arrow(6.2, 5.3, 5.15, 4.55, "最坏背景 |T|≤K", dx=0.2)
arrow(7.0, 5.3, 8.45, 4.55, "T=S\\j 取最小", dx=0.3)
arrow(7.9, 5.3, 11.5, 4.55, "按随机顺序平均", dx=0.7)
ax.set_title("图1  v(S) → 条件边际 → 四种字段量：数学关系与安全语义总览", loc="left", fontsize=12.5)
save(fig, "fig1_relation_map.png")

# 图2 饱和 / 协同阶 ---------------------------------------------------------------
P2 = pd.read_csv(O / "P2_saturation.csv"); P2["gain"] = P2.Mtop - P2.a
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))
for g, color in [("gameA", BLUE), ("gameB", ORANGE)]:
    d = P2[(P2.game == g) & (P2.gain > 0.05)]
    ratio = [(d.a / d.Mtop).values, (d.M1 / d.Mtop).values, (d.M2 / d.Mtop).values, (d.M3 / d.Mtop).values]
    med = [np.median(r) for r in ratio]; q = [np.quantile(r, 0.1) for r in ratio]
    ax[0].plot(range(4), med, color=color, lw=2, marker="o", label=f"{g} 中位数（{len(d)} 个有增强的字段×目标）")
    ax[0].plot(range(4), q, color=color, lw=1.2, ls="--", label=f"{g} 10% 分位")
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["K=0\n单字段", "K=1", "K=2", "K=3"]); ax[0].set_ylim(0, 1.05)
ax[0].set_ylabel("M^(K) / M^(n−1)"); ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")
ax[0].set_title("A. 最坏边际随背景阶迅速饱和", loc="left", fontsize=10.5)
w = 0.38; ks = np.arange(0, 9)
for k, (g, color) in enumerate([("gameA", BLUE), ("gameB", ORANGE)]):
    cnt = P2[P2.game == g].kappa_eps.value_counts().reindex(ks, fill_value=0)
    ax[1].bar(ks + (k - 0.5) * w, cnt / cnt.sum(), width=w - 0.04, color=color, label=g)
ax[1].set_xticks(ks); ax[1].set_xlabel("协同阶 κ_i（M^(K) 距 M^(n−1) 不超过 0.01 的最小 K）"); ax[1].set_ylabel("字段×目标占比")
ax[1].legend(frameon=False); ax[1].grid(axis="x", visible=False)
ax[1].set_title("B. K≤3 内达到全局最坏边际（±0.01）：gameA 88%，gameB 97%", loc="left", fontsize=10.5)
fig.suptitle("图2  背景受限的合理性：协同阶小（gameA p=14 / gameB p=12 真值）", x=0.01, ha="left", fontsize=12)
save(fig, "fig2_saturation_kappa.png")

# 图3 复制稀释 ------------------------------------------------------------------
P4 = pd.read_csv(O / "P4_copies.csv")
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=False)
for k, (src, nm) in enumerate([("forecast_load_mw_latest_available", "负荷预测(最新)"), ("gen_fuel_wind_pct", "风电占比")]):
    d = P4[(P4.src == src) & (P4.conf == "metered_load_mw")]
    for col, lab, color, mk in [("M1", "M^(1)（=M^(2)）", ORANGE, "D"), ("single", "单字段 a_i", GRAY, "o"),
                                ("shapley", "该字段 Shapley", BLUE, "s"), ("shapley_copies_sum", "原字段+全部副本 Shapley 之和", AQUA, "^"), ("loo", "全集删除边际", YEL, "v")]:
        ax[k].plot(d.copies, d[col], color=color, marker=mk, lw=2 if col in ("M1", "shapley") else 1.4, ms=6, label=lab)
    ax[k].set_xticks(range(5)); ax[k].set_xlabel("加入的精确副本数"); ax[k].set_ylim(-0.03, 1.05)
    ax[k].set_title(f"{nm} → metered_load_mw", loc="left", fontsize=10.5)
ax[0].set_ylabel("字段分数（R² 口径）"); ax[1].legend(frameon=False, fontsize=8.5, loc="center right")
fig.suptitle("图3  复制稀释：多发几份副本就能把 Shapley 压低，M^(K) 与单字段能力不变（真值博弈上构造）", x=0.01, ha="left", fontsize=12)
save(fig, "fig3_copy_dilution.png")

# 图4 Harsanyi 分解 ---------------------------------------------------------------
P5 = pd.read_csv(O / "P5_harsanyi.csv"); d = P5[P5.conf == "metered_load_mw"].copy()
nm = {"gen_fuel_wind_mw": "风电出力", "gen_fuel_wind_pct": "风电占比", "forecast_load_mw_latest_available": "负荷预测(最新)",
      "forecast_load_mw_day_ahead": "负荷预测(日前)", "gen_fuel_coal_mw": "燃煤出力"}
d["name"] = d.field.map(nm); y = np.arange(len(d))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].barh(y, d.pos_dividend_share, color=AQUA, height=0.55, label="高阶(|U|≥2)正红利分摊 Σ m(U)/|U|")
ax[0].barh(y, d.neg_dividend_share, color=ORANGE, height=0.55, label="高阶(|U|≥2)负红利分摊")
for yi, (_, r) in zip(y, d.iterrows()):
    ax[0].text(16.5, yi, f"单字段 {r.single:.3f} → φ = {r.shapley:.3f}", va="center", fontsize=8.8, color=INK)
ax[0].set_yticks(y); ax[0].set_yticklabels(d.name); ax[0].set_xlim(-17, 26); ax[0].axvline(0, color="#8a8984", lw=0.8)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2); ax[0].grid(axis="y", visible=False)
ax[0].set_title("A. Shapley = 各阶红利分摊之和：±14 量级相互抵消", loc="left", fontsize=10.5)
ax[1].barh(y, d.single, color=GRAY, height=0.55, label="单字段 a_i")
ax[1].barh(y, d.M1 - d.single, left=d.single, color=ORANGE, height=0.55, label="与见证伙伴的二阶交互 max(0, I_ij)")
for yi, (_, r) in zip(y, d.iterrows()):
    partner = {"gen_fuel_wind_pct": "风电占比", "gen_fuel_wind_mw": "风电出力"}.get(str(r.M1_best_partner), "无（空背景最坏）")
    ax[1].text(r.M1 + 0.01, yi, f"M^(1)={r.M1:.3f}  见证：{partner}", va="center", fontsize=8.8, color=INK)
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(0, 1.55)
ax[1].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2); ax[1].grid(axis="y", visible=False)
ax[1].set_title("B. M^(1) = 两个低阶、可单独认证的量之和", loc="left", fontsize=10.5)
fig.suptitle("图4  metered_load_mw 上的 Harsanyi 分解：Shapley 不可归因到具体结构，M^(1) 自带见证", x=0.01, ha="left", fontsize=12)
save(fig, "fig4_harsanyi_decomposition.png")

# 图5 预算规则 ------------------------------------------------------------------
B3 = pd.read_csv(O / "P3b_budget_tight.csv"); B3 = B3[B3.tau == 0.7]
sel = ["Shapley 和", "单字段和 Σa", "加性预算 ΣM^(1)", "逐阶预算 U^(1)", "逐阶预算 U^(2)"]
color = {"Shapley 和": BLUE, "单字段和 Σa": GRAY, "加性预算 ΣM^(1)": ORANGE, "逐阶预算 U^(1)": AQUA, "逐阶预算 U^(2)": YEL}
B9 = pd.read_csv(O / "P3b_budget_tight.csv"); B9 = B9[B9.tau == 0.9]
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
x = np.arange(len(sel)); w = 0.38
for k, g in enumerate(["gameA", "gameB"]):
    d = B9[B9.game == g].set_index("score").loc[sel]
    ax[0].bar(x + (k - 0.5) * w, d["上界违例率"], width=w - 0.04, color=[color[s] for s in sel], alpha=1 if k == 0 else 0.55)
    for xi, vi in zip(x + (k - 0.5) * w, d["上界违例率"]): ax[0].text(xi, vi + 0.02, f"{vi:.1%}" if vi < 0.995 else "99.7%", ha="center", fontsize=7.2, rotation=90, color=INK)
ax[0].set_xticks(x); ax[0].set_xticklabels(sel, rotation=20, ha="right", fontsize=8.8); ax[0].grid(axis="x", visible=False)
ax[0].set_ylabel("v(A) > 分数 的集合占比"); ax[0].set_title("A. 当作集合风险上界：违例率（深=gameA，浅=gameB）", loc="left", fontsize=10)
for k, (col, title) in enumerate([("|A|≤2", "B. |A|≤2（K=1 的保证范围）τ=0.7"), ("全部", "C. 全部集合 τ=0.7")]):
    for g, mk in [("gameA", "o"), ("gameB", "s")]:
        d = B3[B3.game == g].set_index("score").loc[sel]
        for s in sel:
            ax[k + 1].scatter(d.loc[s, f"误拒_{col}"], d.loc[s, f"漏判_{col}"], s=70, color=color[s], marker=mk,
                              edgecolor="#fcfcfb", linewidth=1.5, label=s if g == "gameA" else None, zorder=3)
    ax[k + 1].set_xlabel("误拒率（安全集合被判不安全）"); ax[k + 1].set_ylabel("漏判率（不安全集合被判安全）")
    ax[k + 1].set_xlim(-0.03, 1.0); ax[k + 1].set_ylim(-0.05, 1.05); ax[k + 1].set_title(title, loc="left", fontsize=10)
ax[2].legend(frameon=False, fontsize=8.2, loc="center right"); ax[2].text(0.5, 0.93, "圆=gameA  方=gameB", fontsize=8.5, color="#52514e")
fig.suptitle("图5  字段分数当“风险预算”：M^(K) 给出零漏判保证（代价是误拒），Shapley 和几乎总是低估", x=0.01, ha="left", fontsize=12)
save(fig, "fig5_budget_rules.png")

# 图6 估计器与认证下界 ------------------------------------------------------------
P6 = pd.read_csv(O / "P6_certify.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for k, est in enumerate(["L1x", "L0ensx", "direct"]):
    d = P6[(P6.game == "gameA") & (P6.est == est)]
    ax[k].plot(d.K, d.true_M, color=INK, lw=2, marker="o", label="真值 M^(K)")
    ax[k].plot(d.K, d.est_M, color=ORANGE, lw=2, marker="D", label="估计器直接给出的 M_hat^(K)")
    ax[k].plot(d.K, d.cert_lower, color=AQUA, lw=2, marker="s", label="估计器选出的背景 → 真值认证（下界）")
    ax[k].set_xticks(range(4)); ax[k].set_xlabel("背景阶 K")
    ax[k].set_title({"L1x": "L1 + x,x²", "L0ensx": "L0 集成 + x,x²", "direct": "共享头 oracle"}[est], loc="left", fontsize=10.5)
ax[0].set_ylabel("字段×目标平均（gameA）"); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle("图6  估计 M^(K) 的两种误差：K 越大，最大值选择偏差越大（估计上偏），认证下界离真值越远", x=0.01, ha="left", fontsize=12)
save(fig, "fig6_estimation_certification.png")
