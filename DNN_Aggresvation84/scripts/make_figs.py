# -*- coding: utf-8 -*-
"""84 号图：杠杆A变体强档召回对比 + 轴2级联前沿 + σ分档诊断。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))

# ---- (a) 杠杆A：各变体 K=0 强档召回 + 负S1数 ----
ax = axes[0]
sv = pd.read_csv(ROOT / "outputs/variant_scores.csv")
sv = sv[sv.variant != "none(sanity)"]
x = np.arange(len(sv))
rec = sv["recS1_0.2-1.01"].values
neg = sv["S1<0_strong"].values
b = ax.bar(x, rec, color="#2980b9", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(sv.variant, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("最强档(>0.20) K=0 S1 Top-30%召回", color="#2980b9")
ax.set_title("(a) 杠杆A：单调/增量约束能否修 K=0 强档？")
for i, (r, ng) in enumerate(zip(rec, neg)):
    ax.annotate(f"{r:.2f}\n负S1={ng}", (i, r + 0.005), ha="center", fontsize=8)
ax.grid(alpha=0.3, axis="y")

# ---- (b) 轴2：级联成本-召回前沿(强档) ----
ax = axes[1]
fr = pd.read_csv(ROOT / "outputs/cascade_frontier.csv")
colors = {"sigma": "#c0392b", "boundary": "#7f8c8d", "s1s2gap": "#27ae60", "random": "#95a5a6"}
for sig in ["s1s2gap", "sigma", "boundary", "random"]:
    s = fr[fr.signal == sig].sort_values("cost_steps")
    ax.plot(s.cost_steps, s.rec_strong, "o-", color=colors[sig], lw=2, label=sig)
full = fr[fr.signal == "all_k25"].iloc[0]
ax.axhline(full.rec_strong, color="k", ls="--", lw=1.2, label="全量K=25 (成本25)")
k0 = fr[fr.signal == "all_k0"].iloc[0]
ax.axhline(k0.rec_strong, color="k", ls=":", lw=1, label="纯K=0免费")
ax.set_xlabel("平均微调步数/候选 (=25×路由预算)"); ax.set_ylabel("最强档(>0.20)召回")
ax.set_title("(b) 轴2：只微调被路由的候选\nS1-S2分歧是最优 truth-free 路由信号")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ---- (c) σ分档诊断：认知不确定性标不出强档 ----
ax = axes[2]
sb = pd.read_csv(ROOT / "outputs/sigma_by_band.csv")
x = np.arange(len(sb))
ax.bar(x, sb.sigma_med, color="#e67e22", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(sb.band, fontsize=9)
ax.set_xlabel("真实三阶协同强度档"); ax.set_ylabel("集成 σ 中位(seed0/1/2 的 S1 std)")
ax.set_title("(c) σ 诊断：最强档 σ 不升反平\n⟹ 摊销盲区是系统偏差,非方差,σ路由失明")
for i, v in enumerate(sb.sigma_med):
    ax.annotate(f"{v:.3f}\nn={sb.n[i]}", (i, v + 0.0002), ha="center", fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("DNN84：杠杆A(单调/增量目标) 与 轴2(不确定性门控级联)", fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / "figures/leverA_and_cascade_zh.png", dpi=150)
print("已写出 figures/leverA_and_cascade_zh.png")
