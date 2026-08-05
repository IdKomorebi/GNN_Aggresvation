# -*- coding: utf-8 -*-
"""80 号图：syn3 与 v(ijk) 排序键的互补性（问题1机制）+ 层级生成覆盖率（问题2）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R80 = Path(__file__).resolve().parents[1]
for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

full = pd.read_parquet(R80 / "outputs/triple_scores.parquet")
full0 = full[full.K == 0]
sc = full0.dropna(subset=["syn3_true"]).copy()
BANDS = [(0.10, 0.12, "0.10-0.12"), (0.12, 0.15, "0.12-0.15"),
         (0.15, 0.20, "0.15-0.20"), (0.20, 1.01, ">0.20")]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

# 左：两键的全局百分位随强度档（互补交叉）
ax = axes[0]
xs, pv, ps = [], [], []
for lo, hi, bn in BANDS:
    b = sc[(sc.syn3_true >= lo) & (sc.syn3_true < hi) & sc.strong]
    xs.append(bn)
    pv.append(np.median([(full0.S2_vijk < v).mean() for v in b.S2_vijk]) * 100)
    ps.append(np.median([(full0.S1_syn3 < v).mean() for v in b.S1_syn3]) * 100)
x = np.arange(len(xs))
ax.plot(x, ps, "o-", color="#2980b9", lw=2.5, ms=9, label="syn3 键（现状 v(ijk)−max·pair）")
ax.plot(x, pv, "s-", color="#c0392b", lw=2.5, ms=9, label="v(ijk) 键（方向A 联合泄露）")
ax.axhline(70, color="gray", ls=":", lw=1.5, label="Top30% 门槛（百分位70）")
ax.set_xticks(x); ax.set_xticklabels(xs)
ax.set_xlabel("真实三阶协同强度档"); ax.set_ylabel("K=0 全局百分位中位（越高越靠前）")
ax.set_title("(a) 两个排序键的偏好完全相反，在最强档交叉\nsyn3 漏最强、v(ijk) 补最强")
ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=0.3); ax.set_ylim(0, 105)
for i in range(len(xs)):
    ax.annotate(f"{ps[i]:.0f}", (x[i], ps[i]+3), color="#2980b9", ha="center", fontsize=8)
    ax.annotate(f"{pv[i]:.0f}", (x[i], pv[i]-6), color="#c0392b", ha="center", fontsize=8)

# 右：层级候选生成覆盖率 vs 塌缩（问题2）
ax = axes[1]
th = [0.05, 0.10, 0.15]
cand = [12792, 9841, 5775]
cov = [100.0, 99.3, 77.6]
ax2 = ax.twinx()
b1 = ax.bar(np.arange(3) - 0.2, cand, 0.4, color="#95a5a6", label="三阶候选数")
ax.axhline(13244, color="k", ls="--", lw=1.2, label="全空间 C(44,3)=13244")
l1 = ax2.plot(np.arange(3) + 0.2, cov, "o-", color="#27ae60", lw=2.5, ms=10, label="强三阶覆盖率")
ax.set_xticks(range(3)); ax.set_xticklabels([f"强二阶阈值\n{t}" for t in th])
ax.set_ylabel("候选三元组数"); ax2.set_ylabel("已知强三阶覆盖率 (%)", color="#27ae60")
ax2.set_ylim(60, 105); ax2.tick_params(axis="y", labelcolor="#27ae60")
for i in range(3):
    ax2.annotate(f"{cov[i]:.1f}%", (i+0.2, cov[i]+1.5), color="#27ae60", ha="center", fontsize=9)
ax.set_title("(b) 层级候选生成（问题2出路）\n强二阶阈值0.1：9841候选覆盖99.3%强三阶")
ax.legend(loc="lower left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)

fig.suptitle("DNN80：问题1机制（左，两键互补）与问题2出路（右，层级生成）", fontsize=13)
fig.tight_layout()
fig.savefig(R80 / "figures/complementary_and_hierarchy_zh.png", dpi=150)
print("已写出 figures/complementary_and_hierarchy_zh.png")
