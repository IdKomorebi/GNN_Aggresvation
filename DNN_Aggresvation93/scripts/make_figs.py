# -*- coding: utf-8 -*-
"""93 号出图：认证塌陷、L1 提升、通用基边界。"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for f in ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"]:
    if any(f in fn.name for fn in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
R91 = ROOT.parent / "DNN_Aggresvation91"
OUT, FIG = ROOT / "outputs", ROOT / "figures"

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

# ---- (a) ★认证塌陷：结构化 syn vs 重训真值 ----
ax = axes[0, 0]
fs = sorted(glob.glob(str(OUT / "certify_o5_s*of*.csv")))
d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
for grp, col, lab in [("strong", "#e74c3c", "90号判定的强五阶"),
                      ("control", "#95a5a6", "对照组(syn<0.05)")]:
    g = d[d.group == grp]
    if len(g):
        ax.scatter(g.syn_struct, g.syn_true_audit, s=42, alpha=0.75,
                   color=col, label=f"{lab} (n={len(g)})", edgecolors="w", linewidths=0.5)
lim = max(0.6, float(d.syn_struct.max()) * 1.05)
ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x（估计准确）")
ax.axhline(0.2, color="#27ae60", ls=":", lw=1.2)
ax.annotate("强协同阈值 0.2", (0.02, 0.21), fontsize=8, color="#27ae60")
ax.set_xlabel("90号 full 字典的结构化 syn")
ax.set_ylabel("专用 DNN 重训的真值 syn")
ax.set_title(f"(a) ★结构化查询的高阶发现全面塌陷\n认证后无一超过 0.2（n={len(d)}）", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
ax.set_xlim(0, lim)
ax.set_ylim(-0.02, lim)

# ---- (b) 子集反超：根本不存在增量 ----
ax = axes[0, 1]
s = d[d.group == "strong"]
if len(s):
    ax.scatter(s.v_true_audit, s.maxsub_true_audit, s=42, alpha=0.8,
               color="#e74c3c", edgecolors="w", linewidths=0.5)
    lo = min(float(s.v_true_audit.min()), float(s.maxsub_true_audit.min())) - 0.02
    hi = max(float(s.v_true_audit.max()), float(s.maxsub_true_audit.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="#e74c3c", alpha=0.08)
    n_ex = int((s.maxsub_true_audit > s.v_true_audit).sum())
    ax.annotate(f"阴影区 = 四阶子集真值\n**高于**五元组自己\n{n_ex}/{len(s)} "
                f"({n_ex/len(s)*100:.0f}%)", (lo + 0.02, hi - 0.08),
                fontsize=9, color="#c0392b")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
ax.set_xlabel("五元组自身的重训真值 v(S)")
ax.set_ylabel("最强四阶子集的重训真值 max_T v(T)")
ax.set_title("(b) ★多数集合根本没有高阶增量\n第 5 个字段反而拉低泛化", fontsize=10)
ax.grid(alpha=0.3)

# ---- (c) ★三组认证对比：谁选的 top 才是真的 ----
ax = axes[0, 2]
import glob as _g
l1 = pd.concat([pd.read_csv(f) for f in
                _g.glob(str(OUT / "certify_o5_l1top_s*.csv"))]).drop_duplicates("S")
groups = [("L1 自选 top", l1, "#27ae60"),
          ("full 字典 top", d[d.group == "strong"], "#e74c3c"),
          ("随机对照", d[d.group == "control"], "#95a5a6")]
pos = np.arange(len(groups))
for i, (lab, g, col) in enumerate(groups):
    y = g.syn_true_audit.values
    ax.scatter(np.random.RandomState(i).normal(i, 0.07, len(y)), y,
               s=30, alpha=0.7, color=col, edgecolors="w", linewidths=0.4)
    ax.hlines(np.median(y), i - 0.24, i + 0.24, color="k", lw=2)
    ax.annotate(f"n={len(y)}\n过0.10: {int((y>0.10).sum())}", (i, -0.012),
                ha="center", fontsize=8, color=col)
ax.axhline(0.10, color="#27ae60", ls=":", lw=1.2)
ax.annotate("0.10", (2.35, 0.103), fontsize=8, color="#27ae60")
ax.set_xticks(pos)
ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("专用 DNN 重训的真值 syn")
ax.set_title("(c) ★★谁选的 top 才是真的\nL1>full p=0.013；full vs 随机 p=0.058(无差异)",
             fontsize=10)
ax.set_ylim(-0.02, 0.175)
ax.grid(alpha=0.3, axis="y")

# ---- (d) ★L1 vs L0：二阶 ----
ax = axes[1, 0]
rows = []
p91 = pd.read_csv(R91 / "outputs/order2_truth.csv")
for k in ["last", "cat3"]:
    r = p91[p91.kind == k]
    if len(r):
        rows.append(dict(ckpt="L0 冻结", kind=k, rho=r.rho.iloc[0], tail=r.rho_tail.iloc[0]))
for tag, lab in [("_l1", "L1 base"), ("_long", "L1 long"), ("_aug8", "L1 aug8")]:
    p = OUT / f"eval_l1_order2{tag}.csv"
    if p.exists():
        x = pd.read_csv(p)
        for k in ["last", "cat3"]:
            r = x[x.kind == k]
            if len(r):
                rows.append(dict(ckpt=lab, kind=k, rho=r.rho.iloc[0], tail=r.rho_tail.iloc[0]))
e2 = pd.DataFrame(rows)
ck = ["L0 冻结", "L1 base", "L1 long", "L1 aug8"]
ck = [c for c in ck if c in set(e2.ckpt)]
x = np.arange(len(ck))
for i, (k, col) in enumerate([("last", "#27ae60"), ("cat3", "#2980b9")]):
    g = e2[e2.kind == k].set_index("ckpt").reindex(ck)
    ax.plot(x, g["tail"].values, "o-", color=col, lw=2, label=f"{k} 尾部ρ")
    ax.plot(x, g.rho.values, "s--", color=col, lw=1.2, alpha=0.6, label=f"{k} 全体ρ")
ax.set_xticks(x)
ax.set_xticklabels(ck, fontsize=8)
ax.set_ylabel("Spearman ρ（二阶，对重训真值）")
ax.set_title("(d) ★闭式头进训练回路后再提升\n尾部 ρ 0.781 → 0.858", fontsize=10)
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# ---- (e) L1 vs L0：三阶 ----
ax = axes[1, 1]
rows = []
p3 = pd.read_csv(R91 / "outputs/order3_truth.csv")
for k in ["last", "cat3"]:
    r = p3[p3.kind == k]
    if len(r):
        rows.append(dict(ckpt="L0 冻结", kind=k, tail=r.rho_tail.iloc[0],
                         mr=r.med_rank_true_top10.iloc[0]))
for tag, lab in [("_l1", "L1 base"), ("_aug8", "L1 aug8")]:
    p = OUT / f"eval_l1_order3{tag}.csv"
    if p.exists():
        x_ = pd.read_csv(p)
        for k in ["last", "cat3"]:
            r = x_[x_.kind == k]
            if len(r):
                rows.append(dict(ckpt=lab, kind=k, tail=r.rho_tail.iloc[0],
                                 mr=r.med_rank_true_top10.iloc[0]))
e3 = pd.DataFrame(rows)
ck3 = [c for c in ["L0 冻结", "L1 base", "L1 aug8"] if c in set(e3.ckpt)]
x = np.arange(len(ck3))
g = e3[e3.kind == "last"].set_index("ckpt").reindex(ck3)
ax.bar(x, g["tail"].values, 0.5, color="#27ae60", label="last 尾部ρ")
ax2 = ax.twinx()
ax2.plot(x, g["mr"].values, "ko--", ms=6, label="真top10排名（右轴，越低越好）")
ax2.set_ylabel("真值 top10 的估计排名中位数")
ax2.legend(fontsize=7, loc="upper right")
ax.set_xticks(x)
ax.set_xticklabels(ck3, fontsize=8)
ax.set_ylabel("三阶尾部 Spearman ρ")
ax.set_title("(e) 三阶同样提升\n真top10排名 9.0 → 4.5", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
ax.grid(alpha=0.3, axis="y")

# ---- (f) ✗通用基边界：增广没修好 ----
ax = axes[1, 2]
frames = []
for p in list(glob.glob(str(R91 / "outputs/inject_l0_summary*.csv"))) + \
         list(glob.glob(str(OUT / "inject_l0_summary_aug8.csv"))):
    x_ = pd.read_csv(p)
    tag = "L1-aug8 " if "93" in p else "L0 "
    x_["lab"] = tag + x_["kind"]
    frames.append(x_)
inj = pd.concat(frames).drop_duplicates(subset=["lab", "alpha"])
for lab, col, ls in [("L0 last", "#27ae60", "-"), ("L0 cat3", "#2980b9", "-"),
                     ("L1-aug8 last", "#27ae60", "--"), ("L1-aug8 cat3", "#2980b9", "--"),
                     ("L0 full", "#7f8c8d", "-")]:
    g = inj[inj.lab == lab].sort_values("alpha")
    if len(g):
        ax.plot(g.alpha, g.pct, "o" + ls, color=col, lw=2 if ls == "-" else 1.6,
                label=lab.replace("L0 full", "full 手工字典"))
ax.axhline(50, color="k", ls=":", lw=1)
ax.set_xlabel("注入的合成纯四阶信号强度 α")
ax.set_ylabel("注入集合在 400 个随机四元组中的百分位")
ax.set_title("(f) ✗合成目标增广没修好通用基\n虚线=aug8；两者都远不及手工 full 字典", fontsize=10)
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.suptitle("93 号：手工字典在高阶上既报假的又漏真的；学出来的特征零假阳性且是唯一找到真协同的",
             fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.975))
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "certify_and_l1.png", dpi=155)
print(f"已保存 {FIG / 'certify_and_l1.png'}")
