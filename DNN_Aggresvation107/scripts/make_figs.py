# -*- coding: utf-8 -*-
"""107 号总览图：实验矩阵 / 核心证据 / 遗留问题优先级。纯汇总，不占 GPU。"""
from pathlib import Path
import numpy as np, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
ROOT = Path(__file__).resolve().parents[1]; A = ROOT / "outputs/analysis"; F = ROOT / "figures"; F.mkdir(exist_ok=True)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 160})
BLUE, ORANGE, AQUA, RED, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#cc3d3d", "#a3a29c", "#2b2b2a"

# ---------------- 图1：实验矩阵（编号 × RQ 覆盖）----------------
rqs = ["A1\nV保真", "A2\n闭式读出", "A3\n随机掩码", "A4\nM保真", "A5\n效率", "B1\n传统指标", "B2\nK语义", "B3\nwitness", "B4\n基底B", "B5\n发布审查"]
nums = ["100", "101", "102", "103", "104", "105", "106"]
#  2=主证据  1=辅助证据  0=无
M = np.array([
    [1, 0, 0, 1, 2, 0, 1, 1, 2, 0],   # 100
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0],   # 101
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],   # 102
    [2, 2, 2, 1, 0, 0, 0, 0, 0, 0],   # 103
    [0, 0, 0, 2, 0, 1, 2, 1, 0, 0],   # 104
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0],   # 105
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 2],   # 106
])
fig, ax = plt.subplots(figsize=(12.6, 4.4))
for i in range(len(nums)):
    for j in range(len(rqs)):
        v = M[i, j]
        if v == 0: continue
        c = BLUE if v == 2 else "#bcd6f2"
        ax.add_patch(Rectangle((j - .42, i - .42), .84, .84, color=c))
        ax.text(j, i, "主" if v == 2 else "辅", ha="center", va="center", fontsize=9,
                color="white" if v == 2 else "#28517d")
ax.set_xlim(-.6, len(rqs) - .4); ax.set_ylim(len(nums) - .4, -.6)
ax.set_xticks(range(len(rqs))); ax.set_xticklabels(rqs, fontsize=9)
ax.set_yticks(range(len(nums)))
ax.set_yticklabels([f"{n} 号" for n in nums], fontsize=10)
ax.grid(False)
ax.set_title("图1  实验编号 × 研究问题覆盖矩阵（深色=主证据，浅色=辅助证据）\n"
             "Part A 五个 RQ 与 Part B 五个 RQ 均有主证据承载；A3 的结论是「掩码必要但分布不敏感」",
             loc="left", fontsize=11)
fig.tight_layout(); fig.savefig(F / "fig1_experiment_matrix.png", bbox_inches="tight"); plt.close(fig)

# ---------------- 图2：三条核心证据 ----------------
fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.3))
# (a) RQ-A2 共享输出头 vs 闭式读出
labs = ["共享输出头\n(direct)", "冻结 φ +\n子集闭式读出"]
x = np.arange(2); w = .34
pj = [0.0892, 0.0276]; ca = [0.0705, 0.0153]
ax[0].bar(x - w/2, pj, w*.92, color=BLUE, label="PJM"); ax[0].bar(x + w/2, ca, w*.92, color=ORANGE, label="CAISO")
for xi, v in zip(x - w/2, pj): ax[0].text(xi, v + .002, f"{v:.4f}", ha="center", fontsize=8.5)
for xi, v in zip(x + w/2, ca): ax[0].text(xi, v + .002, f"{v:.4f}", ha="center", fontsize=8.5)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=9); ax[0].set_ylabel("V-MAE（越低越好）")
ax[0].set_ylim(0, .105); ax[0].legend(frameon=False, fontsize=9); ax[0].grid(axis="x", visible=False)
ax[0].set_title("A. RQ-A2：闭式读出使 V-MAE 降 3.2×/4.6×", loc="left", fontsize=10)
# (b) RQ-B1 PR-AUC
names = ["置换\n重要性", "SAGE", "Pearson", "NMI", "单字段\nM^0(精确)", "估计\nM^1", "估计\nM^2"]
pj2 = [0.675, 0.730, 0.826, 0.822, 0.877, 0.888, 0.877]
ca2 = [0.460, 0.594, 0.729, 0.769, 0.764, 0.823, 0.829]
x2 = np.arange(len(names))
ax[1].bar(x2 - w/2, pj2, w*.92, color=[GRAY]*4 + [AQUA] + [BLUE]*2)
ax[1].bar(x2 + w/2, ca2, w*.92, color=[GRAY]*4 + [AQUA] + [ORANGE]*2)
ax[1].set_xticks(x2); ax[1].set_xticklabels(names, fontsize=7.8); ax[1].set_ylim(0, 1.0)
ax[1].set_ylabel("τ-critical 检测 PR-AUC（τ=0.7, K=2）"); ax[1].grid(axis="x", visible=False)
ax[1].axhline(0.877, color=AQUA, lw=.9, ls="--")
ax[1].set_title("B. RQ-B1：全量模型归因明显劣，但单字段 M^0（绿）已是很强基线\n每组左柱=PJM，右柱=CAISO；虚线=PJM 单字段 M^0 水平", loc="left", fontsize=9.5)
# (c) RQ-B5 发布审查
rules = ["单字段\n≤τ", "相关性\n≤0.5", "直接估计\nVhat≤τ", "逐阶预算\nU^(2)≤τ", "加性预算\nΣM≤τ"]
dang_p = [0.1908, 0.0211, 0.0657, 0.0, 0.0]; dang_c = [0.5640, 0.0315, 0.0264, 0.0, 0.0]
rej_p = [0.0, 0.3789, 0.0, 0.4962, 0.8634]; rej_c = [0.0, 0.2264, 0.0010, 0.3278, 0.7402]
x3 = np.arange(len(rules))
ax[2].bar(x3 - w/2, dang_p, w*.92, color=RED, label="危险放行率 PJM")
ax[2].bar(x3 + w/2, dang_c, w*.92, color="#f0907a", label="危险放行率 CAISO")
ax[2].plot(x3 - w/2, rej_p, "o--", color=INK, ms=4, lw=1, label="误拒率 PJM")
ax[2].plot(x3 + w/2, rej_c, "s--", color=GRAY, ms=4, lw=1, label="误拒率 CAISO")
ax[2].text(3, 0.055, "零危险放行\n且误拒<50%", ha="center", fontsize=8.5, color=AQUA)
ax[2].set_xticks(x3); ax[2].set_xticklabels(rules, fontsize=8); ax[2].set_ylim(0, 1.0)
ax[2].grid(axis="x", visible=False); ax[2].legend(frameon=False, fontsize=7.5, loc="upper center")
ax[2].set_title("C. RQ-B5：单字段规则放行 19%/56% 危险组合", loc="left", fontsize=10)
fig.suptitle("图2  三条核心证据：闭式读出的必要性（A）、M 相对传统指标的定位（B）、发布审查上的决策增益（C）",
             x=0.005, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(F / "fig2_core_evidence.png", bbox_inches="tight"); plt.close(fig)

# ---------------- 图3：遗留问题优先级 + 验收达成 ----------------
fig, ax = plt.subplots(1, 2, figsize=(13.4, 4.6))
# 左：验收标准达成
rows = list(csv.reader((A / "acceptance.csv").open(encoding="utf-8-sig")))[1:]
cmap = {"达成": AQUA, "部分达成": "#e8b93c", "偏离（已论证）": "#e8b93c", "已取消": GRAY, "未做": RED}
t3 = [r for r in rows if r[0] == "三区"]; t2 = [r for r in rows if r[0] == "二区"]
y = 0; ticks = []; labels = []
for grp, name in [(t3, "三区必须"), (t2, "二区建议")]:
    for r in grp:
        ax[0].barh(y, 1, color=cmap.get(r[3], GRAY), height=.72)
        ax[0].text(0.015, y, r[2], va="center", fontsize=7.6, color="white" if r[3] != "已取消" else "#3a3a38")
        ax[0].text(0.985, y, r[3], va="center", ha="right", fontsize=8, color="white" if r[3] != "已取消" else "#3a3a38")
        ticks.append(y); labels.append(f"{name[:2]}{r[1]}"); y += 1
    y += .6
ax[0].set_yticks(ticks); ax[0].set_yticklabels(labels, fontsize=7.5); ax[0].invert_yaxis()
ax[0].set_xlim(0, 1); ax[0].set_xticks([]); ax[0].grid(False)
ax[0].set_title("A. 验收标准达成（绿=达成，黄=部分/已论证偏离，灰=按用户决定取消，红=未做）", loc="left", fontsize=10)
# 右：遗留问题优先级散点
gaps = [("A1 PJM M 保真不足", 0.9, 0.15, RED), ("A2 CAISO 元训练缺失", 0.55, 0.06, "#e8b93c"),
        ("A3 K=3 口径不一致", 0.5, 0.35, "#e8b93c"), ("B1 p≥100 大字段空间", 0.75, 0.85, BLUE),
        ("B2 攻击器能力 sensitivity", 0.6, 0.2, AQUA), ("B3 XGB/LGBM 更强树", 0.65, 0.7, ORANGE),
        ("C 时间外推（已取消）", 0.2, 0.9, GRAY)]
for name, imp, cost, c in gaps:
    ax[1].scatter(cost, imp, s=170, color=c, zorder=3, edgecolor="white", lw=1.2)
    OFF = {"C ": ((-10, -14), "right"), "A2": ((9, -17), "left"), "A1": ((10, 2), "left")}
    dx, ha = OFF.get(name[:2], ((9, 7), "left"))
    ax[1].annotate(name, (cost, imp), textcoords="offset points", xytext=dx, fontsize=8.3, color=INK, ha=ha)
ax[1].axhline(.7, color=GRAY, lw=.8, ls=":"); ax[1].axvline(.4, color=GRAY, lw=.8, ls=":")
ax[1].text(.02, .97, "高重要 / 低成本 → 先做", fontsize=8.5, color=AQUA)
ax[1].text(.62, .04, "低重要 / 高成本 → 写局限", fontsize=8.5, color=GRAY)
ax[1].set_xlim(0, 1.08); ax[1].set_ylim(0, 1.05)
ax[1].set_xlabel("补做成本（相对）"); ax[1].set_ylabel("对投稿的重要性")
ax[1].set_title("B. 遗留问题优先级", loc="left", fontsize=10)
fig.suptitle("图3  验收对照与遗留问题：三区 10 条中 7 条达成、1 条已论证偏离、1 条部分达成、1 条按用户决定取消",
             x=0.005, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .93]); fig.savefig(F / "fig3_acceptance_gaps.png", bbox_inches="tight"); plt.close(fig)
print("3 张图已写入", F)
