# -*- coding: utf-8 -*-
"""114 号 步骤 3：汇总三组结果，并与 113 号（沿用 103 号真值、未剔除目标副本）对照。出 2 张总览图与 report114.md。"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
A = os.path.join(ROOT, "outputs"); FG = os.path.join(ROOT, "figures")
NAME = {"metered_load_mw": "PJM 实际负荷", "total_gen": "PJM 发电总出力", "net_actual_interchange_mw": "PJM 联络线净交换",
        "actual_load__mw__ca_iso_tac": "CAISO 实际负荷"}
rows = []
for g in ["pjm_load", "pjm_gen_ic", "caiso_load"]:
    f = os.path.join(ROOT, "groups", g, "outputs", "analysis", "summary_targets.csv")
    if not os.path.exists(f):
        continue
    spec = json.load(open(os.path.join(ROOT, "groups", g, "outputs", "fields.json"), encoding="utf-8"))
    S = pd.read_csv(f); att = pd.read_csv(os.path.join(ROOT, "groups", g, "outputs", "analysis", "attacker_share.csv"))
    for _, r in S.iterrows():
        d = dict(组=g, 目标=NAME[r.目标], 候选数=len(spec["cand"]), 剔除的目标副本="、".join(spec["removed_copies"]) or "—",
                 最强单字段=r.最强单字段V, M0=r.M0均值, M2=r.M2均值, M3=r.M3均值, 升级=r["背景升级均值_M2减M0"],
                 M2除M3=r["K饱和_M2除M3"], 树模型占比=float(att.set_index("攻击器").loc["梯度提升树", "被val选中比例"]))
        for tau in (0.5, 0.7):
            for k in ("单字段即危险", "单看安全组合危险", "危险小组合", "单字段定级后仍暴露", "最少扣留"):
                d[f"τ{tau}_{k}"] = int(r[f"τ{tau}_{k}"])
        d.update(V误差=r.V_MAE, M2误差=r.M2_MAE, 排序Spearman=r.M2排序Spearman, 认证前3=r.认证下界比_top3)
        rows.append(d)
S = pd.DataFrame(rows)
old = pd.read_csv(os.path.join(REPO, "DNN_Aggresvation113", "outputs", "analysis", "113_summary.csv"))
old["目标"] = old["目标"].str.extract(r"（(.*)）")[0].map(NAME)
cmp_rows = []
for _, r in S.iterrows():
    for var, lab in [("A", "113 口径A（全候选）"), ("B", "113 口径B（剔除次日字段）")]:
        o = old[(old.目标 == r.目标) & (old.口径 == var)].iloc[0]
        cmp_rows.append(dict(目标=r.目标, 口径=lab, 候选数=int(o.候选数), 最强单字段=o.最大M0,
                             单看安全组合危险τ05=int(o["τ0.5_单看安全组合危险"]), 单看安全组合危险τ07=int(o["τ0.7_单看安全组合危险"])))
    cmp_rows.append(dict(目标=r.目标, 口径="114 现实口径（再剔除目标副本，重训）", 候选数=r.候选数, 最强单字段=r.最强单字段,
                         单看安全组合危险τ05=r["τ0.5_单看安全组合危险"], 单看安全组合危险τ07=r["τ0.7_单看安全组合危险"]))
Cm = pd.DataFrame(cmp_rows)
S.to_csv(os.path.join(A, "114_summary.csv"), index=False); Cm.to_csv(os.path.join(A, "114_vs_113.csv"), index=False)

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
INK, GRID = "#0b0b0b", "#e6e5e0"; COL = ["#cfcec8", "#9fc3ea", "#2a78d6"]
tg = S.目标.tolist(); x = np.arange(len(tg)); w = .26
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
labs = Cm.口径.drop_duplicates().tolist()
for j, lab in enumerate(labs):
    d = Cm[Cm.口径 == lab].set_index("目标").reindex(tg)
    axs[0].bar(x + (j - 1) * w, d.最强单字段, w * .92, color=COL[j], label=lab)
    axs[1].bar(x + (j - 1) * w, d.单看安全组合危险τ05 / d.候选数, w * .92, color=COL[j])
    for xi, v, nn in zip(x + (j - 1) * w, d.单看安全组合危险τ05, d.候选数):
        axs[1].text(xi, v / nn + .015, f"{v}/{nn}", ha="center", fontsize=7, color="#52514e")
axs[0].axhline(0.5, color="#8a8984", lw=.8, ls=":"); axs[0].set_ylabel("最强单字段推断能力")
axs[0].set_title("A. 剔除目标副本后，PJM 实际负荷不再被单字段直接暴露", loc="left", fontsize=10, color=INK)
axs[1].set_ylabel("占候选字段的比例"); axs[1].set_title("B. 单看安全、≤2 个背景即越过 τ=0.5 的字段", loc="left", fontsize=10, color=INK)
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
d = S.set_index("目标").reindex(tg)
axs[2].plot(x, d.M0, "o", ms=8, color="#eb6834", label="M^(0)"); axs[2].plot(x, d.M2, "s", ms=8, color="#2a78d6", label="M^(2)")
axs[2].plot(x, d.M3, "^", ms=7, color="#0d366b", label="M^(3)")
for xi, v in zip(x, d.M2除M3):
    axs[2].text(xi, d.M3.max() * 1.08, f"M2/M3\n{v:.3f}", ha="center", fontsize=7.5, color="#52514e")
axs[2].set_ylim(0, d.M3.max() * 1.3); axs[2].legend(frameon=False, fontsize=8, loc="lower right")
axs[2].set_title("C. K 递进（正式三攻击器口径到 K=3）", loc="left", fontsize=10, color=INK)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(tg, fontsize=8); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
fig.suptitle("图1  PJM / CAISO 现实口径（按披露时点收窄候选池 + 剔除目标副本 + 全部重训）与 113 号旧口径对照",
             x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig1_vs_113.png"), bbox_inches="tight"); plt.close(fig)
md = ["# 114 号汇总（自动生成；结论见 CHANGELOG.md）\n", "## 四个目标\n", S.round(4).to_markdown(index=False),
      "\n## 与 113 号对照\n", Cm.round(4).to_markdown(index=False)]
open(os.path.join(A, "report114.md"), "w", encoding="utf-8").write("\n".join(md))
print(S.round(3).T.to_string()); print(Cm.round(3).to_string(index=False))
