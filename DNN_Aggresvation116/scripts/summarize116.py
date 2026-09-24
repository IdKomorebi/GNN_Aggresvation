# -*- coding: utf-8 -*-
"""116 号 步骤 5：汇总——(1) 补全候选池前后（114 → 116）的结论对照；(2) 近似代理字段的敏感性（不重训，在真值表上限制候选）；
(3) 主干对比图。出 3 张图与 report116.md。"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import milp, LinearConstraint, Bounds

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
A = os.path.join(ROOT, "outputs", "analysis"); FG = os.path.join(ROOT, "figures")
NAME = {"metered_load_mw": "PJM 实际负荷", "total_gen": "PJM 发电总出力", "net_actual_interchange_mw": "PJM 联络线净交换",
        "actual_load__mw__ca_iso_tac": "CAISO 实际负荷"}
KEYS = ["最强单字段V", "τ0.5_单字段即危险", "τ0.5_单看安全组合危险", "τ0.5_危险小组合", "τ0.5_单字段定级后仍暴露", "τ0.5_最少扣留",
        "τ0.7_单字段即危险", "τ0.7_单看安全组合危险", "τ0.7_危险小组合", "τ0.7_单字段定级后仍暴露", "τ0.7_最少扣留",
        "M0均值", "M2均值", "M3均值", "K饱和_M2除M3"]


def hit(mus, p):
    if not mus:
        return 0
    Am = np.zeros((len(mus), p))
    for r, m in enumerate(mus):
        Am[r, list(m)] = 1
    return int(round(milp(c=np.ones(p), constraints=LinearConstraint(Am, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).fun))


def restricted(O, drop):
    """在真值表上把候选限制为不含 drop 的字段（不重训）。"""
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8")); z = np.load(os.path.join(O, "D.npz"))
    keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]; V = np.load(os.path.join(O, "V_official.npy"))
    s3 = [k for k in keys if len(k) <= 3]; V3 = V[[len(k) <= 3 for k in keys]]; idx = {k: r for r, k in enumerate(s3)}
    fields = [i for i, c in enumerate(spec["cand"]) if c not in drop]; out = []
    for c, y in enumerate(spec["targ"]):
        rec = dict(目标=NAME[y.replace("Y_", "")], 候选数=len(fields), 最强单字段=float(max(V3[idx[(i,)], c] for i in fields)))
        for tau in (0.5, 0.7):
            mus = pipe.mus_list(V3, s3, fields, c, tau, 3)
            c2 = {i for m in mus for i in m}; c0 = {i for i in fields if V3[idx[(i,)], c] > tau}
            rec[f"τ{tau}_单看安全组合危险"] = f"{len(c2 - c0)}/{len(fields)}"; rec[f"τ{tau}_危险小组合"] = len(mus)
        out.append(rec)
    return out


def main():
    rows = []
    for g in ["pjm_load", "pjm_gen_ic", "caiso_load"]:
        for ver, base in [("114 号（候选未补全）", os.path.join(REPO, "DNN_Aggresvation114", "groups", g, "outputs")),
                          ("116 号（补全候选）", os.path.join(ROOT, "groups", g, "outputs"))]:
            S = pd.read_csv(os.path.join(base, "analysis", "summary_targets.csv"))
            spec = json.load(open(os.path.join(base, "fields.json"), encoding="utf-8"))
            for _, r in S.iterrows():
                rows.append(dict(目标=NAME[r.目标], 版本=ver, 候选数=len(spec["cand"]), **{k: r[k] for k in KEYS}))
    C = pd.DataFrame(rows); C.to_csv(os.path.join(A, "116_vs_114.csv"), index=False)
    # 近似代理敏感性
    pjm_proxy = ["forecast_load_mw_latest_available", "forecast_load_mw_day_ahead"]
    caiso_proxy = ["dam_load_forecast__mw__pge_tac", "dam_load_forecast__mw__sce_tac", "dam_load_forecast__mw__sdge_tac",
                   "dam_schedule__mw__load__tac_north", "dam_schedule__mw__load__tac_ecntr", "dam_schedule__mw__load__tac_south",
                   "dam_schedule__mw__generation__caiso_totals", "dam_schedule__mw__import__caiso_totals", "dam_schedule__mw__export__caiso_totals"]
    sens = []
    for g, drop, lab in [("pjm_gen_ic", [], "现实口径"), ("pjm_gen_ic", pjm_proxy, "再剔除两个系统负荷预测"),
                         ("caiso_load", [], "现实口径"), ("caiso_load", caiso_proxy, "再剔除分区负荷预测、日前分区负荷计划与日前发电/进出口计划")]:
        for r in restricted(os.path.join(ROOT, "groups", g, "outputs"), drop):
            if g == "pjm_gen_ic" and r["目标"] != "PJM 发电总出力":
                continue
            sens.append(dict(口径=lab, **r))
    Sn = pd.DataFrame(sens); Sn.to_csv(os.path.join(A, "116_proxy_sensitivity.csv"), index=False)
    B = pd.read_csv(os.path.join(A, "backbone_compare.csv")); Bm = pd.read_csv(os.path.join(A, "backbone_compare_mean.csv"), index_col=0)
    # ---------------- 图
    font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                         "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
    INK, GRID = "#0b0b0b", "#e6e5e0"; COL = ["#cfcec8", "#2a78d6"]
    tg = C.目标.drop_duplicates().tolist(); x = np.arange(len(tg)); w = .38
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for j, ver in enumerate(C.版本.drop_duplicates()):
        d = C[C.版本 == ver].set_index("目标").reindex(tg)
        axs[0].bar(x + (j - .5) * w, d.最强单字段V, w * .92, color=COL[j], label=ver)
        axs[1].bar(x + (j - .5) * w, d["τ0.5_单看安全组合危险"] / d.候选数, w * .92, color=COL[j])
        axs[2].bar(x + (j - .5) * w, d["τ0.5_危险小组合"], w * .92, color=COL[j])
        for xi, v, nn in zip(x + (j - .5) * w, d["τ0.5_单看安全组合危险"], d.候选数):
            axs[1].text(xi, v / nn + .015, f"{v}/{nn}", ha="center", fontsize=7.5, color="#52514e")
        for xi, v in zip(x + (j - .5) * w, d["τ0.5_危险小组合"]):
            axs[2].text(xi, v + 5, f"{v}", ha="center", fontsize=7.5, color="#52514e")
    axs[0].axhline(.5, color="#8a8984", lw=.8, ls=":"); axs[0].set_ylabel("最强单字段推断能力"); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title("A. 最强单字段", loc="left", fontsize=10, color=INK)
    axs[1].set_ylabel("占候选字段比例"); axs[1].set_title("B. τ=0.5 单看安全、≤2 个背景即越阈", loc="left", fontsize=10, color=INK)
    axs[2].set_ylabel("个数"); axs[2].set_title("C. τ=0.5 规模 ≤3 的危险小组合（最小不安全集）", loc="left", fontsize=10, color=INK)
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(tg, fontsize=8.5); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle("图1  补全候选池（电价分量、日前计划、辅助服务出清总量进入候选）前后的结论对照", x=.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig1_vs_114.png"), bbox_inches="tight"); plt.close(fig)
    order = ["本组主干", "全列主干", "留目标主干"]; cols3 = ["#a3a29c", "#2a78d6", "#eb6834"]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
    m = Bm.reindex(order)
    axs[0].bar(np.arange(3), m.死神经元比例 * 100, .55, color=cols3)
    for xi, v, dm in zip(range(3), m.死神经元比例 * 100, m.有效维度):
        axs[0].text(xi, v + .6, f"{v:.1f}%\n有效维度 {dm:.2f}", ha="center", fontsize=8.5, color="#52514e")
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(order); axs[0].set_ylim(0, 32); axs[0].set_ylabel("死神经元比例（%）"); axs[0].set_title("A. 表征：全列主干的隐藏层更健康", loc="left", fontsize=10, color=INK)
    xx = np.arange(3); w = .26
    for j, bb in enumerate(order):
        axs[1].bar(xx + (j - 1) * w, [m.loc[bb, "V绝对误差"], m.loc[bb, "边际Δ绝对误差"], m.loc[bb, "M2误差"]], w * .92, color=cols3[j], label=bb)
    axs[1].set_xticks(xx); axs[1].set_xticklabels(["V 误差", "边际 Δ 误差", "M^(2) 误差"]); axs[1].legend(frameon=False, fontsize=8)
    axs[1].set_title("B. 误差：V 与边际都降低；留目标主干的 M 被最大值抬高", loc="left", fontsize=10, color=INK)
    axs[2].bar(xx[:2] - .2, [m.loc["本组主干", "认证前1"], m.loc["本组主干", "认证前3"]], .19, color=cols3[0])
    axs[2].bar(xx[:2], [m.loc["全列主干", "认证前1"], m.loc["全列主干", "认证前3"]], .19, color=cols3[1])
    axs[2].bar(xx[:2] + .2, [m.loc["留目标主干", "认证前1"], m.loc["留目标主干", "认证前3"]], .19, color=cols3[2])
    axs[2].set_xticks(xx[:2]); axs[2].set_xticklabels(["认证前 1 个背景", "认证前 3 个背景"]); axs[2].set_ylim(.6, 1.0)
    axs[2].set_title("C. 扫描—认证：留目标主干认证 3 个背景即恢复", loc="left", fontsize=10, color=INK)
    for ax in axs:
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.suptitle("图2  三种主干（读出方式相同）：本组主干 / 全列主干（一次预训练、任意目标复用）/ 留目标主干（预训练时整列删去目标）；4 个目标平均",
                 x=.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, .9]); fig.savefig(os.path.join(FG, "fig2_backbones.png"), bbox_inches="tight"); plt.close(fig)
    md = ["# 116 号汇总（自动生成；结论见 CHANGELOG.md）\n", "## 补全候选前后\n", C.round(4).to_markdown(index=False),
          "\n## 近似代理敏感性\n", Sn.round(4).to_markdown(index=False), "\n## 主干对比（逐目标）\n", B.round(4).to_markdown(index=False),
          "\n## 主干对比（平均）\n", Bm.round(4).to_markdown()]
    open(os.path.join(A, "report116.md"), "w", encoding="utf-8").write("\n".join(md))
    print(C.round(3).to_string(index=False)); print(Sn.to_string(index=False))


if __name__ == "__main__":
    main()
