# -*- coding: utf-8 -*-
"""115 号 步骤 3：各读出方式在 5 个数据/分组、11 个目标上的保真度对照，出 3 张图与报告。"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); REPO = os.path.abspath(os.path.join(ROOT, ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
A = os.path.join(ROOT, "outputs", "analysis"); FG = os.path.join(ROOT, "figures")
EXPS = [("RTS-GMLC", "DNN_Aggresvation111"), ("NEM", "DNN_Aggresvation112"),
        ("PJM 负荷", "DNN_Aggresvation114/groups/pjm_load"), ("PJM 发电/联络线", "DNN_Aggresvation114/groups/pjm_gen_ic"),
        ("CAISO 负荷", "DNN_Aggresvation114/groups/caiso_load")]
VAR = {"A": "A 原口径·三种子拼接", "B": "B 原口径·单种子", "C": "C 三种子拼接+数值修正", "D": "D 单种子+数值修正+5折",
       "E": "E 三种子预测平均", "F": "F 预测平均+交叉拟合", "G": "G 全字段主干复用（设置同 E）"}
KS = [1, 3, 10]


def crit(V, keys3, p, c, tau):
    idx = {k: r for r, k in enumerate(keys3)}; out = np.zeros(p, bool)
    for i in range(p):
        oth = [j for j in range(p) if j != i]
        for T in [()] + [(j,) for j in oth] + [tuple(sorted(x)) for x in __import__("itertools").combinations(oth, 2)]:
            vT = V[idx[T], c] if T else 0.0
            if vT <= tau < V[idx[tuple(sorted(T + (i,)))], c]:
                out[i] = True; break
    return out


rows, curves = [], []
for tag, rel in EXPS:
    O = os.path.join(REPO, rel, "outputs")
    if not os.path.exists(os.path.join(O, "est_variants.npz")) or not os.path.exists(os.path.join(O, "V_official.npy")):
        print("跳过（未完成）", tag); continue
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    ev = np.load(os.path.join(O, "est_variants.npz")); sel = ev["sel"]; keys3 = [k for k, s in zip(keys, sel) if s]
    V = np.load(os.path.join(O, "V_official.npy"))[sel]; E0 = np.load(os.path.join(O, "est.npz"))
    raw = {"A": E0["est"][sel], "B": E0["est_1seed"][sel], "C": ev["C"], "D": ev["D"], "E": ev["E"]}
    est = {k: pipe.closure_max(v, keys3) for k, v in raw.items()}
    Fa, Fb = pipe.closure_max(ev["Fa"], keys3), pipe.closure_max(ev["Fb"], keys3); est["F"] = (Fa + Fb) / 2
    if os.path.exists(os.path.join(O, "est_G.npz")):
        raw["G"] = np.load(os.path.join(O, "est_G.npz"))["G"]; est["G"] = pipe.closure_max(raw["G"], keys3)
    p = len(spec["cand"]); fields = list(range(p))
    _, _, Dt, _, bsz = pipe.m_table(V, keys3, fields, 2); s2 = np.where(bsz <= 2)[0]
    Da = pipe.m_table(Fa, keys3, fields, 2)[2][:, s2]; Db = pipe.m_table(Fb, keys3, fields, 2)[2][:, s2]
    for vk, Ve in est.items():
        De = pipe.m_table(Ve, keys3, fields, 2)[2][:, s2]
        for c, y in enumerate(spec["targ"]):
            dt, de = Dt[:, s2, c], De[:, :, c]; Mt = dt.max(1)
            if vk == "F":   # 交叉拟合：一半挑背景、另一半评估，两个方向取平均
                ja, jb = Da[:, :, c].argmax(1), Db[:, :, c].argmax(1)
                Me = 0.5 * (np.take_along_axis(Db[:, :, c], ja[:, None], 1)[:, 0] + np.take_along_axis(Da[:, :, c], jb[:, None], 1)[:, 0])
            else:
                Me = de.max(1)
            order = np.argsort(-de, 1)
            rec = dict(数据=tag, 目标=y.replace("Y_", ""), 读出=VAR[vk], V误差=float(np.abs(Ve[:, c] - V[:, c]).mean()),
                       M2偏差=float((Me - Mt).mean()), M2误差=float(np.abs(Me - Mt).mean()),
                       M2排序Spearman=float(pd.Series(Me).corr(pd.Series(Mt), method="spearman")),
                       崩溃占比=float(((raw.get(vk, (ev["Fa"] + ev["Fb"]) / 2)[:, c] <= 1e-6) & (V[:, c] > 0.15)).mean()))
            for k in KS:
                rec[f"认证前{k}"] = float(np.take_along_axis(dt, order[:, :k], 1).max(1).sum() / max(Mt.sum(), 1e-9))
            tc, ec = crit(V, keys3, p, c, 0.5), crit(Ve, keys3, p, c, 0.5)
            rec["关键召回τ0.5"] = float((tc & ec).sum() / max(tc.sum(), 1)); rec["关键精确τ0.5"] = float((tc & ec).sum() / max(ec.sum(), 1))
            rows.append(rec)
        print(f"{tag} {VAR[vk]} 完成", flush=True)
    sec = dict(C=None)
R = pd.DataFrame(rows); R.to_csv(os.path.join(A, "115_variants.csv"), index=False)
agg = R.groupby("读出")[["V误差", "M2偏差", "M2误差", "M2排序Spearman", "认证前1", "认证前3", "认证前10", "关键召回τ0.5", "关键精确τ0.5"]].mean()
agg = agg.reindex([v for v in VAR.values() if v in agg.index]); agg.to_csv(os.path.join(A, "115_variants_mean.csv"))

# ---------------- 图
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})
INK, GRID = "#0b0b0b", "#e6e5e0"
PAL = {"A": "#a3a29c", "B": "#cfcec8", "C": "#86b6ef", "D": "#eb6834", "E": "#2a78d6", "F": "#1baf7a", "G": "#e87ba4"}
SHORT = {"metered_load_mw": "实际负荷", "total_gen": "发电总出力", "net_actual_interchange_mw": "联络线净交换",
         "actual_load__mw__ca_iso_tac": "实际负荷"}
R["组"] = R["数据"] + "\n" + R["目标"].map(lambda t: SHORT.get(t, t))
groups = R["组"].drop_duplicates().tolist(); x = np.arange(len(groups)); vks = [k for k in VAR if VAR[k] in R.读出.unique()]
w = 0.8 / len(vks)
fig, axs = plt.subplots(2, 1, figsize=(16, 8.4), sharex=True)
for j, vk in enumerate(vks):
    d = R[R.读出 == VAR[vk]].set_index("组").reindex(groups)
    axs[0].bar(x + (j - (len(vks) - 1) / 2) * w, d.M2偏差, w * .9, color=PAL[vk], label=VAR[vk])
    axs[1].bar(x + (j - (len(vks) - 1) / 2) * w, d.认证前3, w * .9, color=PAL[vk])
axs[0].axhline(0, color=INK, lw=.8); axs[0].set_ylabel("M^(2) 平均偏差（估计 − 精确）")
axs[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); axs[0].grid(axis="y", color=GRID); axs[0].set_axisbelow(True)
axs[0].set_title("A. 字段风险的偏差：原口径三种子在真实数据上系统性高估", loc="left", fontsize=10.2, color=INK)
axs[1].set_ylim(0.3, 1.02); axs[1].axhline(0.95, color="#8a8984", lw=.8, ls=":"); axs[1].set_ylabel("认证前 3 个背景的下界比")
axs[1].grid(axis="y", color=GRID); axs[1].set_axisbelow(True)
axs[1].set_title("B. 扫描—认证：只认证估计器挑出的前 3 个背景，下界能恢复精确 M^(2) 的比例（点线 95%）", loc="left", fontsize=10.2, color=INK)
axs[1].set_xticks(x); axs[1].set_xticklabels(groups, fontsize=7.8)
fig.suptitle("图1  七种读出方式在 5 个数据/分组、11 个目标上的字段风险保真度（G 只在 114 号三组上评估）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(os.path.join(FG, "fig1_variants_by_target.png"), bbox_inches="tight"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8))
for vk in vks:
    d = R[R.读出 == VAR[vk]]
    axs[0].scatter(d.V误差, d.M2误差, s=34, color=PAL[vk], label=VAR[vk], edgecolor="#fcfcfb", lw=.8, zorder=3)
axs[0].set_xlabel("集合推断能力 V 的平均误差"); axs[0].set_ylabel("字段风险 M^(2) 的平均误差")
axs[0].grid(color=GRID); axs[0].set_axisbelow(True); axs[0].legend(frameon=False, fontsize=7.5)
axs[0].set_title("A. V 误差小不代表 M 误差小（每点一个目标）", loc="left", fontsize=10.2, color=INK)
m = agg.reset_index(); xx = np.arange(len(m))
for k, (col, lab) in enumerate([("认证前1", "前 1"), ("认证前3", "前 3"), ("认证前10", "前 10")]):
    axs[1].bar(xx + (k - 1) * .26, m[col], .24, color=["#9fc3ea", "#3987e5", "#0d366b"][k], label=f"认证{lab}个背景")
axs[1].set_xticks(xx); axs[1].set_xticklabels([s.split(" ", 1)[0] for s in m.读出]); axs[1].set_ylim(0.5, 1.02)
axs[1].axhline(0.95, color="#8a8984", lw=.8, ls=":"); axs[1].legend(frameon=False, fontsize=8, loc="lower right")
axs[1].grid(axis="y", color=GRID); axs[1].set_axisbelow(True); axs[1].set_ylabel("目标平均的认证下界比（A–F 为 11 个，G 为 4 个）")
axs[1].set_title("B. 各读出方式的认证效率（G 只有 4 个目标，不宜与其余直接横比）", loc="left", fontsize=10.2, color=INK)
fig.suptitle("图2  选估计器要看边际保真度：认证效率的全局对照", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig2_variant_summary.png"), bbox_inches="tight"); plt.close(fig)

open(os.path.join(A, "report115.md"), "w", encoding="utf-8").write(
    "# 115 号：读出方式保真度对照（自动生成；结论见 CHANGELOG.md）\n\n## 11 个目标平均\n\n" + agg.round(4).to_markdown()
    + "\n\n## 逐目标\n\n" + R.drop(columns="组").round(4).to_markdown(index=False))
print(agg.round(3).to_string())
