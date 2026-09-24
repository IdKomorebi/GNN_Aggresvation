# -*- coding: utf-8 -*-
"""111/112 号通用分析：正式真值（三攻击器 max + 单调闭包）→ 字段风险表、增益矩阵、单字段口径暴露、
最少扣留、通用模型保真度与认证；出 4 张图与 report.md。
用法：analyze_exp.py --exp ../DNN_Aggresvation111 [--gain_target 列名]"""
import os, sys, json, argparse, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "src"))
import pipe  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--gain_target", default="")
a = ap.parse_args()
EXP = os.path.abspath(a.exp); OUT = os.path.join(EXP, "outputs"); AN = os.path.join(OUT, "analysis"); FG = os.path.join(EXP, "figures")
os.makedirs(AN, exist_ok=True); os.makedirs(FG, exist_ok=True)
spec = json.load(open(os.path.join(OUT, "fields.json"), encoding="utf-8"))
z = np.load(os.path.join(OUT, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte"]}
cand, targ = spec["cand"], spec["targ"]; p, C = len(cand), len(targ); KB = spec["kmax"] - 1
short = lambda s: s.replace("Y_", "")

# ---------------- 正式真值
parts, names = [], []
for nm, lab in [("single", "单目标 DNN"), ("multi", "多目标 DNN"), ("tree", "梯度提升树")]:
    f = os.path.join(OUT, f"truth_{nm}.npz")
    if os.path.exists(f):
        t = np.load(f); parts.append((t["clean"], t["val"])); names.append(lab)
V, share = pipe.official_truth(parts, keys)
np.save(os.path.join(OUT, "V_official.npy"), V)
E = np.load(os.path.join(OUT, "est.npz")); Ve = pipe.closure_max(E["est"], keys); Ve1 = pipe.closure_max(E["est_1seed"], keys)
idx = {k: r for r, k in enumerate(keys)}
fields = list(range(p))
M, W, Dt, bks, bsz = pipe.m_table(V, keys, fields, KB)
Me_all, _, De, _, _ = pipe.m_table(Ve, keys, fields, KB)
r2 = pipe.r2_single_linear(D)
sizes = np.array([len(k) for k in keys])

rows, prof = [], []
for c, y in enumerate(targ):
    rec = dict(目标=short(y), 规则依据=spec["rule"].get(y, ""))
    for K in range(KB + 1):
        rec[f"M{K}均值"] = float(M[:, K, c].mean())
    rec["背景升级均值_M2减M0"] = float((M[:, 2, c] - M[:, 0, c]).mean())
    rec["K饱和_M1除M2"] = float(M[:, 1, c].sum() / max(M[:, 2, c].sum(), 1e-9))
    if KB >= 3:
        rec["K饱和_M2除M3"] = float(M[:, 2, c].sum() / max(M[:, 3, c].sum(), 1e-9))
    rec["最强单字段V"] = float(max(V[idx[(i,)], c] for i in fields))
    rec["全部规模≤{}集合最大V".format(spec["kmax"])] = float(V[:, c].max())
    for tau in spec["taus"]:
        crit0 = np.array([V[idx[(i,)], c] > tau for i in fields])
        mus = pipe.mus_list(V, keys, fields, c, tau, 3)
        crit2 = np.array([any(i in m for m in mus) for i in fields])
        single = {m[0] for m in mus if len(m) == 1}
        left = [m for m in mus if not set(m) & single]
        if mus:
            Am = np.zeros((len(mus), p))
            for r_, m in enumerate(mus):
                Am[r_, list(m)] = 1
            hs = int(round(milp(c=np.ones(p), constraints=LinearConstraint(Am, lb=1), integrality=np.ones(p),
                                bounds=Bounds(0, 1)).fun))
        else:
            hs = 0
        rec.update({f"τ{tau}_单字段即危险": int(crit0.sum()), f"τ{tau}_K2关键": int(crit2.sum()),
                    f"τ{tau}_单看安全组合危险": int((crit2 & ~crit0).sum()), f"τ{tau}_危险小组合": len(mus),
                    f"τ{tau}_单字段定级后仍暴露": len(left), f"τ{tau}_最少扣留": hs})
    Mt, Mest, L1, L3 = pipe.certify(Dt[:, :, c], De[:, :, c], bsz, 2)
    rec.update(V_MAE=float(np.abs(Ve[:, c] - V[:, c]).mean()), V_MAE_单种子=float(np.abs(Ve1[:, c] - V[:, c]).mean()),
               M2_MAE=float(np.abs(Mest - Mt).mean()),
               M2排序Spearman=float(pd.Series(Mest).corr(pd.Series(Mt), method="spearman")),
               认证下界比_top1=float(L1.sum() / max(Mt.sum(), 1e-9)), 认证下界比_top3=float(L3.sum() / max(Mt.sum(), 1e-9)))
    rows.append(rec)
    for i in fields:
        d = dict(目标=short(y), 字段=cand[i], r2=float(r2[i, c]))
        for K in range(KB + 1):
            d[f"M{K}"] = float(M[i, K, c])
        d["估计M2"] = float(Me_all[i, 2, c]); d["升级_M2减M0"] = d["M2"] - d["M0"]
        d["K2见证"] = " + ".join(cand[j] for j in W[i][2][c]) or "∅"
        prof.append(d)
S = pd.DataFrame(rows); P = pd.DataFrame(prof)
S.to_csv(os.path.join(AN, "summary_targets.csv"), index=False); P.to_csv(os.path.join(AN, "field_profile.csv"), index=False)
att = pd.DataFrame({"攻击器": names, "被val选中比例": share}); att.to_csv(os.path.join(AN, "attacker_share.csv"), index=False)
by_size = pd.DataFrame([dict(规模=s, 正式真值均值=float(V[sizes == s].mean()),
                             通用模型MAE=float(np.abs(Ve[sizes == s] - V[sizes == s]).mean())) for s in range(1, spec["kmax"] + 1)])
by_size.to_csv(os.path.join(AN, "v_by_size.csv"), index=False)
gt = a.gain_target or targ[0]; gc = targ.index(gt)
top = P[P.目标 == short(gt)].nlargest(12, "M2").字段.tolist(); ti = [cand.index(f) for f in top]
G = pipe.gain_matrix(V, keys, ti)[:, :, gc]
pd.DataFrame(G, index=top, columns=top).to_csv(os.path.join(AN, f"gain_matrix_{short(gt)}.csv"))

# ---------------- 图
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "axes.unicode_minus": False, "font.size": 9,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 170,
                     "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TAG = spec["dataset"]

# 图1：字段风险画像
fig, axs = plt.subplots(1, C, figsize=(6.2 * C, 7.2))
axs = np.atleast_1d(axs)
for ax, y in zip(axs, targ):
    d = P[P.目标 == short(y)].nlargest(18, "M2").sort_values("M2").reset_index(drop=True); ys = np.arange(len(d))
    for k, r in d.iterrows():
        ax.plot([r.M0, r.M2], [k, k], color="#b9b8b2", lw=2, zorder=1)
    ax.scatter(d.r2, ys, marker="D", s=36, color=AQUA, zorder=3, label="r²：单字段线性相关", edgecolor="#fcfcfb", lw=1.2)
    ax.scatter(d.M0, ys, s=52, facecolor="#fcfcfb", edgecolor=ORANGE, lw=2, zorder=4, label="M(0)：单字段推断能力")
    ax.scatter(d.M2, ys, s=52, color=BLUE, zorder=5, label="M^(2)：至多 2 个背景下的最坏增益", edgecolor="#fcfcfb", lw=1.2)
    for k, r in d.iterrows():
        if r.M2 - r.M0 >= 0.10:
            ax.text(r.M2 + .012, k, f"+{r.M2 - r.M0:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(d.字段, fontsize=8); ax.set_xlim(0, 1.08)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_title(short(y), loc="left", fontsize=10.2, color=INK); ax.set_xlabel("推断能力（测试集 R²）")
axs[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle(f"图1  {TAG}：字段风险画像（M^(2) 前 18 个字段；右侧数字为背景带来的升级）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(os.path.join(FG, "fig1_field_profile.png"), bbox_inches="tight"); plt.close(fig)

# 图2：增益矩阵
n = len(G); fig, ax = plt.subplots(figsize=(10, 7.6))
im = ax.imshow(np.clip(G, 0, None), cmap=SEQ, vmin=0, vmax=max(0.6, float(np.nanmax(G))), aspect="auto")
for r in range(n):
    for cc in range(n):
        v = G[r, cc]
        if r == cc:
            ax.add_patch(plt.Rectangle((cc - .5, r - .5), 1, 1, fill=False, edgecolor=INK2, lw=1.2))
        if v >= 0.10 or r == cc:
            ax.text(cc, r, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 0.45 else INK)
ax.set_xticks(range(n)); ax.set_xticklabels(top, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(top, fontsize=8)
ax.set_xlabel("攻击者已掌握的背景字段 j"); ax.set_ylabel("拟公开字段 i")
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02); cb.set_label("增益 V({i,j})−V({j})；对角线为单字段 V({i})", color=INK2)
ax.set_title(f"图2  {TAG}｜{short(gt)}：同一字段在不同背景下的推断增益（M^(2) 前 12 个字段）\n只标注 ≥0.10 的格子；对角框为单字段能力",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(FG, "fig2_gain_matrix.png"), bbox_inches="tight"); plt.close(fig)

# 图3：K 递进与攻击器族
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
Ks = list(range(KB + 1)); cols = [BLUE, ORANGE, AQUA]
for c, y in enumerate(targ):
    axs[0].plot(Ks, [S.loc[c, f"M{K}均值"] for K in Ks], "o-", color=cols[c % 3], lw=2, ms=6, label=short(y))
axs[0].set_xticks(Ks); axs[0].set_xticklabels([f"M^({K})" for K in Ks]); axs[0].set_ylabel("平均 M（全部候选字段）")
axs[0].grid(axis="y", color=GRID); axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title("A. K 递进：多数升级发生在第一个背景字段", loc="left", fontsize=10, color=INK)
axs[1].bar(names, share, color=[BLUE, ORANGE, AQUA][:len(names)], width=.55)
for i_, v in enumerate(share):
    axs[1].text(i_, v + .01, f"{v:.1%}", ha="center", fontsize=9, color=INK2)
axs[1].set_ylabel("在正式真值中被 val 选中的比例"); axs[1].grid(axis="y", color=GRID); axs[1].set_axisbelow(True)
axs[1].set_title("B. 攻击器族：三种攻击器都有贡献", loc="left", fontsize=10, color=INK)
fig.suptitle(f"图3  {TAG}：背景预算递进与攻击器族构成", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig3_escalation_attackers.png"), bbox_inches="tight"); plt.close(fig)

# 图4：通用模型保真度
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
rs = np.random.RandomState(0); sel = rs.choice(len(V), min(3000, len(V)), replace=False)
for c, y in enumerate(targ):
    axs[0].scatter(V[sel, c], Ve[sel, c], s=3, alpha=.35, color=cols[c % 3], label=short(y), rasterized=True)
axs[0].plot([0, 1], [0, 1], "--", color=INK, lw=.8); axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1)
axs[0].set_xlabel("专用重训真值 V(S)"); axs[0].set_ylabel("通用模型估计"); axs[0].legend(frameon=False, fontsize=7.5, markerscale=3)
axs[0].set_title("A. 集合推断能力", loc="left", fontsize=10, color=INK)
for c, y in enumerate(targ):
    axs[1].scatter(M[:, 2, c], Me_all[:, 2, c], s=22, alpha=.7, color=cols[c % 3], label=short(y))
axs[1].plot([0, 1], [0, 1], "--", color=INK, lw=.8); axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1)
axs[1].set_xlabel("精确 M^(2)"); axs[1].set_ylabel("估计 M^(2)"); axs[1].set_title("B. 字段风险 M^(2)", loc="left", fontsize=10, color=INK)
x = np.arange(C); w = .36
axs[2].bar(x - w / 2, S["认证下界比_top1"], w * .92, color="#9fc3ea", label="只认证估计器首选背景")
axs[2].bar(x + w / 2, S["认证下界比_top3"], w * .92, color=BLUE, label="认证估计器前 3 个背景")
for xi, v in zip(x + w / 2, S["认证下界比_top3"]):
    axs[2].text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
axs[2].set_xticks(x); axs[2].set_xticklabels([short(y) for y in targ], fontsize=8); axs[2].set_ylim(.7, 1.05)
axs[2].set_ylabel("认证下界 / 精确 M^(2)"); axs[2].legend(frameon=False, fontsize=8, loc="lower right")
axs[2].set_title("C. 扫描—认证：下界能达到精确值的多少", loc="left", fontsize=10, color=INK)
fig.suptitle(f"图4  {TAG}：通用推断模型的保真度（真值为三攻击器专用重训）", x=.01, ha="left", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(os.path.join(FG, "fig4_estimator_fidelity.png"), bbox_inches="tight"); plt.close(fig)

# ---------------- 报告
md = [f"# {TAG}：自动生成的数值汇总（结论与解读见 CHANGELOG.md）\n",
      "## 目标级汇总\n", S.round(4).to_markdown(index=False), "\n## 攻击器族构成\n", att.round(4).to_markdown(index=False),
      "\n## 按集合规模\n", by_size.round(4).to_markdown(index=False),
      f"\n## 通用模型查询耗时\n\n三种子 {float(E['per_query_s'])*1000:.2f} ms/集合；单种子 {float(E['per_query_s_1seed'])*1000:.2f} ms/集合；"
      f"预训练 {E['train_sec'].round(0).tolist()} 秒/种子\n",
      "\n## 字段风险表（按目标、M^(2) 降序）\n", P.sort_values(["目标", "M2"], ascending=[True, False]).round(3).to_markdown(index=False)]
open(os.path.join(AN, "report.md"), "w", encoding="utf-8").write("\n".join(md))
print(S.round(3).T.to_string()); print(att.to_string(index=False))
