# -*- coding: utf-8 -*-
"""DNN73：对比各训练方案变体与逐集合重训真值的差距。

指标：逐(集合,机密字段)的 MAE / 偏差 / Spearman，按 K 与集合规模带分层。
真值：69 号 truth_long.csv 的 dnn 列（攻击者最优响应重训）。
"""
import itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R69 = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation69")
OUT = Path("/data1/duhaocun/projects/GNN_Aggresvation/DNN_Aggresvation73/outputs")
SCHEMES = [("none", "无随机失活（置零口径）"), ("bern50", "伯努利(0.5)失活"), ("ours", "两段式尺寸均匀（本文）")]

tl = pd.read_csv(R69 / "outputs/truth_long.csv")
truth = tl.set_index(["sid", "conf"])["dnn"].to_dict()

frames = []
for arch in ["mlp", "gnn"]:
    for sc, _ in SCHEMES:
        p = OUT / f"est_{arch}_{sc}_seed0.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d["arch"] = arch; d["scheme"] = sc
        d["truth"] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
        frames.append(d.dropna(subset=["truth"]))
if not frames:
    raise SystemExit("没有可用的评估结果")
df = pd.concat(frames, ignore_index=True)
df["err"] = df.est - df.truth
df.to_csv(OUT / "merged_estimates.csv", index=False)

def band(n):
    if n <= 2: return "1-2"
    if n <= 4: return "3-4"
    if n <= 8: return "5-8"
    if n <= 16: return "9-16"
    if n <= 32: return "17-32"
    return "33-44"
df["band"] = df["size"].map(band)


def agg(g):
    return pd.Series({
        "n": len(g),
        "mae": g.err.abs().mean(),
        "bias": g.err.mean(),
        "spearman": spearmanr(g.est, g.truth).correlation,
    })


# ---- 总体（按 arch × scheme × K）----
overall = df.groupby(["arch", "scheme", "K"]).apply(agg, include_groups=False).reset_index()
overall.to_csv(OUT / "metrics_overall.csv", index=False)
print("=== 总体：各训练方案 vs 重训真值 ===")
for arch in overall.arch.unique():
    print(f"\n--- {arch} ---")
    piv = overall[overall.arch == arch].pivot(index="scheme", columns="K", values="mae")
    piv = piv.reindex([s for s, _ in SCHEMES])
    print("MAE:"); print(piv.round(4).to_string())
    piv2 = overall[overall.arch == arch].pivot(index="scheme", columns="K", values="spearman")
    piv2 = piv2.reindex([s for s, _ in SCHEMES])
    print("Spearman:"); print(piv2.round(4).to_string())

# ---- 按规模带（K=0 与 K=200）----
bysize = df[df.K.isin([0, 200])].groupby(
    ["arch", "scheme", "K", "band"]).apply(agg, include_groups=False).reset_index()
bysize.to_csv(OUT / "metrics_by_size.csv", index=False)
ORDER = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-44"]
print("\n\n=== 按集合规模带的 MAE ===")
for arch in bysize.arch.unique():
    for K in [0, 200]:
        sub = bysize[(bysize.arch == arch) & (bysize.K == K)]
        piv = sub.pivot(index="scheme", columns="band", values="mae").reindex(
            [s for s, _ in SCHEMES])[[b for b in ORDER if b in sub.band.unique()]]
        print(f"\n--- {arch}  K={K} ---")
        print(piv.round(4).to_string())

# ---- 相对提升（ours vs none）----
print("\n\n=== ours 相对 none 的 MAE 降幅 ===")
rows_gain = []
for arch in overall.arch.unique():
    for K in [0, 10, 50, 200]:
        o = overall[(overall.arch == arch) & (overall.scheme == "ours") & (overall.K == K)]
        n = overall[(overall.arch == arch) & (overall.scheme == "none") & (overall.K == K)]
        if len(o) and len(n):
            om, nm = float(o.mae.iloc[0]), float(n.mae.iloc[0])
            print(f"  {arch} K={K}: none={nm:.4f} ours={om:.4f} 降幅={1-om/nm:.1%}")
            rows_gain.append(dict(arch=arch, K=K, none_mae=nm, ours_mae=om, mae_reduction=1 - om / nm))
pd.DataFrame(rows_gain).to_csv(OUT / "gain_vs_none.csv", index=False)

# ---- 等效算力：none 需要多少微调步才能追上 ours@K=0 ----
# 在 (K, 指标) 曲线上对 none 做单调插值，求达到 ours@K=0 水平所需的 K。
print("\n\n=== 等效算力：无失活模型追平 ours@K=0 所需的微调步数 ===")
rows_eq = []
for arch in overall.arch.unique():
    for metric, better in [("mae", "lower"), ("spearman", "higher")]:
        o0 = overall[(overall.arch == arch) & (overall.scheme == "ours") & (overall.K == 0)]
        nn = overall[(overall.arch == arch) & (overall.scheme == "none")].sort_values("K")
        if not len(o0) or not len(nn):
            continue
        target = float(o0[metric].iloc[0])
        ks, vs = nn.K.values.astype(float), nn[metric].values.astype(float)
        keq = np.nan
        for a in range(len(ks) - 1):
            lo, hi = vs[a], vs[a + 1]
            hit = (lo >= target >= hi) if better == "lower" else (lo <= target <= hi)
            if hit and abs(hi - lo) > 1e-12:
                keq = ks[a] + (target - lo) / (hi - lo) * (ks[a + 1] - ks[a]); break
        if np.isnan(keq) and ((better == "lower" and vs[-1] > target) or
                              (better == "higher" and vs[-1] < target)):
            keq = float("inf")  # 200 步内都追不上
        rows_eq.append(dict(arch=arch, metric=metric, ours_K0=target, none_equiv_K=keq))
        s = "追不上(>200)" if keq == float("inf") else (f"K≈{keq:.0f}" if not np.isnan(keq) else "N/A")
        print(f"  {arch} {metric}: ours@K=0 = {target:.4f} → 无失活需 {s}")
pd.DataFrame(rows_eq).to_csv(OUT / "equivalent_compute.csv", index=False)
