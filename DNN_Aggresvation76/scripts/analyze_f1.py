# -*- coding: utf-8 -*-
"""F-1 分析：定位二阶协同的"发现相变" K*。

指标（对 68 号 synergy2_perconf.csv 的重训真值，946 对 × 12 conf = 11352 条）：
  rho          ŝyn 与 syn_true 的 Spearman
  top20        真值 top-20 落入估计 top-20 的比例
  med_rank10   **真值 top-10 条目的估计排名中位数** ← 相变指纹（越小越好）
  recall       真强协同（syn_true>0.2，143 条）中被估计 >0.1 的比例
  fp           真无协同（syn_true<=0.02）中被估计 >0.1 的比例

K* 定义（预注册）：med_rank10 首次 ≤ 50 的 K。
判据：若 uniform 的 K* 显著小于 none 的，"少步数发现"归属随机失活起点；
      若两者相同，则该主张须削弱为"仅省算力、不改相变位置"。两种结果都如实写。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R76 = Path(__file__).resolve().parents[1]
R75 = R76.parent / "DNN_Aggresvation75"
R68 = R76.parent / "DNN_Aggresvation68"
sys.path.insert(0, str(R76 / "src"))
from runlog import log  # noqa: E402

OUT = R76 / "outputs"
STRONG, NULL, DETECT = 0.20, 0.02, 0.10
RANK_GATE = 50          # K* 判据


def build_index():
    ev = json.load(open(R75 / "outputs/evalset.json"))
    s1 = {ev[s]["fields"][0]: s for s in ev if ev[s]["size"] == 1}
    s2 = {tuple(sorted(ev[s]["fields"])): s for s in ev if ev[s]["size"] == 2}
    syn2 = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    syn2["sid_ij"] = [s2.get(tuple(sorted((a, b)))) for a, b in zip(syn2.fi, syn2.fj)]
    syn2["sid_i"] = syn2.fi.map(s1)
    syn2["sid_j"] = syn2.fj.map(s1)
    return syn2.dropna(subset=["sid_ij", "sid_i", "sid_j"]).reset_index(drop=True)


def metrics(syn2, est_map):
    """est_map: (sid, conf) -> v̂。返回该 K 的一组指标。"""
    eij = np.array([est_map.get((s, c), np.nan) for s, c in zip(syn2.sid_ij, syn2.conf)])
    ei = np.array([est_map.get((s, c), np.nan) for s, c in zip(syn2.sid_i, syn2.conf)])
    ej = np.array([est_map.get((s, c), np.nan) for s, c in zip(syn2.sid_j, syn2.conf)])
    e = eij - np.maximum(ei, ej)
    t = syn2.synergy.values
    ok = ~np.isnan(e)
    e, t = e[ok], t[ok]
    order_t = np.argsort(-t)
    rank_e = pd.Series(-e).rank().values
    top20_t, top20_e = set(order_t[:20]), set(np.argsort(-e)[:20])
    strong, null = t > STRONG, t <= NULL
    return dict(n=len(t), rho=spearmanr(t, e).correlation,
                top20=len(top20_t & top20_e) / 20,
                med_rank10=float(np.median(rank_e[order_t[:10]])),
                recall=float((e[strong] > DETECT).mean()),
                fp=float((e[null] > DETECT).mean()))


def main():
    syn2 = build_index()
    files = sorted(OUT.glob("fine_*_seed*.csv"))
    if not files:
        raise SystemExit("还没有 F-1 结果")

    rows = []
    for p in files:
        stem = p.stem.replace("fine_", "")
        scheme, seed = stem.rsplit("_seed", 1)
        d = pd.read_csv(p)
        for K, g in d.groupby("K"):
            m = metrics(syn2, g.set_index(["sid", "conf"]).est.to_dict())
            rows.append(dict(scheme=scheme, seed=int(seed), K=int(K), **m))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "f1_by_seed.csv", index=False)

    agg = df.groupby(["scheme", "K"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    sd = df.groupby(["scheme", "K"]).std(numeric_only=True).reset_index()
    agg.to_csv(OUT / "f1_curves.csv", index=False)

    # ---- 一致性核查：K=0 应复现 75_Correction 的数字 ----
    u0 = agg[(agg.scheme == "uniform") & (agg.K == 0)]
    if len(u0):
        r = float(u0.rho.iloc[0])
        okmark = "OK" if abs(r - 0.6817) <= 0.02 else "偏离"
        print(f"[一致性核查] uniform K=0 协同 ρ = {r:.4f}（75_Correction 为 0.6817）→ {okmark}\n")

    print("=" * 100)
    print("F-1 相变曲线（3 seed 均值；med_rank10 = 真 top10 的估计排名中位数，共 11352 条）")
    print("=" * 100)
    for sc in ["uniform", "bern50", "none"]:
        a = agg[agg.scheme == sc]
        if not len(a):
            continue
        print(f"\n--- {sc} ---")
        print(f"{'K':>4}{'协同ρ':>9}{'top20':>8}{'med_rank10':>12}{'强协同检出':>11}{'误报率':>9}")
        for r in a.itertuples():
            print(f"{r.K:>4}{r.rho:>9.4f}{r.top20:>8.2f}{r.med_rank10:>12.0f}"
                  f"{r.recall:>11.1%}{r.fp:>9.2%}")

    # ---- K*：med_rank10 首次 <= RANK_GATE ----
    print("\n" + "=" * 100)
    print(f"K*（预注册定义：真 top10 的估计排名中位数首次 ≤ {RANK_GATE}）")
    print("=" * 100)
    ks = {}
    for sc in agg.scheme.unique():
        a = agg[agg.scheme == sc].sort_values("K")
        hit = a[a.med_rank10 <= RANK_GATE]
        ks[sc] = int(hit.K.iloc[0]) if len(hit) else None
        # 逐 seed 的 K*，看稳不稳
        per = []
        for sd_ in sorted(df.seed.unique()):
            b = df[(df.scheme == sc) & (df.seed == sd_)].sort_values("K")
            h = b[b.med_rank10 <= RANK_GATE]
            per.append(int(h.K.iloc[0]) if len(h) else None)
        print(f"  {sc:<10} K* = {ks[sc]}    逐 seed: {per}")
    json.dump(ks, open(OUT / "kstar.json", "w"))

    u, n = ks.get("uniform"), ks.get("none")
    if u is not None and n is not None:
        if u < n:
            verdict = (f"uniform 的 K*={u} < none 的 K*={n} → "
                       f"'少步数发现'确属随机失活起点，主张二成立")
        elif u == n:
            verdict = (f"uniform 与 none 的 K* 相同（均为 {u}）→ "
                       f"随机失活**不改变相变位置**，主张二须削弱为'仅省算力'")
        else:
            verdict = f"uniform 的 K*={u} > none 的 K*={n} → 与预期相反，需重新审视"
    else:
        verdict = f"至少一方在 K≤80 内未达门槛（uniform={u}, none={n}）"
    print(f"\n判定：{verdict}")
    log("F-1", "DECISION", note=verdict)
    (OUT / "f1_verdict.txt").write_text(verdict, encoding="utf-8")


if __name__ == "__main__":
    main()
