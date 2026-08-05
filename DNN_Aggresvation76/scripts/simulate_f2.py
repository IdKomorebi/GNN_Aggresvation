# -*- coding: utf-8 -*-
"""F-2：自适应早停的**离线模拟**（零 GPU，直接用 F-1 的完整 K 曲线）。

动机
----
F-1 给出的 K* 是一个**全局**步数；但不同候选的收敛速度差异很大：
真无协同的候选往往几步就稳定在 0 附近（早该弃），而真强协同要等到相变才显形。
逐查询自适应早停能在同样的召回下省下大量步数——扫 13244 个候选时这是实打实的。

停机规则（三条，**全部只用 oracle 自身输出，零真值**）
------------------------------------------------------
R1 收敛停：|ŝyn(K) − ŝyn(K−5)| < eps 连续 patience 次 → 停
R2 早弃  ：ŝyn(K) < delta 且相对上一档没有上升 → 判负例，停
R3 上限  ：K 达到 kmax 强制停

评估
----
在（平均步数, 真强协同召回率）平面上画帕累托前沿，对照固定 K 基线。
真值只在**评估**时使用（红线允许），规则本身不碰真值。
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R76 = Path(__file__).resolve().parents[1]
R75 = R76.parent / "DNN_Aggresvation75"
R68 = R76.parent / "DNN_Aggresvation68"
sys.path.insert(0, str(R76 / "src"))
from runlog import log  # noqa: E402

OUT = R76 / "outputs"
STRONG, DETECT = 0.20, 0.10


def load_syn_curves(scheme="uniform", seeds=(0, 1, 2)):
    """返回 (真值数组 t, 曲线矩阵 E[K_idx, n_entry], K 列表)。"""
    ev = json.load(open(R75 / "outputs/evalset.json"))
    s1 = {ev[s]["fields"][0]: s for s in ev if ev[s]["size"] == 1}
    s2 = {tuple(sorted(ev[s]["fields"])): s for s in ev if ev[s]["size"] == 2}
    syn2 = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    syn2["sid_ij"] = [s2.get(tuple(sorted((a, b)))) for a, b in zip(syn2.fi, syn2.fj)]
    syn2["sid_i"] = syn2.fi.map(s1)
    syn2["sid_j"] = syn2.fj.map(s1)
    syn2 = syn2.dropna(subset=["sid_ij", "sid_i", "sid_j"]).reset_index(drop=True)

    mats = []
    for sd in seeds:
        p = OUT / f"fine_{scheme}_seed{sd}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        Ks = sorted(d.K.unique())
        cur = np.full((len(Ks), len(syn2)), np.nan)
        for a, K in enumerate(Ks):
            m = d[d.K == K].set_index(["sid", "conf"]).est.to_dict()
            eij = np.array([m.get((s, c), np.nan) for s, c in zip(syn2.sid_ij, syn2.conf)])
            ei = np.array([m.get((s, c), np.nan) for s, c in zip(syn2.sid_i, syn2.conf)])
            ej = np.array([m.get((s, c), np.nan) for s, c in zip(syn2.sid_j, syn2.conf)])
            cur[a] = eij - np.maximum(ei, ej)
        mats.append(cur)
    return syn2.synergy.values, mats, Ks


def simulate(E, Ks, eps, delta, patience):
    """返回 (每条目的停机步数, 停机时的 ŝyn 值)。E: (nK, n)"""
    nK, n = E.shape
    stop_at = np.full(n, Ks[-1], dtype=float)
    val = E[-1].copy()
    stable = np.zeros(n, dtype=int)
    active = np.ones(n, dtype=bool)
    for a in range(1, nK):
        if not active.any():
            break
        d = np.abs(E[a] - E[a - 1])
        rise = E[a] - E[a - 1]
        # R1 收敛
        stable = np.where(d < eps, stable + 1, 0)
        r1 = active & (stable >= patience)
        # R2 早弃：低且没在涨
        r2 = active & (E[a] < delta) & (rise <= 0)
        hit = r1 | r2
        stop_at[hit] = Ks[a]
        val[hit] = E[a][hit]
        active &= ~hit
    return stop_at, val


def main():
    t, mats, Ks = load_syn_curves()
    if not mats:
        raise SystemExit("还没有 F-1 结果")
    strong = t > STRONG

    rows = []
    # ---- 固定 K 基线 ----
    for a, K in enumerate(Ks):
        rec, fp, steps = [], [], []
        for E in mats:
            rec.append((E[a][strong] > DETECT).mean())
            fp.append((E[a][t <= 0.02] > DETECT).mean())
            steps.append(K)
        rows.append(dict(rule="fixed", eps=np.nan, delta=np.nan, patience=np.nan,
                         K=K, mean_steps=float(np.mean(steps)),
                         recall=float(np.mean(rec)), fp=float(np.mean(fp))))

    # ---- 自适应 ----
    for eps, delta, pat in itertools.product([0.002, 0.005, 0.01],
                                             [0.02, 0.05, 0.08],
                                             [1, 2]):
        rec, fp, steps = [], [], []
        for E in mats:
            sa, v = simulate(E, Ks, eps, delta, pat)
            rec.append((v[strong] > DETECT).mean())
            fp.append((v[t <= 0.02] > DETECT).mean())
            steps.append(sa.mean())
        rows.append(dict(rule="adaptive", eps=eps, delta=delta, patience=pat, K=np.nan,
                         mean_steps=float(np.mean(steps)),
                         recall=float(np.mean(rec)), fp=float(np.mean(fp))))

    df = pd.DataFrame(rows).sort_values("mean_steps")
    df.to_csv(OUT / "f2_earlystop.csv", index=False)

    print("=" * 84)
    print("F-2 自适应早停 vs 固定 K（uniform，3 seed 均值；真强协同 143 条）")
    print("=" * 84)
    print(f"{'规则':<10}{'eps':>7}{'delta':>7}{'pat':>5}{'K':>5}{'平均步数':>10}"
          f"{'强协同召回':>11}{'误报率':>9}")
    base50 = df[(df.rule == "fixed") & (df.K == 50)].iloc[0]
    for r in df.itertuples():
        if r.rule == "fixed" and r.K % 10 and r.K != 5:
            continue
        print(f"{r.rule:<10}{r.eps if not np.isnan(r.eps) else 0:>7.3f}"
              f"{r.delta if not np.isnan(r.delta) else 0:>7.3f}"
              f"{r.patience if not np.isnan(r.patience) else 0:>5.0f}"
              f"{r.K if not np.isnan(r.K) else 0:>5.0f}"
              f"{r.mean_steps:>10.1f}{r.recall:>11.1%}{r.fp:>9.2%}")

    # ---- 帕累托：召回不低于固定 K=50 且步数最少 ----
    cand = df[(df.rule == "adaptive") & (df.recall >= base50.recall - 1e-9)]
    print("\n" + "=" * 84)
    print(f"固定 K=50 基线：平均步数 {base50.mean_steps:.0f}，"
          f"召回 {base50.recall:.1%}，误报 {base50.fp:.2%}")
    if len(cand):
        b = cand.sort_values("mean_steps").iloc[0]
        save = 1 - b.mean_steps / base50.mean_steps
        msg = (f"推荐规则 eps={b.eps}, delta={b.delta}, patience={int(b.patience)}："
               f"平均步数 {b.mean_steps:.1f}（省 {save:.0%}），"
               f"召回 {b.recall:.1%}（不低于基线），误报 {b.fp:.2%}")
    else:
        b = df[df.rule == "adaptive"].sort_values("recall", ascending=False).iloc[0]
        msg = (f"没有自适应规则能在不损召回的前提下省步数；最接近的是 "
               f"eps={b.eps}, delta={b.delta}, pat={int(b.patience)} → "
               f"步数 {b.mean_steps:.1f}、召回 {b.recall:.1%}（基线 {base50.recall:.1%}）")
    print(msg)
    log("F-2", "DECISION", note=msg)
    (OUT / "f2_verdict.txt").write_text(msg, encoding="utf-8")


if __name__ == "__main__":
    main()
