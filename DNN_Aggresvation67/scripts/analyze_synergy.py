"""DNN67 分析：K 扫描 × 尺寸校准，对齐重训真值，验证协同排序 + 定 K。

真值：990 个重训（44 单 + 946 对），逐 conf v_c。
估计：63 oracle 暖启动微调 K∈{0,10,50,100,200,500} 的逐 conf v̂_c + 耗时。
校准：2 折交叉拟合 per-(size,K) 标量偏差（注意：尺寸校准只平移绝对值，不改协同排序，
      故对 Spearman 无效——用于绝对误差那一栏）。
协同真值：syn_c(i,j) = v_c(ij) − max(v_c(i), v_c(j))，逐 (pair,conf)。
输出：outputs/synergy_ksweep.csv / .json / synergy_ksweep.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
KGRID = [0, 10, 50, 100, 200, 500]
CONF = None


def load_truth():
    """返回 {sid: {conf: v}} 与 fields。"""
    truth = {}; fields = {}
    for f in (ROOT / "outputs/retrain").glob("*_dnn_seed0.json"):
        d = json.loads(f.read_text()); sid = d["subset_id"]
        truth[sid] = d["per_conf_r2"]; fields[sid] = d["fields"]
    return truth, fields


def load_est():
    """返回 {sid: {K: {conf: v}}}, {sid:{K:time}}, size。"""
    est = {}; tim = {}; size = {}
    for f in (ROOT / "outputs/est").glob("*.json"):
        d = json.loads(f.read_text()); sid = d["subset_id"]; size[sid] = d["size"]
        est[sid] = {int(k): v["per_conf_r2"] for k, v in d["by_k"].items()}
        tim[sid] = {int(k): v["time"] for k, v in d["by_k"].items()}
    return est, tim, size


def main():
    global CONF
    truth, fields = load_truth()
    est, tim, size = load_est()
    common = sorted(set(truth) & set(est))
    singles = [s for s in common if size[s] == 1]
    pairs = [s for s in common if size[s] == 2]
    CONF = sorted(next(iter(truth.values())).keys())
    print(f"真值 {len(truth)}，估计 {len(est)}，共用 {len(common)}（单 {len(singles)}，对 {len(pairs)}）")

    # ---- 长表：每 (sid,K,conf) 的 truth/est ----
    rows = []
    for s in common:
        for K in KGRID:
            for c in CONF:
                rows.append({"sid": s, "size": size[s], "K": K, "conf": c,
                             "truth": truth[s][c], "est": est[s][K][c]})
    df = pd.DataFrame(rows)

    # ---- 2 折交叉拟合 per-(size,K) 标量偏差，得 est_cal ----
    rng = np.random.RandomState(0)
    fold = {s: rng.randint(2) for s in common}
    df["fold"] = df["sid"].map(fold)
    df["bias"] = 0.0
    for (sz, K), g in df.groupby(["size", "K"]):
        for fo in [0, 1]:
            fit = g[g.fold != fo]
            b = (fit["est"] - fit["truth"]).mean()
            df.loc[(df["size"] == sz) & (df["K"] == K) & (df["fold"] == fo), "bias"] = b
    df["est_cal"] = df["est"] - df["bias"]

    # ---- 绝对误差 vs K（raw / cal），分 size ----
    abs_rows = []
    for K in KGRID:
        for sz, tag in [(1, "single"), (2, "pair")]:
            g = df[(df.K == K) & (df["size"] == sz)]
            abs_rows.append({"K": K, "group": tag,
                             "mae_raw": (g.est - g.truth).abs().mean(),
                             "mae_cal": (g.est_cal - g.truth).abs().mean(),
                             "signed_raw": (g.est - g.truth).mean()})
    abs_df = pd.DataFrame(abs_rows)

    # ---- 协同排序验证 ----
    tmap = {(s, c): truth[s][c] for s in common for c in CONF}
    def emap(K, key):
        col = df[df.K == K].set_index(["sid", "conf"])[key]
        return {idx: v for idx, v in col.items()}

    def pair_idx(sid):
        _, a, b = sid.split("_")[0][1:], *sid.split("_")  # p{i}_{j}
        i = int(sid[1:].split("_")[0]); j = int(sid.split("_")[1]); return i, j

    syn_rows = []
    for K in KGRID:
        e_raw = emap(K, "est"); e_cal = emap(K, "est_cal")
        st, sr, sc = [], [], []
        for p in pairs:
            i, j = pair_idx(p); si, sj = f"s{i:02d}", f"s{j:02d}"
            if si not in truth or sj not in truth:
                continue
            for c in CONF:
                t = tmap[(p, c)] - max(tmap[(si, c)], tmap[(sj, c)])
                r = e_raw[(p, c)] - max(e_raw[(si, c)], e_raw[(sj, c)])
                cc = e_cal[(p, c)] - max(e_cal[(si, c)], e_cal[(sj, c)])
                st.append(t); sr.append(r); sc.append(cc)
        st, sr, sc = np.array(st), np.array(sr), np.array(sc)
        # top-k 重合：真值 top-K 协同 entry 有多少落在估计 top-K
        def topk_overlap(a, b, k):
            ta = set(np.argsort(-a)[:k]); tb = set(np.argsort(-b)[:k])
            return len(ta & tb) / k
        syn_rows.append({"K": K,
                         "spearman_raw": spearmanr(sr, st).statistic,
                         "spearman_cal": spearmanr(sc, st).statistic,
                         "top20_raw": topk_overlap(st, sr, 20),
                         "top50_raw": topk_overlap(st, sr, 50),
                         "mae_syn_raw": np.abs(sr - st).mean(),
                         "mae_syn_cal": np.abs(sc - st).mean(),
                         "time_per_subset": np.mean([tim[s][K] for s in common])})
    syn_df = pd.DataFrame(syn_rows)

    abs_df.to_csv(ROOT / "outputs/synergy_abs.csv", index=False)
    syn_df.to_csv(ROOT / "outputs/synergy_ksweep.csv", index=False)
    print("\n=== 绝对误差 vs K ===")
    print(abs_df.round(4).to_string(index=False))
    print("\n=== 协同排序 & 耗时 vs K ===")
    print(syn_df.round(4).to_string(index=False))

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for sz, tag in [(1, "single"), (2, "pair")]:
        g = abs_df[abs_df.group == tag]
        ax[0].plot(g.K, g.mae_raw, "o-", label=f"{tag} raw")
        ax[0].plot(g.K, g.mae_cal, "s--", label=f"{tag} calibrated")
    ax[0].set_xscale("symlog"); ax[0].set_xlabel("fine-tune steps K")
    ax[0].set_ylabel("mean |v̂_c − truth|"); ax[0].set_title("(a) 绝对精度：校准修偏差")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(syn_df.K, syn_df.spearman_raw, "o-", label="Spearman (raw)")
    ax[1].plot(syn_df.K, syn_df.top20_raw, "^-", label="top-20 overlap")
    ax[1].plot(syn_df.K, syn_df.top50_raw, "v-", label="top-50 overlap")
    ax[1].set_xscale("symlog"); ax[1].set_xlabel("fine-tune steps K")
    ax[1].set_ylabel("协同排序吻合"); ax[1].set_title("(b) 协同排序 vs 重训真值")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 1)

    ax[2].plot(syn_df.K, syn_df.time_per_subset, "o-", c="tab:red")
    for _, r in syn_df.iterrows():
        ax[2].annotate(f"{r.time_per_subset:.2f}s", (r.K, r.time_per_subset), fontsize=7)
    ax[2].set_xscale("symlog"); ax[2].set_xlabel("fine-tune steps K")
    ax[2].set_ylabel("秒 / 子集"); ax[2].set_title("(c) 单子集微调耗时")
    ax[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(ROOT / "outputs/synergy_ksweep.png", dpi=150)
    print(f"\n图存 outputs/synergy_ksweep.png")


if __name__ == "__main__":
    main()
