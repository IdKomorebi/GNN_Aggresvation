"""DNN64 汇总：微调收敛曲线，回答"暖启动聪不聪明"。

真值 = 60 号 v_best（best-of-struct 重训）。对每个子集每条臂，误差(K) = v̂_arm(K) − truth。
关注：
  1) 偏差闭合曲线：mean|err| vs K（暖 vs 冷 vs 高lr暖 vs 只调头），分尺寸带；
  2) 等步/等时对比：暖启动在小 K 是否显著领先冷启动；
  3) 各臂 K=500 的收敛天花板（是否暖启动最终也更高）。
判据（预注册）：暖启动 K≤10 步把 63 号 −0.08 系统低估压进噪声底(mean|err|<0.02)且快于冷启动。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "outputs/ft"
ARMS = ["cold", "warm_full", "warm_full_hi", "warm_head"]
KGRID = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
BANDS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]


def band_of(s):
    for lo, hi in BANDS:
        if lo <= s <= hi:
            return f"{lo}-{hi}"
    return "?"


def main():
    rv = pd.read_csv(ROOT.parent / "DNN_Aggresvation60/outputs/random_eval.csv").set_index("subset_id")
    recs = []
    for f in sorted(FT.glob("*.json")):
        j = json.loads(f.read_text())
        sid = j["subset_id"]; truth = float(rv.loc[sid, "v_best"])
        for arm in ARMS:
            r2 = j["arms"][arm]["r2"]; tm = j["arms"][arm]["time"]
            for K in KGRID:
                recs.append({"subset": sid, "size": j["size"], "band": band_of(j["size"]),
                             "seed": j["seed"], "arm": arm, "K": K,
                             "r2": r2[str(K)], "time": tm.get(str(K), 0.0),
                             "truth": truth, "err": r2[str(K)] - truth})
    df = pd.DataFrame(recs)
    df.to_csv(ROOT / "outputs/ft_long.csv", index=False)

    # 先对 seed 平均（同一子集同一臂同一 K）
    g = df.groupby(["subset", "band", "size", "arm", "K"], as_index=False).agg(
        r2=("r2", "mean"), err=("err", "mean"), truth=("truth", "mean"), time=("time", "mean"))

    # 偏差闭合：mean|err| vs K（全体 + 分带）
    def mabs_by_K(sub):
        return sub.groupby(["arm", "K"])["err"].apply(lambda e: np.abs(e).mean()).unstack("K")

    overall = mabs_by_K(g)
    print("=== mean|v̂(K) − truth| vs K（全体 50 子集）===")
    print(overall.round(4).to_string())
    print("\n（对照：63 号 oracle K0 系统低估≈0.08；噪声底≈0.004；判据线 0.02）")

    # warm_full 相对 cold 的等步领先
    print("\n=== warm_full − cold 的 mean|err| 差（负=暖启动更好）===")
    diff = (overall.loc["warm_full"] - overall.loc["cold"])
    print(diff.round(4).to_string())

    # 分带在关键 K 的表
    print("\n=== 各尺寸带 mean|err|（K=0/5/10/50/500）warm_full vs cold ===")
    for b, _ in [(f"{lo}-{hi}", 0) for lo, hi in BANDS]:
        sub = g[g["band"] == b]
        if sub.empty:
            continue
        w = mabs_by_K(sub).loc["warm_full"]; c = mabs_by_K(sub).loc["cold"]
        print(f"  [{b}] warm: " + " ".join(f"K{k}={w[k]:.3f}" for k in [0, 5, 10, 50, 500]))
        print(f"  [{b}] cold: " + " ".join(f"K{k}={c[k]:.3f}" for k in [0, 5, 10, 50, 500]))

    summary = {
        "overall_mabs_by_K": {a: {int(k): float(overall.loc[a, k]) for k in KGRID} for a in ARMS},
        "criterion_warm_full": {
            "K10_mean_abs_err": float(overall.loc["warm_full", 10]),
            "passes_within_0.02_by_K10": bool(overall.loc["warm_full", 10] < 0.02),
            "warm_beats_cold_at_K10": bool(overall.loc["warm_full", 10] < overall.loc["cold", 10]),
            "warm_K10_vs_warm_K0": [float(overall.loc["warm_full", 0]), float(overall.loc["warm_full", 10])],
        },
        "ceiling_K500": {a: float(overall.loc[a, 500]) for a in ARMS},
    }
    (ROOT / "outputs/ft_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    Kx = np.array(KGRID)
    for arm in ARMS:
        axes[0].plot(Kx, [overall.loc[arm, k] for k in KGRID], "o-", label=arm)
    axes[0].axhline(0.02, color="gray", ls="--", lw=1, label="判据线 0.02")
    axes[0].set_xscale("symlog"); axes[0].set_xlabel("fine-tune steps K")
    axes[0].set_ylabel("mean |v̂(K) − truth|"); axes[0].set_title("偏差闭合 vs K（全体）")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    # vs 墙钟
    for arm in ["cold", "warm_full", "warm_full_hi"]:
        t = g[g["arm"] == arm].groupby("K")["time"].mean().reindex(KGRID).values
        e = [overall.loc[arm, k] for k in KGRID]
        axes[1].plot(t, e, "o-", label=arm)
    axes[1].set_xlabel("wall-clock time (s)"); axes[1].set_ylabel("mean |v̂ − truth|")
    axes[1].set_title("偏差闭合 vs 耗时"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    # 分带 warm_full 在 K=10 的 |err|
    xb = [f"{lo}-{hi}" for lo, hi in BANDS]
    xm = np.arange(len(xb))
    w10 = [np.abs(g[(g.band == b) & (g.arm == "warm_full") & (g.K == 10)]["err"]).mean() for b in xb]
    c10 = [np.abs(g[(g.band == b) & (g.arm == "cold") & (g.K == 10)]["err"]).mean() for b in xb]
    axes[2].bar(xm - 0.2, w10, 0.4, label="warm_full K10")
    axes[2].bar(xm + 0.2, c10, 0.4, label="cold K10")
    axes[2].set_xticks(xm); axes[2].set_xticklabels(xb)
    axes[2].set_xlabel("size band"); axes[2].set_ylabel("mean |err| @K=10")
    axes[2].set_title("K=10 时的偏差（分带）"); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "outputs/ft_plots.png", dpi=150)
    print(f"\n完成，结果在 {ROOT}/outputs/")


if __name__ == "__main__":
    main()
