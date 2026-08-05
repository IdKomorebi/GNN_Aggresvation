# -*- coding: utf-8 -*-
"""84 号 轴2：不确定性门控级联模拟（零 GPU，复用已存 K 网格 + 集成）。

问题：一次前向≈免费，K 步微调×爆炸空间=千万倍。能否只把 K 步路由到"需要"的少数候选？
本脚本比较四种 truth-free 路由信号，看哪种能用最小成本恢复"全量微调"的召回：

Tier0（免费）：ensemble-mean S1 @ K=0 排名。
路由信号（决定谁下沉 Tier1 做 K=25 微调）：
  sigma   —— seed0/1/2 的 S1 标准差（认知不确定性）
  boundary—— 到 Top-30% 判定边界的排名距离（最可能翻盘的）
  s1s2gap —— |rank_S1 − rank_S2|（82 号救援信号：S2 上界筛与 S1 分歧大处）
  random  —— 同预算随机（对照）
Tier1：routed 候选用 79 号 K=25 的 S1 替换（同 seed0 轨迹）；未 routed 保持 K=0。

成本 = routed_fraction × 25 步/候选（相对"全量 K=25"= 25 步×全部）。
对每个预算 b∈{0,5,...,60}% 报 Top-30% 加权召回(整体 & 强档>0.20)，画成本-召回帕累托。

诚实关注点：认知 σ 若在强档偏低（三 seed 同样系统性低估），σ 路由会漏最强档——
这正是要如实检验的（系统偏差 vs 方差）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R79 = REPO / "DNN_Aggresvation79"
KSTAR = "25"


def load() -> pd.DataFrame:
    seeds = []
    for s in range(3):
        d = pd.read_parquet(ROOT / f"outputs/eval_uniform_seed{s}.parquet")
        d["indices"] = d["indices"].apply(lambda t: tuple(int(v) for v in t))
        seeds.append(d.set_index("indices")[["S1", "S2"]].rename(
            columns={"S1": f"S1_{s}", "S2": f"S2_{s}"}))
    df = seeds[0].join(seeds[1]).join(seeds[2]).reset_index()
    meta = pd.read_parquet(ROOT / "outputs/eval_uniform_seed0.parquet")
    meta["indices"] = meta["indices"].apply(lambda t: tuple(int(v) for v in t))
    df = df.merge(meta[["indices", "syn3_true", "weight", "strong", "split"]], on="indices")
    df["S1_mean"] = df[["S1_0", "S1_1", "S1_2"]].mean(axis=1)
    df["S2_mean"] = df[["S2_0", "S2_1", "S2_2"]].mean(axis=1)
    df["sigma"] = df[["S1_0", "S1_1", "S1_2"]].std(axis=1)
    # K=25 微调后的 S1（seed0 轨迹，79 号）
    k = pd.read_csv(R79 / "outputs/triple_score_kgrid.csv")
    k["indices"] = k["indices"].apply(lambda s: tuple(ast.literal_eval(s)))
    df = df.merge(k[["indices", "0", KSTAR]].rename(columns={"0": "S1_k0", KSTAR: "S1_k25"}), on="indices")
    return df


def wrecall(df: pd.DataFrame, score: np.ndarray, lo: float, hi: float, frac=0.30) -> float:
    n_top = int(round(len(df) * frac))
    order = np.argsort(-score)
    top = set(df.iloc[order[:n_top]]["indices"])
    band = df[(df.syn3_true > lo) & (df.syn3_true <= hi)]
    if band.weight.sum() == 0:
        return float("nan")
    return float(band[band["indices"].isin(top)].weight.sum() / band.weight.sum())


def routed_score(df: pd.DataFrame, signal: str, budget: float, rng) -> np.ndarray:
    """返回路由后每候选的 S1：被选中的用 K=25，否则 K=0。"""
    n_route = int(round(len(df) * budget))
    base_rank = df["S1_mean"].rank(ascending=False)
    cutoff = int(round(len(df) * 0.30))
    if signal == "sigma":
        pri = df["sigma"].values
    elif signal == "boundary":
        pri = -np.abs(base_rank.values - cutoff)      # 离边界越近优先级越高
    elif signal == "s1s2gap":
        pri = np.abs(df["S1_mean"].rank(ascending=False).values
                     - df["S2_mean"].rank(ascending=False).values)
    elif signal == "random":
        pri = rng.random(len(df))
    else:
        raise ValueError(signal)
    routed = set(df.iloc[np.argsort(-pri)[:n_route]]["indices"])
    s = df["S1_k0"].values.copy()
    mask = df["indices"].isin(routed).values
    s[mask] = df["S1_k25"].values[mask]
    return s


def main() -> None:
    df = load()
    rng = np.random.RandomState(0)
    budgets = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
    signals = ["sigma", "boundary", "s1s2gap", "random"]
    rows = []
    # 参照：全量 K=25（预算100%）与纯 K=0（预算0）
    for tag, score in [("all_k25", df["S1_k25"].values), ("all_k0", df["S1_k0"].values)]:
        rows.append(dict(signal=tag, budget=1.0 if tag == "all_k25" else 0.0,
                         cost_steps=(25.0 if tag == "all_k25" else 0.0),
                         rec_all=wrecall(df, score, 0.10, 1.01),
                         rec_strong=wrecall(df, score, 0.20, 1.01)))
    for sig in signals:
        for b in budgets:
            score = routed_score(df, sig, b, rng)
            rows.append(dict(signal=sig, budget=b, cost_steps=25.0 * b,
                             rec_all=wrecall(df, score, 0.10, 1.01),
                             rec_strong=wrecall(df, score, 0.20, 1.01)))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/cascade_frontier.csv", index=False)
    # σ 在各强度档的分布（检验"系统偏差 vs 方差"）
    diag = []
    for lo, hi, nm in [(0.10, 0.12, "0.10-0.12"), (0.12, 0.15, "0.12-0.15"),
                       (0.15, 0.20, "0.15-0.20"), (0.20, 1.01, ">0.20")]:
        b = df[(df.syn3_true > lo) & (df.syn3_true <= hi)]
        diag.append(dict(band=nm, n=len(b), sigma_med=float(b.sigma.median()),
                         s1k0_med=float(b.S1_k0.median()), s1k25_med=float(b.S1_k25.median())))
    allsig = df.sigma.median()
    dg = pd.DataFrame(diag)
    dg.to_csv(ROOT / "outputs/sigma_by_band.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:.3f}")
    print("=== 级联前沿(成本=平均步数/候选) ===")
    print(out.to_string(index=False))
    print(f"\n=== σ 分档(全体中位 σ={allsig:.3f}) —— 检验σ能否标记强档 ===")
    print(dg.to_string(index=False))


if __name__ == "__main__":
    main()
