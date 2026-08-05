# -*- coding: utf-8 -*-
"""DNN74：汇总各图架构变体的保真度，并按预注册判据给出结论。

主指标：K=0 时对 394 个分层子集重训真值的 MAE 与 Spearman（逐 conf 口径）。
对照基准（来自 73 号，直接读取不重跑）：
    MLP-oracle(ours)      MAE 0.0597 / Spearman 0.9698   ← 目标
    GNN-oracle(ours, 63号) MAE 0.0963 / Spearman 0.9197   ← 现状

预注册判据（跑之前写死在 CHANGELOG，跑完不许改）：
    翻案        MAE ≤ 0.0597 且 Spearman ≥ 0.9698
    有戏(70%)   MAE ≤ 0.0700 且 Spearman ≥ 0.9550
    放弃(<45%)  MAE > 0.0800

子命令：
    --check-protocol   口径一致性自检（评测子集/真值来源/base.yaml）
    --check-repro      V1 复现闸门：A0_repro 是否复现 63 号的数字
    --round A|B|final  汇总某一轮的结果并给判据
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

R74 = Path(__file__).resolve().parents[1]
R69 = R74.parent / "DNN_Aggresvation69"
R73 = R74.parent / "DNN_Aggresvation73"
OUT = R74 / "outputs"

# ---- 73 号实测的参照值（现状与目标）----
REF = {
    "mlp_ours":  dict(r2_full=0.8370, K0_mae=0.0597, K0_rho=0.9698, K200_mae=0.0163),
    "gnn_ours":  dict(r2_full=0.7593, K0_mae=0.0963, K0_rho=0.9197, K200_mae=0.0405),
}
# ---- 预注册判据 ----
GATE = dict(overturn_mae=0.0597, overturn_rho=0.9698,
            promising_mae=0.0700, promising_rho=0.9550,
            abandon_mae=0.0800)
# ---- V1 复现闸门容差 ----
REPRO_TOL = dict(r2_full=0.010, K0_mae=0.008, K0_rho=0.010, K200_mae=0.008)

BAND_ORDER = ["1-2", "3-4", "5-8", "9-16", "17-32", "33-44"]


def band(n):
    if n <= 2: return "1-2"
    if n <= 4: return "3-4"
    if n <= 8: return "5-8"
    if n <= 16: return "9-16"
    if n <= 32: return "17-32"
    return "33-44"


def load_truth():
    tl = pd.read_csv(R69 / "outputs/truth_long.csv")
    return tl.set_index(["sid", "conf"])["dnn"].to_dict()


def metrics_for(path: Path, truth: dict) -> pd.DataFrame:
    """读一份 est csv，返回按 K 聚合的 MAE/bias/Spearman（含分带）。"""
    d = pd.read_csv(path)
    d["truth"] = [truth.get((s, c), np.nan) for s, c in zip(d.sid, d.conf)]
    d = d.dropna(subset=["truth"])
    d["err"] = d.est - d.truth
    d["band"] = d["size"].map(band)
    return d


def agg(g):
    return pd.Series({"n": len(g), "mae": g.err.abs().mean(), "bias": g.err.mean(),
                      "spearman": spearmanr(g.est, g.truth).correlation})


def collect(variants: list[str], truth: dict) -> pd.DataFrame:
    """把所有 (variant, seed) 的评测汇总成一张长表。"""
    rows = []
    for v in variants:
        for p in sorted(OUT.glob(f"est_{v}_seed*.csv")):
            seed = int(p.stem.split("seed")[-1])
            ck = OUT / f"oracle_{v}_seed{seed}.pt"
            meta = torch.load(ck, map_location="cpu", weights_only=False) if ck.exists() else {}
            d = metrics_for(p, truth)
            byK = d.groupby("K").apply(agg, include_groups=False)
            byband = d[d.K == 0].groupby("band").apply(agg, include_groups=False)
            r = dict(variant=v, seed=seed,
                     n_params=meta.get("n_params", np.nan),
                     r2_full=meta.get("r2_full", np.nan),
                     epochs_run=meta.get("epochs_run", np.nan),
                     epochs_budget=meta.get("epochs_budget", np.nan))
            for K in [0, 10, 50, 200]:
                if K in byK.index:
                    r[f"K{K}_mae"] = byK.loc[K, "mae"]
                    r[f"K{K}_rho"] = byK.loc[K, "spearman"]
            for b in BAND_ORDER:
                if b in byband.index:
                    r[f"K0_mae_{b}"] = byband.loc[b, "mae"]
            rows.append(r)
    return pd.DataFrame(rows)


def verdict(mae: float, rho: float) -> str:
    if mae <= GATE["overturn_mae"] and rho >= GATE["overturn_rho"]:
        return "翻案"
    if mae <= GATE["promising_mae"] and rho >= GATE["promising_rho"]:
        return "有戏"
    if mae > GATE["abandon_mae"]:
        return "放弃"
    return "中间态"


def closure(mae: float) -> float:
    """相对 MLP 的差距缩掉了百分之多少（0 = 与现状 GNN 持平，1 = 追平 MLP）。"""
    lo, hi = REF["mlp_ours"]["K0_mae"], REF["gnn_ours"]["K0_mae"]
    return (hi - mae) / (hi - lo)


# ---------------------------------------------------------------- #
def cmd_check_protocol():
    ok = True
    print("=" * 74)
    print("V2 口径一致性自检")
    print("=" * 74)

    r = subprocess.run(["diff", str(R69 / "base.yaml"), str(R74 / "base.yaml")],
                       capture_output=True, text=True)
    print(f"[{'OK ' if r.returncode == 0 else 'FAIL'}] base.yaml 与 69 号一致")
    ok &= r.returncode == 0

    sys.path.insert(0, str(R74 / "scripts"))
    from eval_variant import pick_subsets
    reg = json.load(open(R69 / "outputs/subsets.json"))
    sids = set(pick_subsets(reg))
    print(f"[{'OK ' if len(sids) == 394 else 'FAIL'}] 评测子集数 = {len(sids)}（应为 394）")
    ok &= len(sids) == 394

    ref = R73 / "outputs/est_gnn_ours_seed0.csv"
    if ref.exists():
        same = set(pd.read_csv(ref).sid.unique()) == sids
        print(f"[{'OK ' if same else 'FAIL'}] 子集 sid 集合与 73 号 est_gnn_ours_seed0.csv 相同")
        ok &= same
    else:
        print("[WARN] 找不到 73 号参照文件，跳过 sid 集合比对")

    tl = R69 / "outputs/truth_long.csv"
    print(f"[{'OK ' if tl.exists() else 'FAIL'}] 真值来源 {tl.relative_to(R74.parent)}（dnn 列）")
    ok &= tl.exists()

    print("\n结论：" + ("口径一致，可以开跑。" if ok else "口径不一致，先修好再跑！"))
    return ok


def cmd_check_repro():
    truth = load_truth()
    df = collect(["A0_repro"], truth)
    if df.empty:
        raise SystemExit("还没有 A0_repro 的评测结果，先跑 train_variant + eval_variant")
    m = df.mean(numeric_only=True)
    print("=" * 74)
    print("V1 复现闸门：A0_repro（新代码）是否复现 63/69/73 号的 GNNOracle")
    print("=" * 74)
    print(f"（{len(df)} 个 seed 的均值）\n")
    print(f"{'指标':<14}{'目标值':>10}{'实测':>10}{'差':>10}{'容差':>10}  判定")
    allok = True
    for key, label in [("r2_full", "满输入 R²"), ("K0_mae", "K=0 MAE"),
                       ("K0_rho", "K=0 Spearman"), ("K200_mae", "K=200 MAE")]:
        tgt, got, tol = REF["gnn_ours"][key], m.get(key, np.nan), REPRO_TOL[key]
        good = abs(got - tgt) <= tol
        allok &= bool(good)
        print(f"{label:<14}{tgt:>10.4f}{got:>10.4f}{got-tgt:>+10.4f}{tol:>10.3f}  "
              f"{'OK' if good else 'FAIL'}")
    print("\n结论：" + ("复现通过，新代码可信，继续 Round A。"
                       if allok else "**复现失败 —— 停下来查 bug，禁止往下跑。**"))
    return allok


def cmd_round(which: str):
    truth = load_truth()
    variants = sorted({p.stem.replace("est_", "").rsplit("_seed", 1)[0]
                       for p in OUT.glob("est_*_seed*.csv")})
    if which == "A":
        variants = [v for v in variants if v.startswith("A")]
    elif which == "B":
        variants = [v for v in variants if v.startswith(("A0", "A1", "A2", "A3", "A4", "B", "D"))]
    if not variants:
        raise SystemExit("没有可用的评测结果")

    df = collect(variants, truth)
    df.to_csv(OUT / f"raw_by_seed_{which}.csv", index=False)

    g = df.groupby("variant")
    summ = g.mean(numeric_only=True)
    std = g.std(numeric_only=True)
    summ["n_seed"] = g.size()
    summ["K0_mae_sd"] = std["K0_mae"]
    summ["K0_rho_sd"] = std["K0_rho"]
    summ["closure"] = summ["K0_mae"].map(closure)
    summ["verdict"] = [verdict(r.K0_mae, r.K0_rho) for r in summ.itertuples()]
    summ = summ.sort_values("K0_mae")
    summ.to_csv(OUT / f"variant_comparison_{which}.csv")

    print("=" * 96)
    print(f"Round {which} 主结果（按 K=0 MAE 升序；参照 MLP 0.0597/0.9698，现状 GNN 0.0963/0.9197）")
    print("=" * 96)
    hdr = (f"{'variant':<16}{'seed':>5}{'参数k':>8}{'满输入R²':>10}"
           f"{'K0_MAE':>9}{'±σ':>7}{'K0_rho':>8}{'±σ':>7}{'K200_MAE':>10}{'缩差':>7}  判定")
    print(hdr)
    print("-" * 96)
    for v, r in summ.iterrows():
        print(f"{v:<16}{int(r.n_seed):>5}{r.n_params/1000:>8.0f}{r.r2_full:>10.4f}"
              f"{r.K0_mae:>9.4f}{r.K0_mae_sd if pd.notna(r.K0_mae_sd) else 0:>7.4f}"
              f"{r.K0_rho:>8.4f}{r.K0_rho_sd if pd.notna(r.K0_rho_sd) else 0:>7.4f}"
              f"{r.K200_mae:>10.4f}{r.closure:>6.0%}  {r.verdict}")
    print("-" * 96)
    print(f"{'[参照] MLP-oracle':<16}{'':>5}{157:>8}{0.8370:>10.4f}{0.0597:>9.4f}"
          f"{'':>7}{0.9698:>8.4f}{'':>7}{0.0163:>10.4f}{'100%':>7}")
    print(f"{'[参照] 63号GNN':<16}{'':>5}{250:>8}{0.7593:>10.4f}{0.0963:>9.4f}"
          f"{'':>7}{0.9197:>8.4f}{'':>7}{0.0405:>10.4f}{'0%':>7}")

    # 分尺寸带
    bcols = [f"K0_mae_{b}" for b in BAND_ORDER if f"K0_mae_{b}" in summ.columns]
    if bcols:
        print(f"\n=== K=0 分尺寸带 MAE（现状 GNN：{'/'.join(['0.072','0.133','0.167','0.201','0.138','0.107'])}）===")
        print(summ[bcols].round(4).to_string())

    # D_bypass 是【诊断变体】不是候选方案（预注册已写明），排除在最优候选之外；
    # MLP 是参照基线，同样不是"图结构"的候选。
    cand = summ[~summ.index.isin(["D_bypass", "MLP"])]
    best = cand.index[0]
    br = cand.loc[best]
    print("\n" + "=" * 96)
    print(f"最优变体：{best}  →  K0 MAE {br.K0_mae:.4f} / Spearman {br.K0_rho:.4f} "
          f"/ 缩掉差距 {br.closure:.0%}  →  **{br.verdict}**")
    if which in ("B", "final"):
        print("\n判据对照：")
        print(f"  翻案       MAE ≤ {GATE['overturn_mae']} 且 rho ≥ {GATE['overturn_rho']}")
        print(f"  有戏(70%)  MAE ≤ {GATE['promising_mae']} 且 rho ≥ {GATE['promising_rho']}")
        print(f"  放弃(<45%) MAE > {GATE['abandon_mae']}")
        if "D_bypass" in summ.index:
            d = summ.loc["D_bypass"]
            print(f"\n诊断旁路 D_bypass：MAE {d.K0_mae:.4f}（MLP 单独 0.0597）"
                  f" → 图分支{'几乎无贡献' if d.K0_mae >= 0.0597 - 0.003 else '携带互补信息'}")
    print("=" * 96)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-protocol", action="store_true")
    ap.add_argument("--check-repro", action="store_true")
    ap.add_argument("--round", choices=["A", "B", "final"])
    a = ap.parse_args()
    if a.check_protocol:
        sys.exit(0 if cmd_check_protocol() else 1)
    if a.check_repro:
        sys.exit(0 if cmd_check_repro() else 1)
    if a.round:
        cmd_round(a.round)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
