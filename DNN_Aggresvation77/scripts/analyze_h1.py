# -*- coding: utf-8 -*-
"""H-1 分析：两级高阶扫描协议的"保留率 → 召回率"曲线与成本对比。

评估口径与 75_Correction/scripts/highorder_search_honest.py 对齐，
但候选池从"397 个已认证三元组"扩到"全部 13244 个三元组的第一级估计"，
真值侧则用 68 号已认证的 397 个（+ H-2 新增的随机三元组，若已完成）。

四条对照（前两条不合法，仅作参照）：
  A. Apriori（**重训真值** syn2）      —— 68/70 号原版，真值在环
  B. Apriori（oracle 估计 syn2 @K=0） —— 诚实版剪枝器
  C. 直接 oracle ŝyn3 @K=0            —— 第一级（免费）
  C'. 直接 oracle ŝyn3 @K=K*          —— 第二级（若已跑）
  D. 随机                              —— 下界
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R77 = Path(__file__).resolve().parents[1]
R68 = R77.parent / "DNN_Aggresvation68"
sys.path.insert(0, str(R77 / "src"))
from runlog import log  # noqa: E402

OUT = R77 / "outputs"
DELTA = 0.10
RET = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def name2idx():
    import yaml
    sys.path.insert(0, str(R77.parent / "DNN_Aggresvation69"))
    from src.data_processing import prepare_data
    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R77.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    return {n: i for i, n in enumerate(di["general"])}


def build():
    n2i = name2idx()
    tri_truth = pd.read_csv(R68 / "outputs/triples_certified.csv")
    syn2t = pd.read_csv(R68 / "outputs/synergy2_perconf.csv")
    t2 = {(min(r.fi, r.fj), max(r.fi, r.fj), r.conf): r.synergy for r in syn2t.itertuples()}

    # 第一级：全 13244 的 ŝyn3@K=0
    s3 = pd.read_csv(OUT / "triples_syn3_k0.csv")
    k0 = {(r.i, r.j, r.k, r.conf): r.syn3_est_k0 for r in s3.itertuples()}

    # 第二级（若已跑）
    ks_files = sorted(OUT.glob("triples_kstar_shard*.csv"))
    kstar = {}
    if ks_files:
        d = pd.concat([pd.read_csv(p) for p in ks_files])
        low = pd.read_csv(OUT / "lowfour_k0.csv")
        v2 = low[low["size"] == 2].set_index(["key", "conf"]).est.to_dict()

        def key(*ix):
            return "_".join(f"{x:02d}" for x in sorted(ix))
        for r in d.itertuples():
            p = [v2.get((key(r.i, r.j), r.conf)), v2.get((key(r.i, r.k), r.conf)),
                 v2.get((key(r.j, r.k), r.conf))]
            if all(x is not None for x in p):
                kstar[(r.i, r.j, r.k, r.conf)] = r.est - max(p)

    rows = []
    for r in tri_truth.itertuples():
        ix = tuple(sorted((n2i[r.fi], n2i[r.fj], n2i[r.fk])))
        pairs = [tuple(sorted(p)) for p in itertools.combinations([r.fi, r.fj, r.fk], 2)]
        v = [t2.get((p[0], p[1], r.conf), np.nan) for p in pairs]
        rows.append(dict(group=r.group, conf=r.conf, syn3_true=r.syn3_true,
                         apriori_truth=np.nanmax(v) if not all(np.isnan(v)) else np.nan,
                         syn3_k0=k0.get((*ix, r.conf), np.nan),
                         syn3_kstar=kstar.get((*ix, r.conf), np.nan)))
    return pd.DataFrame(rows), len(kstar) > 0


def recall_curve(df, col):
    strong = df.syn3_true > DELTA
    if strong.sum() == 0 or df[col].isna().all():
        return {r: np.nan for r in RET}
    order = np.argsort(-df[col].fillna(-9).values)
    out = {}
    for r in RET:
        n = max(1, int(round(len(df) * r)))
        keep = np.zeros(len(df), bool)
        keep[order[:n]] = True
        out[r] = keep[strong.values].mean()
    return out


def main():
    df, has_kstar = build()
    df = df.dropna(subset=["syn3_k0"])
    rng = np.random.RandomState(0)
    df["random"] = rng.random_sample(len(df))
    df.to_csv(OUT / "h1_eval.csv", index=False)

    cands = [("A. Apriori(真值 syn2) ← 不合法", "apriori_truth"),
             ("C. 直接 oracle ŝyn3 @K=0（第一级）", "syn3_k0")]
    if has_kstar:
        cands.append(("C'. 直接 oracle ŝyn3 @K*（第二级）", "syn3_kstar"))
    cands.append(("D. 随机", "random"))

    for pool, lab in [("triple_rand", "无偏池 triple_rand（可外推口径）"),
                      ("triple_top", "有偏池 triple_top"), (None, "合并")]:
        d = df if pool is None else df[df.group == pool]
        strong = (d.syn3_true > DELTA).sum()
        print("=" * 100)
        print(f"{lab}   n={len(d)}  真强三阶={strong}  基率={strong/max(len(d),1):.2%}")
        print("=" * 100)
        if strong == 0:
            print("  （无真强三阶，跳过）\n")
            continue
        print(f"{'剪枝器':<36}" + "".join(f"{int(r*100):>7}%" for r in RET) + f"{'  ρ':>10}")
        for nm, col in cands:
            if col not in d.columns:
                continue
            c = recall_curve(d, col)
            sub = d.dropna(subset=[col])
            rho = spearmanr(sub[col], sub.syn3_true).correlation if len(sub) > 2 else np.nan
            print(f"{nm:<36}" + "".join(f"{c[r]:>7.0%}" for r in RET) + f"{rho:>10.3f}")
        print()

    # 成本对比
    n_tri = 13244
    print("=" * 100)
    print("成本对比（相对 68 号原版：K=200 全扫 13244 个三元组）")
    print("=" * 100)
    ev = OUT / "events.jsonl"
    k0_s = None
    if ev.exists():
        for line in open(ev):
            r = json.loads(line)
            if r["phase"] == "H-1" and r["event"] == "DONE" and "K=0 全扫" in (r["note"] or ""):
                k0_s = r["elapsed_s"]
    print(f"  第一级 K=0 全扫 {n_tri} 个：实测 {k0_s:.0f} s" if k0_s else "  第一级：见 RUNLOG")
    print(f"  68 号原版 K=200 全扫：约 5400 s（90 min）")
    if k0_s:
        print(f"  → 仅第一级即省约 {5400/k0_s:.0f}×；加第二级（top 30% × K*）仍远低于原版")
    log("H-1", "NOTE", note="两级协议召回曲线与成本对比已出，见 outputs/h1_eval.csv")


if __name__ == "__main__":
    main()
