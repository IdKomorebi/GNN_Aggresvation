# -*- coding: utf-8 -*-
"""95_caiso 收束分析（改自 94 号 cross_checks.py，全部零 GPU 查本目录落盘结果）。

A. L1 是否否决 full 字典的 top（CAISO 上 full 未报 >0.2，此处判定的是"两个估计器
   的 top 榜单是否重叠、full 的 top 在 L1 排名如何"）；
B. 反层级检查：已认证为真的 o5 集合（若有），其最好四阶父集在 o4 扫描的排名；
C. 真值版衰减律（CAISO）：o3(FDR 统计口径 + 认证) → o4(认证) → o5(认证)。
"""
from __future__ import annotations

import ast
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_scan(prefix: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / f"outputs/{prefix}_s*of*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


def load_cert(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / f"outputs/{pattern}")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("S")
    d["Stup"] = d["S"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


def main() -> None:
    scan4 = load_scan("scan_o4_oracle_l1_aug8_seed0_last")
    scan5 = load_scan("scan_o5_oracle_l1_aug8_seed0_last")
    l1_4 = {r.S: r.syn for r in scan4.itertuples()}
    l1_5 = {r.S: r.syn for r in scan5.itertuples()}
    rank4 = {S: i + 1 for i, S in enumerate(
        scan4.sort_values("syn", ascending=False)["S"])}
    rank5 = {S: i + 1 for i, S in enumerate(
        scan5.sort_values("syn", ascending=False)["S"])}

    # ---- A. full top 在 L1 视角下的判定 ----
    print("=" * 72)
    print("A. full 字典的 top 在 L1 全扫中的估计与排名（+认证真值）")
    for order, cert_pat, l1map, rankmap, n_space in [
            (5, "certify_o5_fulltop_s*.csv", l1_5, rank5, len(scan5)),
            (4, "certify_o4_fulltop_s*.csv", l1_4, rank4, len(scan4))]:
        try:
            cert = load_cert(cert_pat)
        except ValueError:
            print(f"  order-{order}: 认证文件尚未就绪，跳过")
            continue
        s = cert[cert.group == "strong"] if "group" in cert else cert
        est = np.array([l1map.get(S, np.nan) for S in s.Stup])
        rk = np.array([rankmap.get(S, -1) for S in s.Stup])
        print(f"  order-{order}: full 的 top-{len(s)}（结构化 syn 均值 "
              f"{s.syn_struct.mean():.3f}，认证均值 {s.syn_true_audit.mean():.3f}，"
              f"认证>0.10 的 {int((s.syn_true_audit > 0.10).sum())} 个）")
        print(f"    → L1 给它们的估计: 均值 {np.nanmean(est):.4f}  最大 {np.nanmax(est):.4f}")
        print(f"    → L1 排名: 中位 {int(np.median(rk)):,} / {n_space:,}"
              f"（前 1000 名内: {(rk <= 1000).sum()} 个；前 100: {(rk <= 100).sum()}）")

    # ---- B. 反层级检查 ----
    print()
    print("=" * 72)
    print("B. 已认证为真的 o5 协同（syn_true>0.10），其最好四阶父集在 o4 扫描的排名")
    l1top5 = load_cert("certify_o5_l1top_s*.csv")
    true5 = l1top5[(l1top5.group == "strong") & (l1top5.syn_true_audit > 0.10)]
    if len(true5) == 0:
        print("  （无认证 >0.10 的 o5 集合——CAISO 高阶弱，改看认证 top-5 的父集排名）")
        true5 = l1top5[l1top5.group == "strong"].nlargest(5, "syn_true_audit")
    rows = []
    for r in true5.sort_values("syn_true_audit", ascending=False).itertuples():
        parents = [tuple(sorted(t)) for t in combinations(r.Stup, 4)]
        pranks = sorted(rank4.get(p, 10 ** 9) for p in parents)
        rows.append(dict(S=str(r.Stup), syn_true=round(r.syn_true_audit, 4),
                         best_parent_rank=pranks[0], parent_ranks=str(pranks)))
    b = pd.DataFrame(rows)
    print(b.to_string(index=False))
    for B_ in (100, 500, 1000, 5000):
        miss = int((b.best_parent_rank > B_).sum())
        print(f"  beam 宽度 B={B_:5d}: 会漏掉 {miss}/{len(b)} 个上述集合")
    b.to_csv(ROOT / "outputs/hierarchy_check_o5.csv", index=False)

    # ---- C. 真值版衰减律（CAISO） ----
    print()
    print("=" * 72)
    print("C. CAISO 真值版衰减律")
    fdr = pd.read_csv(ROOT / "outputs/paired_fdr_o3_cat3.csv")
    n_fdr = int(fdr["rej@0.1"].sum())
    cert3 = load_cert("certify_o3_fdrtop_s*.csv")
    s3 = cert3[cert3.group == "strong"]
    l1top4 = load_cert("certify_o4_l1top_s*.csv")
    s4 = l1top4[l1top4.group == "strong"]
    t5 = l1top5[l1top5.group == "strong"]
    rows = [
        dict(阶=3, 最强真值=round(float(s3.syn_true_audit.max()), 3),
             口径=f"专用DNN重训（FDR-top-{len(s3)}；FDR 统计口径最强 "
                f"{fdr[fdr['rej@0.1']==True].syn_audit.max():.3f} 互证），"  # noqa: E712
                f"{int((s3.syn_true_audit>0.10).sum())}个>0.10",
             强协同数=f"{n_fdr} 个过 FDR(τ=0.10, q=0.05)——统计保证的计数"),
        dict(阶=4, 最强真值=round(float(s4.syn_true_audit.max()), 3),
             口径=f"专用DNN重训（L1自选top-{len(s4)}，{int((s4.syn_true_audit>0.10).sum())}个>0.10）",
             强协同数=f"L1 估计 {int((scan4.syn>0.1).sum())} 个>0.10"
                    f"（top精确率 {int((s4.syn_true_audit>0.10).sum())}/{len(s4)}）"),
        dict(阶=5, 最强真值=round(float(t5.syn_true_audit.max()), 3),
             口径=f"专用DNN重训（L1自选top-{len(t5)}，{int((t5.syn_true_audit>0.10).sum())}个>0.10）",
             强协同数=f"L1 估计 {int((scan5.syn>0.1).sum())} 个>0.10"
                    f"（top精确率 {int((t5.syn_true_audit>0.10).sum())}/{len(t5)}）"),
    ]
    c = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(c.to_string(index=False))
    c.to_csv(ROOT / "outputs/decay_law_certified.csv", index=False)
    m3, m4, m5 = (float(s3.syn_true_audit.max()), float(s4.syn_true_audit.max()),
                  float(t5.syn_true_audit.max()))
    d3 = n_fdr / len(fdr)
    p4 = (s4.syn_true_audit > 0.10).mean() if len(s4) else float("nan")
    p5 = (t5.syn_true_audit > 0.10).mean() if len(t5) else float("nan")
    d4 = p4 * (scan4.syn > 0.1).sum() / len(scan4)
    d5 = p5 * (scan5.syn > 0.1).sum() / len(scan5)
    print(f"\n  强度衰减: {m3:.3f} → {m4:.3f} → {m5:.3f}")
    print(f"  密度(>0.10)估计: {d3*100:.3f}% → {d4*100:.4f}% → {d5*100:.5f}%"
          f"（o3 为 FDR 计数，o4/o5 为精确率校准估计）")
    print(f"  （PJM 对照：强度 0.455→0.196→0.154；密度 0.498%→0.069%→0.0051%）")


if __name__ == "__main__":
    main()
