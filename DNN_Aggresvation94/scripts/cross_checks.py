# -*- coding: utf-8 -*-
"""94 号 分析①：三个零 GPU 的交叉判定（全部查 93 号已落盘的扫描/认证结果）。

A. **L1 是否"拒绝"full 的假 top**：full 字典在 o4/o5 报的最强集合（认证已证全假），
   在 L1 的全扫里排第几、估计值多少。若 L1 给它们的 syn 都很低 ⟹ L1 不只是"另选了一批"，
   而是**正确地否决了 full 的整个候选榜**。
B. **层级搜索会不会漏掉真协同**（用户最初的核心问题）：对已认证为真的 o5 集合，
   查它们的 5 个四阶父集在 L1 o4 全扫 syn 排名里的最好名次。
   若最好父集排在 beam 宽度 B 之外 ⟹ 层级搜索（beam）会漏掉它——用**认证过的真集合**
   回答"反层级是否存在"，而不是 88 号那种用代理尾部回答。
C. **真值版衰减律汇总**：o3(82号重训 0.455 / FDR 0.461) → o4(本批认证) → o5(93号认证)。
"""
from __future__ import annotations

import ast
import glob
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R93 = ROOT.parent / "DNN_Aggresvation93"
R90 = ROOT.parent / "DNN_Aggresvation90"


def load_scan(prefix: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(R93 / f"outputs/{prefix}_s*of*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d["S"] = d["indices"].map(lambda s: tuple(ast.literal_eval(s)))
    return d


def load_cert(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(R93 / f"outputs/{pattern}")))
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

    # ---- A. L1 对 full 假 top 的判定 ----
    print("=" * 72)
    print("A. L1 是否正确否决 full 字典的假 top（认证已证其为假）")
    for order, cert_pat, l1map, rankmap, n_space in [
            (5, "certify_o5_s*of[24].csv", l1_5, rank5, len(scan5)),
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
              f"{s.syn_struct.mean():.3f}，认证均值 {s.syn_true_audit.mean():.3f}）")
        print(f"    → L1 给它们的估计: 均值 {np.nanmean(est):.4f}  最大 {np.nanmax(est):.4f}")
        print(f"    → L1 排名: 中位 {int(np.median(rk)):,} / {n_space:,}"
              f"（前 1000 名内: {(rk <= 1000).sum()} 个；前 100: {(rk <= 100).sum()}）")

    # ---- B. 层级搜索会不会漏掉认证过的真协同 ----
    print()
    print("=" * 72)
    print("B. 已认证为真的 o5 协同（syn_true>0.10），其最好四阶父集在 o4 扫描的排名")
    l1top5 = load_cert("certify_o5_l1top_s*.csv")
    true5 = l1top5[l1top5.syn_true_audit > 0.10].sort_values(
        "syn_true_audit", ascending=False)
    rows = []
    for r in true5.itertuples():
        parents = [tuple(sorted(t)) for t in combinations(r.Stup, 4)]
        pranks = sorted(rank4.get(p, 10 ** 9) for p in parents)
        rows.append(dict(S=str(r.Stup), syn_true=round(r.syn_true_audit, 4),
                         best_parent_rank=pranks[0],
                         parent_ranks=str(pranks)))
    b = pd.DataFrame(rows)
    print(b.to_string(index=False))
    for B_ in (100, 500, 1000, 5000):
        miss = int((b.best_parent_rank > B_).sum())
        print(f"  beam 宽度 B={B_:5d}: 会漏掉 {miss}/{len(b)} 个已认证真协同")
    b.to_csv(ROOT / "outputs/hierarchy_check_o5.csv", index=False)

    # ---- C. 真值版衰减律 ----
    print()
    print("=" * 72)
    print("C. 真值版衰减律（全部经认证/统计保证，两种手工字典的版本均已被证伪）")
    l1top4 = load_cert("certify_o4_l1top_s*.csv")
    s4 = l1top4[l1top4.group == "strong"]
    t5 = l1top5
    rows = [
        dict(阶=3, 最强真值=0.455, 口径="82号专用DNN重训（FDR配对检验独立给 0.461，互证）",
             强协同数="66 个过 FDR(τ=0.10, q=0.05)——统计保证的计数"),
        dict(阶=4, 最强真值=round(float(s4.syn_true_audit.max()), 3),
             口径=f"本批专用DNN重训（L1自选top-{len(s4)}，{int((s4.syn_true_audit>0.10).sum())}个>0.10）",
             强协同数=f"L1 估计 {int((scan4.syn>0.1).sum())} 个>0.10"
                    f"（top精确率 {int((s4.syn_true_audit>0.10).sum())}/{len(s4)}）"),
        dict(阶=5, 最强真值=round(float(t5.syn_true_audit.max()), 3),
             口径=f"93号专用DNN重训（L1自选top-{len(t5)}，{int((t5.syn_true_audit>0.10).sum())}个>0.10）",
             强协同数=f"L1 估计 {int((scan5.syn>0.1).sum())} 个>0.10"
                    f"（top精确率 {int((t5.syn_true_audit>0.10).sum())}/{len(t5)}）"),
    ]
    c = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(c.to_string(index=False))
    c.to_csv(ROOT / "outputs/decay_law_certified.csv", index=False)
    dens = [66 / 13244, (s4.syn_true_audit > 0.10).mean() * (scan4.syn > 0.1).sum() / len(scan4),
            (t5.syn_true_audit > 0.10).mean() * (scan5.syn > 0.1).sum() / len(scan5)]
    print(f"\n  强度衰减: 0.455 → {s4.syn_true_audit.max():.3f} → {t5.syn_true_audit.max():.3f}"
          f"  （{0.455/s4.syn_true_audit.max():.1f}× 再 {s4.syn_true_audit.max()/t5.syn_true_audit.max():.1f}×，温和单调，不是断崖也不是不衰减）")
    print(f"  密度(>0.10)估计: {dens[0]*100:.3f}% → {dens[1]*100:.4f}% → {dens[2]*100:.5f}%"
          f"  （每升一阶约 {dens[0]/dens[1]:.0f}× / {dens[1]/dens[2]:.0f}×；"
          f"o3 为 FDR 计数，o4/o5 为精确率校准的估计）")


if __name__ == "__main__":
    main()
