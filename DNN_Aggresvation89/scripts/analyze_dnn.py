#!/usr/bin/env python3
"""用专用 DNN 结果比较 any-conf 与 same-conf 父级信号。"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RET = ROOT / "outputs/retrain"


def sid(key) -> str:
    return "s" + "_".join(f"{int(x):02d}" for x in sorted(key))


def load(key, seed=0):
    d = json.loads((RET / f"{sid(key)}_seed{seed}.json").read_text(encoding="utf-8"))
    return np.asarray(list(d["per_conf_r2"].values()), dtype=np.float64), list(
        d["per_conf_r2"].keys()
    )


def compute_rows(meta, seeds: list[int], aggregate: bool) -> list[dict]:
    rows = []
    seed_values = [None] if aggregate else seeds
    for seed in seed_values:
        for qid, info in meta.items():
            quad = tuple(info["indices"])

            def get(key):
                if seed is not None:
                    return load(key, seed)[0]
                return np.mean([load(key, s)[0] for s in seeds], axis=0)

            vq = get(quad)
            # confidential 名称顺序对所有输出固定一致
            conf_names = load(quad, seeds[0])[1]
            triples = list(combinations(quad, 3))
            vt = np.stack([get(t) for t in triples])
            syn4 = vq - vt.max(axis=0)
            c4 = int(np.argmax(syn4))

            syn3 = []
            for t, vtri in zip(triples, vt):
                pairs = list(combinations(t, 2))
                vp = np.stack([get(p) for p in pairs])
                syn3.append(vtri - vp.max(axis=0))
            syn3 = np.stack(syn3)
            _, c3 = np.unravel_index(int(np.argmax(syn3)), syn3.shape)

            rows.append(
                {
                    "qid": qid,
                    "indices": ",".join(map(str, quad)),
                    "seed": "mean_r2" if seed is None else seed,
                    "selection_group": info["selection_group"],
                    "proxy_child_search_syn": info["proxy_child_search_syn"],
                    "proxy_child_audit_syn": info["proxy_child_audit_syn"],
                    "proxy_any_rank_search": info["proxy_any_rank_search"],
                    "proxy_aligned_rank_search": info["proxy_aligned_rank_search"],
                    "proxy_child_search_conf": info["proxy_child_search_conf"],
                    "proxy_child_audit_conf": info["proxy_child_audit_conf"],
                    "dnn_syn4": float(syn4[c4]),
                    "dnn_child_conf": conf_names[c4],
                    "dnn_child_conf_index": c4,
                    "proxy_search_conf_matches_dnn": (
                        info["proxy_child_search_conf"] == c4
                    ),
                    "proxy_audit_conf_matches_dnn": (
                        info["proxy_child_audit_conf"] == c4
                    ),
                    "dnn_parent_any": float(syn3.max()),
                    "dnn_parent_aligned": float(syn3[:, c4].max()),
                    "dnn_best_parent_conf": conf_names[c3],
                    "dnn_parent_conf_match": c3 == c4,
                    "dnn_any_parent_gt_005": syn3.max() > 0.05,
                    "dnn_aligned_parent_gt_005": syn3[:, c4].max() > 0.05,
                    "dnn_any_parent_gt_010": syn3.max() > 0.10,
                    "dnn_aligned_parent_gt_010": syn3[:, c4].max() > 0.10,
                }
            )
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("selection_group")
        .agg(
            n=("qid", "size"),
            dnn_syn4=("dnn_syn4", "mean"),
            dnn_parent_any=("dnn_parent_any", "mean"),
            dnn_parent_aligned=("dnn_parent_aligned", "mean"),
            conf_match=("dnn_parent_conf_match", "mean"),
            any_gt005=("dnn_any_parent_gt_005", "mean"),
            aligned_gt005=("dnn_aligned_parent_gt_005", "mean"),
            any_gt010=("dnn_any_parent_gt_010", "mean"),
            aligned_gt010=("dnn_aligned_parent_gt_010", "mean"),
        )
        .reset_index()
    )


def strong_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tau in (0.05, 0.10, 0.15):
        g = df[df.dnn_syn4 > tau]
        rows.append(
            {
                "child_tau": tau,
                "n_strong": len(g),
                "n_aligned_selected": int((g.selection_group == "aligned").sum()),
                "n_cross_conf_selected": int(
                    (g.selection_group == "cross_conf").sum()
                ),
                "n_blind_selected": int((g.selection_group == "blind").sum()),
                "parent_conf_match": g.dnn_parent_conf_match.mean(),
                "aligned_parent_gt005": g.dnn_aligned_parent_gt_005.mean(),
                "aligned_parent_gt010": g.dnn_aligned_parent_gt_010.mean(),
                "mean_parent_any": g.dnn_parent_any.mean(),
                "mean_parent_aligned": g.dnn_parent_aligned.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    meta = json.loads(
        (ROOT / "outputs/dnn_quad_meta.json").read_text(encoding="utf-8")
    )
    seedwise = pd.DataFrame(compute_rows(meta, seeds, aggregate=False))
    seedwise.to_csv(ROOT / "outputs/dnn_hierarchy_seedwise.csv", index=False)
    df = pd.DataFrame(compute_rows(meta, seeds, aggregate=True))
    df.to_csv(ROOT / "outputs/dnn_hierarchy.csv", index=False)
    summary = summarize(df)
    summary.to_csv(ROOT / "outputs/dnn_hierarchy_summary.csv", index=False)
    strong = strong_summary(df)
    strong.to_csv(ROOT / "outputs/dnn_strong_summary.csv", index=False)

    agreement = pd.DataFrame(
        [
            {
                "sample_note": "24 search-selected quads; not a population estimate",
                "search_proxy_vs_dnn_spearman": df.proxy_child_search_syn.corr(
                    df.dnn_syn4, method="spearman"
                ),
                "audit_proxy_vs_dnn_spearman": df.proxy_child_audit_syn.corr(
                    df.dnn_syn4, method="spearman"
                ),
                "search_proxy_vs_dnn_mae": np.mean(
                    np.abs(df.proxy_child_search_syn - df.dnn_syn4)
                ),
                "audit_proxy_vs_dnn_mae": np.mean(
                    np.abs(df.proxy_child_audit_syn - df.dnn_syn4)
                ),
                "search_proxy_conf_match_dnn": df.proxy_search_conf_matches_dnn.mean(),
                "audit_proxy_conf_match_dnn": df.proxy_audit_conf_matches_dnn.mean(),
            }
        ]
    )
    agreement.to_csv(ROOT / "outputs/dnn_proxy_agreement.csv", index=False)

    # 同一四元组在不同训练种子上是否保持“强协同”与同一最强目标。
    pivot = seedwise.pivot(index="qid", columns="seed", values="dnn_syn4")
    s0 = seedwise[seedwise.seed == seeds[0]].set_index("qid")
    s1 = seedwise[seedwise.seed == seeds[1]].set_index("qid")
    stable = pd.DataFrame(
        {
            "metric": [
                "syn4_seed_mae",
                "syn4_seed_spearman",
                "strong_gt005_agreement",
                "strong_gt010_agreement",
                "child_conf_agreement",
            ],
            "value": [
                float(np.mean(np.abs(pivot[seeds[0]] - pivot[seeds[1]])))
                if len(seeds) == 2
                else float("nan"),
                float(pivot[seeds[0]].corr(pivot[seeds[1]], method="spearman"))
                if len(seeds) == 2
                else float("nan"),
                float(
                    np.mean(
                        (pivot[seeds[0]] > 0.05)
                        == (pivot[seeds[1]] > 0.05)
                    )
                )
                if len(seeds) == 2
                else float("nan"),
                float(
                    np.mean(
                        (pivot[seeds[0]] > 0.10)
                        == (pivot[seeds[1]] > 0.10)
                    )
                )
                if len(seeds) == 2
                else float("nan"),
                float(
                    np.mean(
                        s0.dnn_child_conf == s1.dnn_child_conf
                    )
                )
                if len(seeds) == 2
                else float("nan"),
            ],
        }
    )
    stable.to_csv(ROOT / "outputs/dnn_seed_stability.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print("\nDNN 强四阶样本的父级结构：")
    print(strong.round(4).to_string(index=False))
    print("\n代理与 DNN 的一致性（仅限有意选择的 24 个样本）：")
    print(agreement.round(4).to_string(index=False))
    print("\n种子稳定性：")
    print(stable.round(4).to_string(index=False))
    print("\n全部样本：")
    print(
        df[
            [
                "selection_group",
                "indices",
                "dnn_syn4",
                "dnn_parent_any",
                "dnn_parent_aligned",
                "dnn_parent_conf_match",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
