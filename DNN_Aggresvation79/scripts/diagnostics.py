# -*- coding: utf-8 -*-
"""定位剩余召回瓶颈：目标字段差异、K0漏检救回率和候选稳定性。"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KGRID = (0, 1, 5, 10, 25, 50)
N_ALL = 13244
STRONG = 0.10


def wmean(values, weights):
    return float(np.sum(np.asarray(values) * np.asarray(weights)) / np.sum(weights))


def main() -> None:
    score = pd.read_csv(OUT / "triple_score_kgrid.csv", index_col=0)
    score.index = score.index.map(ast.literal_eval)
    score.columns = [int(column) for column in score.columns]
    truth = pd.read_csv(OUT / "truth_design_split.csv")
    truth["indices"] = truth.indices.map(ast.literal_eval)
    truth["strong"] = truth.strong.astype(bool)

    # 找每个三元组真值最大的 confidential 字段，检查全局 max-conf 筛选是否偏科。
    syn = pd.read_csv(OUT / "syn3_entries_kgrid.csv.gz")
    syn["indices"] = list(zip(syn.i, syn.j, syn.k))
    truth_entries = []
    old = pd.read_csv(ROOT.parent / "DNN_Aggresvation68" / "outputs" / "triples_certified.csv")
    # 已在主分析中保存了 entry 设计，直接从原真值与 score 的 join 不需要再构造权重；
    # target 字段可从 h2 pool + old397 的 per-conf 真值取得。
    design_entries = pd.read_csv(ROOT.parent / "DNN_Aggresvation77" / "outputs" / "h2_unbiased_pool.csv")
    design_entries["indices"] = design_entries.ix.map(ast.literal_eval)
    extension = design_entries[design_entries.group == "triple_rand_ext"][
        ["indices", "conf", "syn3_true"]
    ]

    # old397 的字段名转索引：从 K 网格与旧文件按三字段名称无法直接映射，
    # 借用 78 号已保存的 2000 池覆盖 old397，取其中非 extension 的相同集合即可。
    # h2_unbiased_pool 本身含旧 200 个而非全部397，因此对缺失的197用其 group/真值文件，
    # 字段索引映射从 78 的 correction 脚本口径重建。
    import sys
    import yaml

    repo = ROOT.parent
    r69 = repo / "DNN_Aggresvation69"
    sys.path.insert(0, str(r69))
    from src.data_processing import prepare_data

    cfg = yaml.safe_load((repo / "DNN_Aggresvation77" / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(repo / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    data = prepare_data(cfg)
    name_to_idx = {name: idx for idx, name in enumerate(data["general"])}
    old["indices"] = [
        tuple(sorted((name_to_idx[a], name_to_idx[b], name_to_idx[c])))
        for a, b, c in zip(old.fi, old.fj, old.fk)
    ]
    old_entries = old[["indices", "conf", "syn3_true"]]
    entries = pd.concat([old_entries, extension], ignore_index=True)
    target = entries.loc[entries.groupby("indices").syn3_true.idxmax()][
        ["indices", "conf", "syn3_true"]
    ].rename(columns={"conf": "target_conf"})
    target = truth.merge(target, on=["indices", "syn3_true"], how="left", validate="one_to_one")

    per_target_rows = []
    keep_sets = {}
    n_keep = int(round(N_ALL * 0.30))
    for k_value in KGRID:
        keep = set(score[k_value].nlargest(n_keep).index)
        keep_sets[k_value] = keep
        target["hit"] = target.indices.map(lambda key: key in keep)
        for conf, group in target[target.strong].groupby("target_conf"):
            per_target_rows.append(
                {
                    "K": k_value,
                    "target_conf": conf,
                    "n_sample": len(group),
                    "estimated_strong_count": float(group.weight.sum()),
                    "recall_top30": wmean(group.hit.astype(float), group.weight),
                }
            )
    pd.DataFrame(per_target_rows).to_csv(OUT / "per_target_recall_top30.csv", index=False)

    strong = truth[truth.strong].copy()
    strong["k0_hit"] = strong.indices.map(lambda key: key in keep_sets[0])
    missed = strong[~strong.k0_hit].copy()
    rescue_rows = []
    for k_value in KGRID[1:]:
        hit = missed.indices.map(lambda key: key in keep_sets[k_value]).astype(float)
        rescue_rows.append(
            {
                "K": k_value,
                "estimated_k0_missed_count": float(missed.weight.sum()),
                "rescued_fraction_of_k0_misses": wmean(hit, missed.weight),
            }
        )
    pd.DataFrame(rescue_rows).to_csv(OUT / "k0_miss_rescue_top30.csv", index=False)

    overlap_rows = []
    for left in KGRID:
        for right in KGRID:
            intersection = len(keep_sets[left] & keep_sets[right])
            union = len(keep_sets[left] | keep_sets[right])
            overlap_rows.append(
                {
                    "K_left": left,
                    "K_right": right,
                    "jaccard_top30": intersection / union,
                    "intersection": intersection,
                }
            )
    pd.DataFrame(overlap_rows).to_csv(OUT / "candidate_overlap_top30.csv", index=False)


if __name__ == "__main__":
    main()
