# -*- coding: utf-8 -*-
"""从79号同保真K网格构造S1、任意目标S2、与S1目标对齐的S2。"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R79 = ROOT.parent / "DNN_Aggresvation79" / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from common import KGRID, N_ALL  # noqa: E402
from runlog import Timer, log  # noqa: E402


def main() -> None:
    log("BUILD", "START", note="构造同K的S1与两种S2")
    with Timer() as timer:
        entries = pd.read_csv(R79 / "syn3_entries_kgrid.csv.gz")
        assert set(entries.K.unique()) == set(KGRID)
        group_keys = ["i", "j", "k", "K"]

        # S1最大的confidential目标；aligned S2严格取同一个目标的父集合值。
        aligned_idx = entries.groupby(group_keys).syn3_est.idxmax()
        aligned = entries.loc[
            aligned_idx,
            group_keys + ["conf", "syn3_est", "est", "best_pair_est"],
        ].copy()
        aligned = aligned.rename(
            columns={
                "conf": "s1_conf",
                "syn3_est": "s1_syn3",
                "est": "s2_parent_aligned",
                "best_pair_est": "best_pair_aligned",
            }
        )

        any_parent = (
            entries.groupby(group_keys, as_index=False)
            .agg(
                s2_parent_any=("est", "max"),
                s1_min=("syn3_est", "min"),
            )
        )
        cube = aligned.merge(any_parent, on=group_keys, validate="one_to_one")
        cube["indices"] = list(zip(cube.i, cube.j, cube.k))

        truth = pd.read_csv(R79 / "truth_design_split.csv")
        truth["indices"] = truth.indices.map(ast.literal_eval)
        cube = cube.merge(
            truth[
                ["indices", "source", "syn3_true", "weight", "strong", "split"]
            ],
            on="indices",
            how="left",
            validate="many_to_one",
        )
        cube["strong"] = cube.strong.astype("boolean")
        cube.to_parquet(ROOT / "outputs" / "score_cube.parquet", index=False)

        # 与79号发布的S1逐项核对，防止聚合口径漂移。
        old = pd.read_csv(R79 / "triple_score_kgrid.csv", index_col=0)
        old.index = old.index.map(ast.literal_eval)
        old.columns = [int(column) for column in old.columns]
        new = cube.pivot(index="indices", columns="K", values="s1_syn3")
        new = new.loc[old.index, old.columns]
        max_diff = float(np.abs(new.to_numpy() - old.to_numpy()).max())
        assert max_diff < 1e-12
        assert len(cube) == N_ALL * len(KGRID)
        assert cube.groupby(["i", "j", "k"]).ngroups == N_ALL

    log(
        "BUILD",
        "DONE",
        note="S1与79号完全一致；新增S2-any和S2-aligned",
        elapsed_s=timer.elapsed,
        rows=len(cube),
        max_s1_diff=max_diff,
    )
    print(
        f"完成 score_cube.parquet：{len(cube)}行；"
        f"S1最大差异={max_diff:.3g}"
    )


if __name__ == "__main__":
    main()

