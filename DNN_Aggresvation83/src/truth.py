"""修正后的三阶真值加载。

D82 `load_truth()` 把 D77 的 `syn2_true_max`（二阶“协同增量”最大值）
误当成 `best_pair`（二元集合本身的 R²），再用二者相加重建 parent R²。
这会污染新增 1800 个随机三元组的 parent/pair 误差，但不污染其 syn3_true。

这里直接从 D77 的三元重训 JSON 和 D67 的低阶重训 JSON 重建三者。
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
R67 = REPO / "DNN_Aggresvation67"
R68 = REPO / "DNN_Aggresvation68"
R69 = REPO / "DNN_Aggresvation69"
R77 = REPO / "DNN_Aggresvation77"
R79 = REPO / "DNN_Aggresvation79"
sys.path.insert(0, str(R69))

from src.data_processing import prepare_data  # noqa: E402

N_ALL = 13244


def load_correct_truth() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load((R77 / "base.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["csv_path"] = str(
        REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv"
    )
    data = prepare_data(cfg)
    name_to_idx = {name: idx for idx, name in enumerate(data["general"])}
    conf_names = data["confidential"]

    # 已认证 397 个：文件本身直接保存 parent、best_pair、syn3。
    exact = pd.read_csv(R68 / "outputs/triples_certified.csv")
    exact["indices"] = [
        tuple(
            sorted(
                (
                    name_to_idx[row.fi],
                    name_to_idx[row.fj],
                    name_to_idx[row.fk],
                )
            )
        )
        for row in exact.itertuples()
    ]
    exact = exact.rename(
        columns={"v_ijk_true": "parent_true", "best_pair": "pair_true"}
    )
    exact = exact[
        ["indices", "conf", "parent_true", "pair_true", "syn3_true"]
    ]
    exact["source"] = "exact397"
    exact["weight"] = 1.0

    # 全部单元/二元集合的逐 conf 重训 R²。
    low_truth: dict[tuple[int, ...], dict[str, float]] = {}
    registry = json.loads((R67 / "outputs/subsets.json").read_text(encoding="utf-8"))
    for subset_id in registry:
        path = R67 / "outputs/retrain" / f"{subset_id}_dnn_seed0.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(sorted(name_to_idx[field] for field in payload["fields"]))
        low_truth[key] = payload["per_conf_r2"]

    # 1800 个随机三元组：直接读 parent，再从三个真实 pair 中取最大值。
    rows = []
    for path in sorted((R77 / "outputs/retrain").glob("r*_dnn_seed0.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = tuple(sorted(name_to_idx[field] for field in payload["fields"]))
        pairs = [tuple(sorted(pair)) for pair in combinations(key, 2)]
        for conf in conf_names:
            pair_values = [low_truth[pair][conf] for pair in pairs]
            parent = float(payload["per_conf_r2"][conf])
            best_pair = float(max(pair_values))
            rows.append(
                {
                    "indices": key,
                    "conf": conf,
                    "parent_true": parent,
                    "pair_true": best_pair,
                    "syn3_true": parent - best_pair,
                }
            )
    random = pd.DataFrame(rows)
    assert random.indices.nunique() == 1800
    random["source"] = "remainder1800"
    random["weight"] = (N_ALL - 397) / 1800

    entries = pd.concat([exact, random], ignore_index=True)
    design = pd.read_csv(R79 / "outputs/truth_design_split.csv")
    design["indices"] = design.indices.map(eval_tuple)
    entries = entries.merge(
        design[["indices", "split"]],
        on="indices",
        how="left",
        validate="many_to_one",
    )
    assert not entries.split.isna().any()
    triples = (
        entries.groupby(["indices", "source", "weight", "split"], as_index=False)
        .agg(syn3_true=("syn3_true", "max"))
    )
    triples["strong"] = triples.syn3_true > 0.10
    return entries, triples


def eval_tuple(value: str) -> tuple[int, ...]:
    # 输入仅来自项目生成的整数 tuple CSV；显式解析避免 ast 在热路径重复导入。
    return tuple(int(part.strip()) for part in value.strip("()").split(",") if part.strip())
