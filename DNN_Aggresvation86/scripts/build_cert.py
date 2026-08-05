# -*- coding: utf-8 -*-
"""86 号认证：选结构化强四阶 + 随机参照，建 subsets.json，供 DNN 重训核对。

结构化查询是"攻击者下界"；本步用真正的 DNN 重训核对：结构化说强的四阶，
在重训真值下是否真强(syn4_true = v(quad) − max 4个三元子集 v)。复用 77 号 retrain_worker。
三元子集真值能复用 68 号的就复用（同 base.yaml/切分），只补缺的。
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
R69 = REPO / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402

N_TOP = 20     # 结构化 syn4 最强
N_RAND = 20    # 无偏随机参照


def main() -> None:
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(REPO / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    gen = list(di["general"])

    gt = pd.read_parquet(ROOT / "outputs/enum4_syn.parquet")
    gt["indices"] = gt["indices"].apply(lambda t: tuple(int(v) for v in t))
    gt = gt.sort_values("syn4", ascending=False).reset_index(drop=True)

    top = gt.head(N_TOP)["indices"].tolist()
    rng = np.random.RandomState(2026)
    rand_pool = gt.iloc[N_TOP:]["indices"].tolist()
    rand = [rand_pool[i] for i in rng.choice(len(rand_pool), N_RAND, replace=False)]
    quads = [(q, "top") for q in top] + [(q, "rand") for q in rand]

    subsets = {}
    quad_map = {}

    def sid_of(fields_idx):
        key = tuple(sorted(fields_idx))
        sid = "s" + "_".join(f"{i:02d}" for i in key)
        if sid not in subsets:
            subsets[sid] = {"group": f"cert{len(key)}", "size": len(key),
                            "fields": [gen[i] for i in key]}
        return sid

    for quad, tag in quads:
        qid = sid_of(quad)
        trip_ids = [sid_of([a for a in quad if a != d]) for d in quad]
        quad_map[qid] = {"tag": tag, "quad": list(quad),
                         "syn4_struct": float(gt[gt.indices == quad].syn4.iloc[0]),
                         "triples": trip_ids}

    (ROOT / "outputs/subsets.json").write_text(json.dumps(subsets, indent=1, ensure_ascii=False))
    (ROOT / "outputs/quad_map.json").write_text(json.dumps(quad_map, indent=1, ensure_ascii=False))
    print(f"待认证四元组 {len(quads)} 个 (top{N_TOP}+rand{N_RAND})；"
          f"需重训子集(含三元) {len(subsets)} 个 → outputs/subsets.json")


if __name__ == "__main__":
    main()
