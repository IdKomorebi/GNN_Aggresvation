#!/usr/bin/env python3
"""DNN60 子集生成：Phase 0 的两组子集，确定性生成，写入 outputs/subsets.json。

1. 噪声底组 nf00..nf09：10 个尺寸覆盖 {1,2,4,8,12,16,24,32,40,44} 的随机子集，
   每个子集后续用 5 个训练种子重训，测 v_retrain 的种子标准差（噪声底）。
2. 无偏评测组 rs00..rs49：50 个随机子集，按尺寸 5 个分层带各 10 个，
   均匀覆盖子集空间（59 号的 110 个真值点全是 top-k 结构，有排名偏置）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data

MASTER_SEED = 2026
NF_SIZES = [1, 2, 4, 8, 12, 16, 24, 32, 40, 44]
RS_BANDS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 44)]
RS_PER_BAND = 10


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    general = di["general"]
    n_general = di["n_general"]
    assert n_general == 44, f"预期 44 个 general 字段，实际 {n_general}"

    rng = np.random.RandomState(MASTER_SEED)
    subsets = {}

    for i, size in enumerate(NF_SIZES):
        idx = sorted(rng.choice(n_general, size=size, replace=False).tolist())
        subsets[f"nf{i:02d}"] = {
            "group": "noise_floor",
            "size": size,
            "fields": [general[j] for j in idx],
        }

    sid = 0
    for lo, hi in RS_BANDS:
        for _ in range(RS_PER_BAND):
            size = int(rng.randint(lo, hi + 1))
            idx = sorted(rng.choice(n_general, size=size, replace=False).tolist())
            subsets[f"rs{sid:02d}"] = {
                "group": "random_eval",
                "size": size,
                "fields": [general[j] for j in idx],
            }
            sid += 1

    out = ROOT / "outputs/subsets.json"
    out.write_text(json.dumps(subsets, indent=2, ensure_ascii=False))
    sizes = [s["size"] for s in subsets.values()]
    print(f"共 {len(subsets)} 个子集写入 {out}")
    print(f"  噪声底组: {len(NF_SIZES)} 个，尺寸 {NF_SIZES}")
    print(f"  随机评测组: {sid} 个，尺寸范围 [{min(sizes)}, {max(sizes)}]")


if __name__ == "__main__":
    main()
