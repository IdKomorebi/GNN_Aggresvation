"""DNN68：补中段密集真值子集，写 outputs/subsets.json（供重训调度）。

60 号在 5–17 覆盖稀（9-13 仅 4 个）。这里每个尺寸带补足 ≥15 个随机子集，
重点 5–17，与 60/67 真值合并成跨尺寸密集评测集。master seed 独立可复现。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data

MASTER_SEED = 6808
# (尺寸带, 每带子集数)：中段密集，两端由 60 号已覆盖故少补
PLAN = [(5, 5, 15), (6, 9, 20), (10, 13, 20), (14, 17, 20), (18, 25, 10)]


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    general = di["general"]; nG = len(general)
    rng = np.random.RandomState(MASTER_SEED)
    subsets = {}
    sid = 0
    for lo, hi, n in PLAN:
        for _ in range(n):
            size = int(rng.randint(lo, hi + 1))
            idx = sorted(rng.choice(nG, size=size, replace=False).tolist())
            subsets[f"bt{sid:03d}"] = {"group": "band_truth", "size": size,
                                       "fields": [general[j] for j in idx]}
            sid += 1
    (ROOT / "outputs/subsets.json").write_text(json.dumps(subsets, indent=1, ensure_ascii=False))
    sizes = [s["size"] for s in subsets.values()]
    print(f"写入 {len(subsets)} 个中段密集子集，尺寸范围 [{min(sizes)},{max(sizes)}]")
    for lo, hi, _ in PLAN:
        c = sum(1 for z in sizes if lo <= z <= hi)
        print(f"  带[{lo}-{hi}]: {c}")


if __name__ == "__main__":
    main()
