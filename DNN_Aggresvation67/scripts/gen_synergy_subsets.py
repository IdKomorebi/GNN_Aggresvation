"""DNN67：生成协同验证所需的子集——44 个单字段 + 946 个字段对，写 outputs/subsets.json。
真值口径与 60 号一致（重训 DNN 攻击者，逐 conf R²）。
"""
from __future__ import annotations

import json, sys
from itertools import combinations
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_processing import prepare_data


def main():
    cfg = yaml.safe_load((ROOT / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(ROOT.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    general = di["general"]; nG = len(general)
    subsets = {}
    for i in range(nG):
        subsets[f"s{i:02d}"] = {"group": "single", "size": 1, "fields": [general[i]]}
    for i, j in combinations(range(nG), 2):
        subsets[f"p{i:02d}_{j:02d}"] = {"group": "pair", "size": 2,
                                        "fields": [general[i], general[j]]}
    (ROOT / "outputs/subsets.json").write_text(json.dumps(subsets, indent=1, ensure_ascii=False))
    n_single = sum(1 for v in subsets.values() if v["group"] == "single")
    n_pair = sum(1 for v in subsets.values() if v["group"] == "pair")
    print(f"写入 {len(subsets)} 个子集：{n_single} 单字段 + {n_pair} 对 (nG={nG})")


if __name__ == "__main__":
    main()
