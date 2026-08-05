# -*- coding: utf-8 -*-
"""H-2：抽 1800 个**新的随机三元组**，把无偏认证池从 200 扩到 2000。

为什么
------
68 号的三阶认证集是 top200（估计器挑的，有偏）+ rand200（无偏）。
75_Correction §7.2 只能在 rand200 上做可外推的召回评估，而那里只有 **19 条**
真强三阶（syn3_true>0.1），Wilson 95% CI ≈ [0.75, 0.99] —— 撑不起论文主张。

本脚本从 C(44,3)=13244 里排除 68 号已认证的 397 个，用独立 master seed 随机抽 1800 个。
合并后无偏池 = 200 + 1800 = 2000，预计真强三阶条目从 19 增至约 150，CI 收到约 ±5%。

口径
----
必须与 68 号既有三阶真值完全同口径：同一份 `retrain_worker.py`（逐字复制自 67/68），
`--struct dnn --seed 0`，数据切分 seed 42。否则新旧真值不可合并。

输出：outputs/subsets.json（供 retrain_sched.py 调度）
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

R77 = Path(__file__).resolve().parents[1]
R69 = R77.parent / "DNN_Aggresvation69"
R68 = R77.parent / "DNN_Aggresvation68"
sys.path.insert(0, str(R69))
sys.path.insert(0, str(R77 / "src"))
from src.data_processing import prepare_data   # noqa: E402
from runlog import log                          # noqa: E402

MASTER_SEED = 7702        # 独立于 60(2026)/68(6808)/75(7501)
N_NEW = 1800


def main():
    cfg = yaml.safe_load((R77 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    general = di["general"]
    nG = len(general)
    name2idx = {n: i for i, n in enumerate(general)}

    # 排除 68 号已认证的 397 个（top197 + rand200）
    reg68 = json.load(open(R68 / "outputs/subsets.json"))
    used = set()
    for m in reg68.values():
        used.add(tuple(sorted(name2idx[f] for f in m["fields"])))
    print(f"68 号已认证 {len(used)} 个三元组，从 C({nG},3)={len(list(combinations(range(nG),3)))} 中排除")

    all_tri = [t for t in combinations(range(nG), 3) if t not in used]
    rng = np.random.RandomState(MASTER_SEED)
    pick = rng.choice(len(all_tri), size=min(N_NEW, len(all_tri)), replace=False)

    subsets = {}
    for t in sorted(all_tri[i] for i in pick):
        sid = "r%02d_%02d_%02d" % t          # r 前缀，与 68 号的 t 前缀区分
        subsets[sid] = {"group": "triple_rand_ext", "size": 3,
                        "fields": [general[j] for j in t]}

    out = R77 / "outputs/subsets.json"
    out.write_text(json.dumps(subsets, indent=1, ensure_ascii=False))
    print(f"写入 {len(subsets)} 个新随机三元组 -> {out.name}")
    print(f"合并后无偏池 = 200(68号 rand) + {len(subsets)} = {200 + len(subsets)}")
    log("H-2", "NOTE", note=f"抽样完成：{len(subsets)} 个新随机三元组"
                            f"（master seed {MASTER_SEED}，已排除 68 号 397 个）")


if __name__ == "__main__":
    main()
