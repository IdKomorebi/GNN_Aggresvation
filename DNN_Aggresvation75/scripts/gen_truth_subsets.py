# -*- coding: utf-8 -*-
"""DNN75 Phase 0：补大集合重训真值子集。

为什么要补
----------
73/74 用的 394 子集评测集里 **89% 是 |S|<=3**，中大规模带每带只有 10 个子集
（全部来自 60 号的 50 个 wide_random）。而本子项目（D1，采样测度改进）完全是关于
**尺寸分配**的——用那个评测集根本分辨不出"小集合变好、大集合变差"这个权衡。

把 68 号现成的 85 个中段真值（bt*.json，尺寸 5-25）接回来之后，5-16 带够用了，
但 **33-44 带仍只有 10 个随机子集**，而防护集管线恰好密集查询 |S|∈[13,44]
（72 号 direct_greedy 每步评估 |共享集合| = 44-k-1，k 从 0 到 30）。

因此在 17-24 / 25-32 / 33-38 / 39-43 四带各补 15 个随机子集。

口径
----
与 59/60/67/68 的真值完全一致：DNN 攻击者从头重训、数据切分 seed 42、训练 seed 0。
（注意：`retrain_worker.train_generic` 用测试集做 early-stop，这是 72Supplement 已指出的
已知问题；但这里**必须沿用同一协议**，否则新真值与既有 1437 个真值不可比。
本评测集是【测量尺度】不是【对外证书】，一致性优先。）

输出：outputs/subsets.json（供 retrain_sched.py 调度）
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

R75 = Path(__file__).resolve().parents[1]
R69 = R75.parent / "DNN_Aggresvation69"
sys.path.insert(0, str(R69))
from src.data_processing import prepare_data  # noqa: E402

MASTER_SEED = 7501                      # 独立于 60(2026) / 68(6808) 的 master seed
PLAN = [(17, 24, 15), (25, 32, 15), (33, 38, 15), (39, 43, 15)]


def main():
    cfg = yaml.safe_load((R75 / "base.yaml").read_text())
    cfg["dataset"]["csv_path"] = str(R69.parent / "data/Processed/pjm_rto_hourly_2025_cleaned.csv")
    di = prepare_data(cfg)
    general = di["general"]
    nG = len(general)

    rng = np.random.RandomState(MASTER_SEED)
    subsets, sid = {}, 0
    seen = set()
    for lo, hi, n in PLAN:
        got = 0
        while got < n:
            size = int(rng.randint(lo, hi + 1))
            idx = tuple(sorted(rng.choice(nG, size=size, replace=False).tolist()))
            if idx in seen:                       # 去重，避免同一集合重复重训
                continue
            seen.add(idx)
            subsets[f"lg{sid:03d}"] = {"group": "large_random", "size": size,
                                       "fields": [general[j] for j in idx]}
            sid += 1
            got += 1

    out = R75 / "outputs/subsets.json"
    out.write_text(json.dumps(subsets, indent=1, ensure_ascii=False))
    sizes = sorted(s["size"] for s in subsets.values())
    print(f"写入 {len(subsets)} 个大集合子集 -> {out.name}，尺寸 [{sizes[0]},{sizes[-1]}]")
    for lo, hi, _ in PLAN:
        print(f"  带[{lo:>2}-{hi:>2}]: {sum(1 for z in sizes if lo <= z <= hi)} 个")
    print(f"  预计重训耗时: {len(subsets)} × ~6.5min / 4 卡 ≈ {len(subsets)*6.5/4/60:.1f} h")


if __name__ == "__main__":
    main()
