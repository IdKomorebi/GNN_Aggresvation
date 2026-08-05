# -*- coding: utf-8 -*-
"""合并 build_subs 的分片为整块缓存（scan_highorder.py 只在 full 模式自动合并）。"""
import sys
from math import comb
from pathlib import Path

import numpy as np

R96 = Path(__file__).resolve().parents[2] / "DNN_Aggresvation96"
stem = sys.argv[1]                       # 例：oracle_l1_r93_seed0_last
n_gen, order = 44, 5
parts = sorted(R96.glob(f"outputs/subs_o{order}_{stem}_part*of*.npy"))
assert parts, f"没找到分片：subs_o{order}_{stem}_part*"
n_sub = comb(n_gen, order)
v = None
seen = np.zeros(n_sub, dtype=bool)
for p in parts:
    d = np.load(p)
    r = d[:, 0].astype(np.int64)
    if v is None:
        v = np.zeros((n_sub, d.shape[1] - 1), dtype=np.float32)
    v[r] = d[:, 1:]
    seen[r] = True
    print(f"  {p.name}: {len(r)} 条")
assert seen.all(), f"缓存不完整，缺 {int((~seen).sum())} 个"
out = R96 / f"outputs/subs_o{order}_{stem}.npy"
np.save(out, v)
print(f"[merge] {len(parts)} 分片 → {out.name}  {v.shape}")
