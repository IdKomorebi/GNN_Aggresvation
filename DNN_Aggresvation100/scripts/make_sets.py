# -*- coding: utf-8 -*-
"""生成 100 号全部集合文件：每个数据集规模 ≤4 的全部集合（41 个字段 → 112,791 个）；PJM 另加公开基底场景。"""
import sys, pickle
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from common100 import load, enumerate_sets, BASE_PJM
for ds in ["pjm", "caiso"]:
    D = load(ds); nG = len(D["general"])
    M, keys = enumerate_sets(D["active"], nG, 4)
    k3 = np.array([len(k) <= 3 for k in keys])
    np.save(ROOT / f"outputs/sets/{ds}_k4.npy", M); np.save(ROOT / f"outputs/sets/{ds}_k3.npy", M[k3])
    pickle.dump(dict(keys=keys, keys_k3=[k for k in keys if len(k) <= 3], general=D["general"], conf=D["conf"], active=D["active"]),
                open(ROOT / f"outputs/sets/{ds}_meta.pkl", "wb"))
    print(ds, "active", len(D["active"]), "k4", len(keys), "k3", int(k3.sum()), "rows tr/te", len(D["Xtr"]), len(D["Xte"]))
    if ds == "pjm":
        base = [D["general"].index(f) for f in BASE_PJM]
        Mb, kb = enumerate_sets(D["active"], nG, 3, base=base)
        np.save(ROOT / "outputs/sets/pjm_base.npy", Mb); pickle.dump(dict(keys=kb, base=base), open(ROOT / "outputs/sets/pjm_base_meta.pkl", "wb"))
        print("pjm_base", len(kb))
