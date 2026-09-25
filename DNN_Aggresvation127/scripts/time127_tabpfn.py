# -*- coding: utf-8 -*-
"""127 号：TabPFN v2 每集合耗时（全部目标；每组同一批 200 个集合的前 20 个；预热后计时）。须用 tabpfn_venv 与 SCIPY_ARRAY_API=1。"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); a = ap.parse_args(); dev = f"cuda:{a.gpu}"; rows = []
reg = TabPFNRegressor(device=dev, random_state=0)
for tag, rel, _, _ in registry.DATASETS:
    T = tag.replace("/", "_"); d = registry.load_ds(rel); D, keys = d["D"], d["keys"]; k3 = [k for k in keys if len(k) <= 3]
    sets = [list(k3[i]) for i in np.random.RandomState(2).choice(len(k3), min(200, len(k3)), replace=False)][:20]; C = D["Ytr"].shape[1]
    reg.fit(D["Xtr"][:, sets[0]], D["Ytr"][:, 0]); reg.predict(D["Xte"][:, sets[0]])   # 预热
    torch.cuda.synchronize(dev); t0 = time.time()
    for S in sets:
        for c in range(C):
            reg.fit(D["Xtr"][:, S], D["Ytr"][:, c]); reg.predict(D["Xte"][:, S])
    torch.cuda.synchronize(dev); rows.append(dict(数据=tag, 方法="tabpfn", 每集合ms=(time.time() - t0) / len(sets) * 1000, 目标数=C))
    print(tag, f"{rows[-1]['每集合ms']:.0f} ms（{C} 个目标）", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs", "time_tabpfn.csv"), index=False)
