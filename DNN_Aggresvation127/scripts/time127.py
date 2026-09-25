# -*- coding: utf-8 -*-
"""127 号：外部基线的每集合耗时（与 126 号同口径：同一空闲 GPU、每组同一批 200 个集合的前若干个、预热后计时）。
Dropout 用 200 个集合（批 8）；热启动 20 个；LazyVI 5 个（双精度）。TabPFN 另见 time127_tabpfn.py。输出 outputs/time_<组>.csv"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src")); import base127 as B  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--gpu", type=int, default=1); a = ap.parse_args(); dev = f"cuda:{a.gpu}"
sync = lambda: torch.cuda.synchronize(dev)
for tag, rel, _, _ in registry.DATASETS:
    T = tag.replace("/", "_"); D = registry.load_ds(rel)["D"]
    n3 = json.load(open(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_sets.json")))["n3"]
    M = np.load(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_masks.npy"))[:n3]
    M = M[np.random.RandomState(2).choice(n3, min(200, n3), replace=False)]
    ms = []
    for sd in range(3):
        m = B.MLP(D["Xtr"].shape[1], D["Ytr"].shape[1]).to(dev); m.load_state_dict(torch.load(os.path.join(ROOT, "outputs", "full", f"{T}_seed{sd}.pt"), map_location=dev)); ms.append(m.eval())
    rows = []
    for name, fn, k in [("dropout", lambda X: B.dropout_est(D, ms, X, dev, B=8), 200), ("ws", lambda X: B.ws_est(D, ms[0], X, dev), 20),
                        ("lazyvi", lambda X: B.lazy_est(D, ms[0], X, dev, dtype=torch.float64), 5)]:
        fn(M[:2]); sync(); t0 = time.time(); fn(M[:k]); sync(); rows.append(dict(数据=tag, 方法=name, 每集合ms=(time.time() - t0) / k * 1000))
        print(tag, name, f"{rows[-1]['每集合ms']:.1f} ms", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "outputs", f"time_{T}.csv"), index=False)
