# -*- coding: utf-8 -*-
"""125 号：把选定候选的规模 ≤3 估计写成下游可读的文件（与各号 est_*.npz 同格式：sel、E）。
用法：make_final.py <来源 122|125> <变体名> <输出目录>"""
import os, sys, json
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src")); import registry  # noqa: E402
src, v, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
for tag, rel, _, _ in registry.DATASETS:
    T = tag.replace("/", "_"); d = registry.load_ds(rel); keys = d["keys"]; sel = np.array([len(k) <= 3 for k in keys])
    n3 = json.load(open(os.path.join(REPO, "DNN_Aggresvation122", "outputs", "est", T, "eval_sets.json")))["n3"]; assert n3 == sel.sum()
    base = os.path.join(REPO, "DNN_Aggresvation122" if src == "122" else "DNN_Aggresvation125", "outputs", "est", T, f"{v}.npz")
    z = np.load(base); np.savez(os.path.join(out, f"{T}.npz"), sel=sel, E=z["est"][:n3], sec=z["sec"], source=f"{src}:{v}")
    print(tag, "→", os.path.join(out, f"{T}.npz"))
