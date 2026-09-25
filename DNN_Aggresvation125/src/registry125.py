# -*- coding: utf-8 -*-
"""125 号：与 117 号 registry 相同的数据集登记，只把估计文件换成本号固定的新估计器（outputs/final_est/<组>.npz，键 E）。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation117", "src"))
import registry as _r  # noqa: E402
FINAL = os.environ.get("FINAL125", os.path.join(HERE, "..", "outputs", "final_est"))   # 自检时可指向旧估计器
DATASETS = [(t, rel, os.path.abspath(os.path.join(FINAL, t.replace("/", "_") + ".npz")), "E") for t, rel, _, _ in _r.DATASETS]
load_ds = _r.load_ds
