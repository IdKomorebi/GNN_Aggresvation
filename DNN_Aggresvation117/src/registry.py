# -*- coding: utf-8 -*-
"""117–119 号共用：参与对比的数据集 / 目标登记。每项给出实验输出目录与新主口径 E 的估计文件。"""
import os, json
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATASETS = [  # (标签, 输出目录, E 估计文件, 键名)
    ("RTS-GMLC", "DNN_Aggresvation111/outputs", "est_variants.npz", "E"),
    ("NEM", "DNN_Aggresvation112/outputs", "est_variants.npz", "E"),
    ("PJM-load", "DNN_Aggresvation116/groups/pjm_load/outputs", "est_group.npz", "E"),
    ("PJM-gen/ic", "DNN_Aggresvation116/groups/pjm_gen_ic/outputs", "est_group.npz", "E"),
    ("CAISO-load", "DNN_Aggresvation116/groups/caiso_load/outputs", "est_group.npz", "E"),
]


def load_ds(rel, est_file=None, est_key="E"):
    O = os.path.join(REPO, rel)
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    z = np.load(os.path.join(O, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
    D = {k: z[k] for k in ["Xtr", "Ytr", "Xte", "Yte", "fit_idx", "val_idx"]}
    if not os.path.exists(os.path.join(O, "V_official.npy")):
        raise FileNotFoundError(O)
    V = np.load(os.path.join(O, "V_official.npy"))
    out = dict(O=O, spec=spec, D=D, keys=keys, V=V)
    if est_file and os.path.exists(os.path.join(O, est_file)):
        e = np.load(os.path.join(O, est_file)); out["sel3"] = e["sel"]; out["Vhat3"] = e[est_key]
    return out
