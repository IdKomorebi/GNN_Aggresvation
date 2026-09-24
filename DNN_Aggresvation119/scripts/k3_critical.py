# -*- coding: utf-8 -*-
"""119 号 步骤 3：K=2 与 K=3 的关键字段对照——只有在 3 个背景下才越阈的字段（属于规模为 4 的最小不安全集、但不属于任何规模 ≤3 的）。"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
rows = []
SRC = [("RTS-GMLC", os.path.join(REPO, "DNN_Aggresvation111/outputs"), None), ("NEM", os.path.join(REPO, "DNN_Aggresvation112/outputs"), "nem"),
       ("PJM-load", os.path.join(REPO, "DNN_Aggresvation116/groups/pjm_load/outputs"), None), ("PJM-gen/ic", os.path.join(REPO, "DNN_Aggresvation116/groups/pjm_gen_ic/outputs"), None),
       ("CAISO-load", os.path.join(REPO, "DNN_Aggresvation116/groups/caiso_load/outputs"), None)]
for tag, O, special in SRC:
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    if special == "nem":
        k3 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O, "D.npz"))["keys"]]
        k4 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(ROOT, "groups/nem_k4/outputs/D.npz"))["keys"]]
        keys = k3 + k4; V = np.load(os.path.join(ROOT, "groups/nem_k4/outputs/V_official_k4.npy"))
    else:
        keys = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O, "D.npz"))["keys"]]; V = np.load(os.path.join(O, "V_official.npy"))
    p = len(spec["cand"])
    for c, y in enumerate(spec["targ"]):
        for tau in (0.5, 0.7):
            m3 = pipe.mus_list(V, keys, list(range(p)), c, tau, 3); m4 = pipe.mus_list(V, keys, list(range(p)), c, tau, 4)
            c3 = {i for m in m3 for i in m}; c4 = {i for m in m4 for i in m}
            rows.append(dict(数据=tag, 目标=y.replace("Y_", ""), τ=tau, 字段数=p, K2关键=len(c3), K3关键=len(c4), 仅K3关键=len(c4 - c3),
                             规模4最小不安全集=sum(1 for m in m4 if len(m) == 4), 规模4最大V=float(V[[len(k) == 4 for k in keys], c].max())))
R = pd.DataFrame(rows); R.to_csv(os.path.join(ROOT, "outputs", "analysis", "k3_critical.csv"), index=False); print(R.round(3).to_string(index=False))

# ---- 处置影响：按 K=2（规模 ≤3 的 MUS）求最小命中集扣留后，规模 4 的 MUS 还剩多少；以及按 ≤4 求的最小命中集大小
from scipy.optimize import milp, LinearConstraint, Bounds


def hitset(mus, p):
    if not mus:
        return set()
    Am = np.zeros((len(mus), p))
    for r, m in enumerate(mus):
        Am[r, list(m)] = 1
    return set(np.where(milp(c=np.ones(p), constraints=LinearConstraint(Am, lb=1), integrality=np.ones(p), bounds=Bounds(0, 1)).x > .5)[0])


rows2 = []
for tag, O, special in SRC:
    spec = json.load(open(os.path.join(O, "fields.json"), encoding="utf-8"))
    if special == "nem":
        k3 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O, "D.npz"))["keys"]]
        k4 = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(ROOT, "groups/nem_k4/outputs/D.npz"))["keys"]]
        keys = k3 + k4; V = np.load(os.path.join(ROOT, "groups/nem_k4/outputs/V_official_k4.npy"))
    else:
        keys = [tuple(int(x) for x in k.split("|")) for k in np.load(os.path.join(O, "D.npz"))["keys"]]; V = np.load(os.path.join(O, "V_official.npy"))
    p = len(spec["cand"])
    for c, y in enumerate(spec["targ"]):
        for tau in (0.5, 0.7):
            m4 = pipe.mus_list(V, keys, list(range(p)), c, tau, 4); m3 = [m for m in m4 if len(m) <= 3]; only4 = [m for m in m4 if len(m) == 4]
            W3, W4 = hitset(m3, p), hitset(m4, p)
            rows2.append(dict(数据=tag, 目标=y.replace("Y_", ""), τ=tau, 规模4MUS=len(only4), K2命中集大小=len(W3), K2命中集后残余规模4MUS=sum(1 for m in only4 if not set(m) & W3),
                              K3命中集大小=len(W4)))
R2 = pd.DataFrame(rows2); R2.to_csv(os.path.join(ROOT, "outputs", "analysis", "k3_disposal.csv"), index=False); print(R2.to_string(index=False))
