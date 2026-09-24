# -*- coding: utf-8 -*-
"""Read-only data adapters retained from Claude V4_en/scripts/make_figs.py.
Used by the Codex V1 figure redesign; no training or experiment writes.
"""
import os, sys, json
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import style as S
import names as N

S.setup()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "figures")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
GROUPS = [("RTS-GMLC", "DNN_Aggresvation111/outputs"), ("NEM", "DNN_Aggresvation112/outputs"),
          ("PJM", "DNN_Aggresvation116/groups/pjm_load/outputs"), ("PJM", "DNN_Aggresvation116/groups/pjm_gen_ic/outputs"),
          ("CAISO", "DNN_Aggresvation116/groups/caiso_load/outputs")]
DSCOL = {"RTS-GMLC": S.BLUE, "NEM": S.ORANGE, "PJM": S.AQUA, "CAISO": S.VIOLET}
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#dbe9f8", "#9fc3ea", "#3987e5", "#1c5cab", "#0d366b"])


def R(*p):
    return os.path.join(REPO, *p)

def summaries():
    rows = []
    for ds, rel in GROUPS:
        f = R(rel, "analysis", "summary_targets.csv")
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        for _, r in pd.read_csv(f).iterrows():
            d = r.to_dict(); d.update(ds=ds, rel=rel, p=len(spec["cand"]), name=N.target(r["目标"])); rows.append(d)
    return pd.DataFrame(rows)

def max_v3(r):
    z = np.load(R(r.rel, "D.npz")); keys = [k for k in z["keys"]]; V = np.load(R(r.rel, "V_official.npy"))
    spec = json.load(open(R(r.rel, "fields.json"), encoding="utf-8")); c = [t.replace("Y_", "") for t in spec["targ"]].index(r["目标"])
    s3 = np.array([k.count("|") <= 2 for k in keys]); return float(V[s3, c].max())

def _est_sets():
    sys.path.insert(0, R("DNN_Aggresvation111", "src")); import pipe
    out = []
    for (ds, rel), (_, _, ef, ek) in zip(GROUPS, [("", "", "est_variants.npz", "E"), ("", "", "est_variants.npz", "E"),
                                                  ("", "", "est_group.npz", "E"), ("", "", "est_group.npz", "E"), ("", "", "est_group.npz", "E")]):
        if not os.path.exists(R(rel, "V_official.npy")) or not os.path.exists(R(rel, ef)):
            raise FileNotFoundError(f"Missing reference or estimate arrays in {R(rel)}")
        z = np.load(R(rel, "D.npz")); keys = [tuple(int(x) for x in k.split("|")) for k in z["keys"]]
        e = np.load(R(rel, ef)); s3 = e["sel"]; k3 = [k for k, t in zip(keys, s3) if t]
        V = np.load(R(rel, "V_official.npy"))[s3]; Ve = pipe.closure_max(e[ek], k3); p = z["Xtr"].shape[1]
        _, _, Dt, _, bsz = pipe.m_table(V, k3, list(range(p)), 2); _, _, De, _, _ = pipe.m_table(Ve, k3, list(range(p)), 2)
        spec = json.load(open(R(rel, "fields.json"), encoding="utf-8"))
        out.append(dict(ds=ds, V=V, Ve=Ve, Dt=Dt, De=De, bsz=bsz, targ=spec["targ"], pipe=pipe))
    return out
