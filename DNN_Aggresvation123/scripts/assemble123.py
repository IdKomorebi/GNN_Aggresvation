# -*- coding: utf-8 -*-
"""123 号 步骤 2：组装字段（11 个，全部 2,047 个子集可精确重训）与目标 κ_t（机组 313_CC_1 的私有报价加成）。"""
import os, sys, json, importlib.util
import numpy as np, pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(REPO, "DNN_Aggresvation111", "src")); import pipe  # noqa: E402
spec = importlib.util.spec_from_file_location("rts111", os.path.join(REPO, "DNN_Aggresvation111/scripts/build_rts.py"))
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
TAG = os.environ.get("TAG123", "wide"); OUTD = os.path.join(ROOT, "outputs", TAG)
z = np.load(os.path.join(OUTD, "dispatch123.npz")); h = z["hours"]; L = z["LMP"]; PU = z["PU"]; BIND = z["BIND"]
ids = list(R.B); th = R.th; unit = lambda n: int(th.index[th["GEN UID"] == n][0])
typ = R.rn["Unit Type"].values
df = pd.DataFrame({
    "P_g (unit 313 output)": PU[:, unit("313_CC_1")],
    "LMP_g (bus 313)": L[:, ids.index(313)],
    "LMP_near (bus 306)": L[:, ids.index(306)],
    "LMP_far (bus 303, behind C6)": L[:, ids.index(303)],
    "LMP_area2 (bus 223)": L[:, ids.index(223)],
    "System load": z["LOAD"],
    "Area-3 load": R.load_rt[h, 2],
    "Wind available": R.avail[h][:, typ == "WIND"].sum(1),
    "PV available": R.avail[h][:, np.isin(typ, ["PV", "RTPV"])].sum(1),
    "Congestion C6": BIND[:, int(np.where(R.br["UID"].values == "C6")[0][0])].astype(float),
    "P_comp (unit 316 output)": PU[:, unit("316_STEAM_1")],
})
cand = list(df.columns); df["Y_markup"] = z["KAPPA"]
df.to_csv(os.path.join(OUTD, "dataset.csv"), index=False)
D = pipe.prep(df, cand, ["Y_markup"]); masks, keys = pipe.enumerate_sets(len(cand), len(cand))
np.savez(os.path.join(OUTD, "D.npz"), **{k: v for k, v in D.items() if isinstance(v, np.ndarray)},
         masks=masks, keys=np.array(["|".join(map(str, k)) for k in keys]))
json.dump(dict(dataset="RTS-GMLC 受控机制算例：机组报价加成推断", cand=cand, targ=["Y_markup"], kmax=len(cand), taus=[0.5, 0.7]),
          open(os.path.join(OUTD, "fields.json"), "w"), ensure_ascii=False, indent=1)
print(f"样本 {D['n']}，候选 {len(cand)}，集合 {len(keys)}；C6 阻塞小时 {df['Congestion C6'].mean():.3f}")
print(df.corr()["Y_markup"].round(3).to_string())
